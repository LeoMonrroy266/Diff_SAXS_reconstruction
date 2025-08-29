# -*- coding: utf-8 -*-
"""
map2iq.py
------------------------------------------------------------------
Reads and parses SAXS data
Calculates SAXS profiles from voxel objects.
FFT-based I(q) calculation + scoring
"""


import numpy as np
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
try:
    from scipy import ndimage  # optional (used only for gaussian_filter)
    _have_scipy = True
except ModuleNotFoundError:
    _have_scipy = False


@dataclass
class IqData:
    q: np.ndarray
    i: np.ndarray
    s: np.ndarray = None

def read_iq_ascii(fname: str | Path) -> IqData:
    """Read three-column ASCII: q  I(q)  sigma (σ optional)."""
    arr = np.loadtxt(fname)
    if arr.shape[1] == 2:
        q, i = arr.T
        s = np.ones_like(i)
    elif arr.shape[1] >= 3:
        q, i, s = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        raise ValueError("Expect 2 or 3 columns: q  I  [σ]")
    return IqData(q=q, i=i, s=s)


# ─────────────────────── FFT → radial average (q in Å⁻¹) ──────────────────────
def fft_intensity_scale_factor(voxel: np.ndarray, voxel_size: float = 3.33) -> tuple[np.ndarray, np.ndarray]:
    """
        Compute spherically averaged scattering intensity I(q) from a 3D real voxel grid.

        Parameters
        ----------
        voxel      : 3D ndarray
        voxel_size : float, size of each voxel in Å

        Returns
        -------
        q  : 1D array of q-values in Å⁻¹
        Iq : 1D array of intensity values
        """
    Fq = np.fft.fftn(voxel)

    I3d = np.abs(Fq) ** 2
    N = voxel.shape[0]

    freqs = np.fft.fftfreq(N, d=voxel_size)
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    qgrid = 2 * np.pi * np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    q_flat = qgrid.ravel()
    I_flat = I3d.ravel()

    dq = 0.002
    nbins = int(np.ceil(q_flat.max() / dq))
    bins = np.linspace(0, q_flat.max(), nbins + 1)

    idx = np.clip(np.digitize(q_flat, bins) - 1, 0, nbins - 1)
    I_sum = np.bincount(idx, weights=I_flat, minlength=nbins)
    N_sum = np.bincount(idx, minlength=nbins)

    valid = N_sum > 0
    I_rad = I_sum[valid] / N_sum[valid]

    q_mid = 0.5 * (bins[1:] + bins[:-1])
    q_rad = q_mid[valid]

    return q_rad, I_rad
def fft_intensity(voxel: np.ndarray, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spherically averaged scattering intensity I(q) from a 3D real voxel grid.

    Parameters
    ----------
    voxel : 3D ndarray
        Density map in a cubic voxel grid.
    rmax  : float
        Physical radius in Å that the cube represents (half the box size).

    Returns
    -------
    q  : 1D array of q-values in Å⁻¹
    Iq : 1D array of intensity values
    """
    N = voxel.shape[0]
    voxel_size = (2 * rmax) / N  # Å per voxel

    Fq = np.fft.fftn(voxel)
    I3d = np.abs(Fq) ** 2

    freqs = np.fft.fftfreq(N, d=voxel_size)
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    qgrid = 2 * np.pi * np.sqrt(kx**2 + ky**2 + kz**2)

    q_flat = qgrid.ravel()
    I_flat = I3d.ravel()

    dq = 2 * np.pi / (2 * rmax)  # reasonable spacing ~ fundamental frequency

    nbins = int(np.ceil(q_flat.max() / dq))
    bins = np.linspace(0, q_flat.max(), nbins + 1)

    idx = np.clip(np.digitize(q_flat, bins) - 1, 0, nbins - 1)
    I_sum = np.bincount(idx, weights=I_flat, minlength=nbins)
    N_sum = np.bincount(idx, minlength=nbins)

    valid = N_sum > 0
    I_rad = I_sum[valid] / N_sum[valid]
    q_mid = 0.5 * (bins[1:] + bins[:-1])
    q_rad = q_mid[valid]

    return q_rad, I_rad

# ───────────────────────── least-square scale ──────────────────────
def lsq_scale(calc: np.ndarray, exp: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Linear least-squares: scale*calc + offset ≈ exp."""
    w   = 1.0 / (sigma**2)
    A   = np.vstack([calc, np.ones_like(calc)]).T
    Aw  = A * w[:, None]
    params, *_ = np.linalg.lstsq(Aw, exp*w, rcond=None)
    scale, off = params
    return float(scale), float(off)


# ───────────────────────── ED_map class ────────────────────────────
class ED_map:
    """
    • compute_saxs_profile(): FFT → radial average
    • target(): χ²-like deviation metric
    """
    def __init__(self,
                 iq_data: IqData,
                 rmax: float,
                 voxel_size: float = 3.33,
                 qmax: float = 0.2):

        self.exp  = iq_data
        self.rmax  = rmax
        self.voxel_size = voxel_size
        # keep only q ≤ qmax
        sel = iq_data.q <= qmax
        self.exp_q = iq_data.q[sel]
        self.exp_I = iq_data.i[sel]
        self.exp_s = iq_data.s[sel]


    # ------------- public API ------------------
    def compute_saxs_profile(self, voxel: np.ndarray, save_plot: bool = False,
                             filename: str = "saxs_profile.png") -> np.ndarray:
        """
        Calculates the difference SAXS profile of generated voxel object and assumed starting structure
        (generated voxel - starting structure).

        Parameters
        ----------
        voxel : np.ndarray
        save_plot : Bool, optional
        filename : str, optional

        Returns
        -------
        Difference SAXS curve interpolated to experimental q-range
        """
        q, Iq = fft_intensity(voxel, rmax=self.rmax)

        #dark_q, dark_I = fft_intensity(self.dark, voxel_size=self.voxel_size)  # calc scatter for dark

        #scale = self.best_scaling_factor(Iq, dark_I)


        # Interpolate difference onto experimental q grid
        Iq_interpolate = np.interp(self.exp_q, q, Iq)


        #if save_plot:
        #    print(scale)
        #    plt.figure(figsize=(8, 5))
        #    plt.plot(q, Iq, label='Voxel Model I(q)')
        ##    plt.plot(dark_q, dark_I, label='Dark Model I(q)')
        #   plt.xlabel('q (Å$^{-1}$)')
        #    plt.ylabel('Intensity I(q)')
        #    plt.title('SAXS Profiles')
        #    plt.legend()
        #    plt.tight_layout()
        #    plt.savefig(filename)
        #    plt.close()

        return Iq_interpolate


    def target(self, voxel: np.ndarray) -> float:
        """
        Calculate the R² and Chi² between computed and experimental difference curves.
        Parameters
        ----------
        voxel : ndarray

        Returns
        -------
        r2 : float
        """

        calc = self.compute_saxs_profile(voxel)
        # Use unit weights if no experimental error is given
        if self.exp_s is None:
            weights = np.ones_like(calc)
        else:
            weights = self.exp_s

        # Apply least-squares scaling

        #scale, offset = lsq_scale(calc, self.exp_I,self.exp_s)
        #calc = (scale * calc) # Scale to exp.
        calc = calc/calc[0]
        self.exp_I = self.exp_I/self.exp_I[0]
        # Score depending on availability of exp_s
        if self.exp_s is None:
            # Use plain L2 distance
            chi = np.linalg.norm(calc - self.exp_I)
            r2 = r2_score(self.exp_I, calc)
        else:
            # Use chi-squared
            #chi = np.linalg.norm((calc - self.exp_I) / self.exp_s)
            chi = np.linalg.norm(calc - self.exp_I)
            r2 = r2_score(self.exp_I, calc)
        print('R²:', r2, 'chi²:', chi)
        #return float(r2)
        return float(chi)
# ─────────────────── wrappers for your GA code ────────────────────
def run_withrmax(voxel: np.ndarray,
                 iq_file: IqData | Path,
                 rmax_center: float,
                 voxel_size: float = 3.33) -> float:
    voxel = voxel.reshape((31,31,31))
    data = iq_file
    ed   = ED_map(data, rmax_center, voxel_size=voxel_size)
    return ed.target(voxel.reshape(-1))

def run_withoutrmax(voxel_and_rmax: np.ndarray,
                    iq_file: IqData | Path,
                    search_span: int = 6,
                    step: int = 3,
                    voxel_size: float = 3.33):
    """
    voxel_and_rmax:  (Nvox+1)  last element is rmax guess
    """

    # Last element is rmax, the rest is flattened voxel
    voxel_flat = voxel_and_rmax[:-1]
    rmax = voxel_and_rmax[-1]
    voxel = voxel_flat.reshape(31, 31, 31)
    guess = rmax
    data = iq_file
    best_d, best_r = 1e9, guess
    for r in range(int(guess-search_span), int(guess+search_span)+1, step):
        if r <= 10: continue
        d = ED_map(data, r, voxel_size=voxel_size).target(voxel)
        if d < best_d: best_d, best_r = d, r
    return best_d, best_r
