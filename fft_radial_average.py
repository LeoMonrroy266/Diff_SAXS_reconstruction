#!/usr/bin/env python3
# coding=utf-8
"""
Radially averaged scattering from a 3-D voxel array
--------------------------------------------------
* voxels: 3-D NumPy array (float32 or float64)
  • value = scattering-length density contrast at each voxel
  • origin of reciprocal grid is at index (0,0,0) in NumPy’s FFT convention
* output: two 1-D arrays (q values, I(q))
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def forward_FFT(voxel_obj):
# Forward FFT  → complex amplitude F(q)
    Fq = np.fft.fftn(voxel_obj)          # complex ndarray, same shape
    I_reciprocal = np.abs(Fq) ** 2
    return I_reciprocal

import numpy as np

def radial_average_3d(I_recip,
                      voxel_size=3.2,
                      q_bins="auto",
                      return_q=True):
    """
    Radially average a 3-D reciprocal-space intensity map.

    Parameters
    ----------
    I_recip : ndarray, shape (Nx, Ny, Nz)
        Real-valued intensities |F(q)|² on the FFT grid.
    voxel_size : float, optional
        Edge length of one real-space voxel (Å, nm, …).
        The q-axis is returned in reciprocal units of
        2π / voxel_size by default.
    q_bins : "auto" | int | 1-D array, optional
        • "auto"  – use the natural FFT spacing (Δq = 2π / (N · voxel_size)).
        • int     – number of equally spaced bins up to q_max.
        • array   – explicit bin edges (in same q units).
    return_q : bool, optional
        If True (default) return (q, I_q);
        otherwise return I_q only.

    Returns
    -------
    q : 1-D ndarray
        Bin centres |q|  (omitted if return_q == False).
    I_q : 1-D ndarray
        Orientationally averaged intensity for each q bin.
    """
    I_recip = np.asarray(I_recip, dtype=np.float64)
    nx, ny, nz = I_recip.shape
    if not (nx == ny == nz):
        raise ValueError("Grid must be cubic for isotropic radial averaging")

    # ------------------------------------------------------------------
    # Build the |q| array corresponding to NumPy's FFT ordering
    # ------------------------------------------------------------------
    freq = np.fft.fftfreq(nx, d=voxel_size)    # cycles per unit length
    q_axis = 2 * np.pi * freq                  # reciprocal-space units (rad⁻¹)

    Qx, Qy, Qz = np.meshgrid(q_axis,
                             q_axis,
                             q_axis,
                             indexing='ij')
    q_mod = np.sqrt(Qx**2 + Qy**2 + Qz**2).ravel()
    I_flat = I_recip.ravel()

    # ------------------------------------------------------------------
    # Decide bin edges
    # ------------------------------------------------------------------
    q_max = q_mod.max()
    if isinstance(q_bins, str) and q_bins == "auto":
        dq = 2 * np.pi / (nx * voxel_size)
        n_bins = int(np.ceil(q_max / dq))
        edges = np.linspace(0, q_max, n_bins + 1)
    elif isinstance(q_bins, int):
        edges = np.linspace(0, q_max, q_bins + 1)
    else:  # assume array-like of explicit edges
        edges = np.asarray(q_bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("q_bins must define at least two edges")

    # ------------------------------------------------------------------
    # Histogram: sum intensities and counts per shell
    # ------------------------------------------------------------------
    I_sum, _ = np.histogram(q_mod, bins=edges, weights=I_flat)
    counts, _ = np.histogram(q_mod, bins=edges)
    mask = counts > 0                       # avoid divide-by-zero
    I_q = I_sum[mask] / counts[mask]        # average intensity
    if not return_q:
        return I_q

    q_centres = 0.5 * (edges[:-1] + edges[1:])[mask]
    return q_centres, I_q

def save_and_plot_iq(q, Iq, out_txt="Iq_radial.dat", out_png="Iq_radial.png", title="Radially averaged scattering", show_plot=True, use_loglog=False):
    """
    Save (q, I(q)) to a text file and generate a figure.

    Parameters
    ----------
    q, Iq : 1-D array-likes
        The q-vector and corresponding intensities.
    out_txt : str or Path
        Filename for the ASCII table (two columns).
    out_png : str or Path
        Filename for the rendered plot.
    title : str
        Plot title.
    show_plot : bool, default True
        If True, display the figure in an interactive window.
    use_loglog : bool, default True
        If True, plot on log–log axes; else linear axes.
    """
    q = np.asarray(q, dtype=float)
    Iq = np.asarray(Iq, dtype=float)

    # --------------------------------------------------
    # 1.  Save the data
    # --------------------------------------------------
    out_txt = Path(out_txt)
    header = f"# q\tI(q)\n# Saved by save_and_plot_iq\n"
    np.savetxt(out_txt, np.column_stack((q, Iq)), header=header, delimiter=',')
    print(f"Saved {len(q)} points → {out_txt}")

    # --------------------------------------------------
    # 2.  Make the figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=150)
    if use_loglog:
        ax.loglog(q, Iq, marker="o", ls="", ms=3)
    else:
        ax.plot(q, Iq)

    ax.set_xlabel(r"$q$ ( Å⁻¹)")
    ax.set_ylabel(r"$I(q)$ (arb. units)")
    ax.set_title(title)
    ax.set_xlim([0,0.2])
    ax.grid(True, which="both", lw=0.3)

    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, transparent=True)
    print(f"Plot written → {out_png}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)