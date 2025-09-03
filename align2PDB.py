#!/usr/bin/env python3
"""
align_map_to_pdb_3d_refined.py
---------------------------------------------
Align a map PDB onto a full-atom PDB using 3D rotation + FFT translation,
with fine rotation refinement, sub-voxel translation, and optional voxel output.

Now includes Phenix-style masked CC calculation.

Dependencies:
- numpy
- scipy
- biopython
"""

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from scipy.ndimage import gaussian_filter
from numpy.fft import fftn, ifftn, fftshift
import itertools
import sys
from scipy.stats import pearsonr

# -----------------------------
# 1. Load PDB atoms
# -----------------------------
def parse_pdb_atoms(pdbfile):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdbfile)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atoms.append(atom.coord)
    return np.array(atoms)


# -----------------------------
# 2. Convert coords to voxel grid
# -----------------------------
def coords_to_grid(coords, grid_size=31, box_size=100.0, sigma=1.0):
    grid = np.zeros((grid_size, grid_size, grid_size))
    scale = (grid_size - 1) / box_size
    coords_centered = coords - coords.mean(axis=0) + (box_size / 2)
    indices = np.round(coords_centered * scale).astype(int)
    indices = np.clip(indices, 0, grid_size-1)
    np.add.at(grid, tuple(indices.T), 1)
    return gaussian_filter(grid, sigma=sigma)

# -----------------------------
# 3. Pad moving map grid to match target grid
# -----------------------------
def pad_grid_to_match(grid_moving, grid_target):
    shape_target = np.array(grid_target.shape)
    shape_moving = np.array(grid_moving.shape)
    pad_before = ((shape_target - shape_moving) // 2).astype(int)
    pad_after = (shape_target - shape_moving - pad_before).astype(int)
    padding = tuple((b, a) for b, a in zip(pad_before, pad_after))
    return np.pad(grid_moving, padding, mode='constant', constant_values=0)

# -----------------------------
# 4. FFT cross-correlation
# -----------------------------
def fft_cross_correlation(grid_target, grid_moving):
    grid_moving_padded = pad_grid_to_match(grid_moving, grid_target)
    f_target = fftn(grid_target)
    f_moving = fftn(grid_moving_padded)
    cross_corr = fftshift(ifftn(f_target * np.conj(f_moving)).real)
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    center = np.array(cross_corr.shape) // 2
    shift_vox = np.array(max_idx) - center
    norm_cc = cross_corr.max() / (np.linalg.norm(grid_target) * np.linalg.norm(grid_moving_padded))
    return shift_vox, norm_cc

# -----------------------------
# 5. Apply voxel shift to coordinates
# -----------------------------
def apply_shift(coords, shift_vox, grid_size_target, grid_size_moving, box_size=100.0):
    pad_offset = (grid_size_target - grid_size_moving) / 2
    total_shift = shift_vox - pad_offset
    voxel_size = box_size / (grid_size_target - 1)
    return coords + total_shift * voxel_size

# -----------------------------
# 6. Rotate coordinates
# -----------------------------
def rotate_coords(coords, angles):
    rx, ry, rz = np.deg2rad(angles)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    return (R @ coords.T).T

# -----------------------------
# 7. Coarse 3D rotation search
# -----------------------------
def coarse_rotation_align(coords_moving, grid_target, grid_size_moving, box_size=100.0, sigma=1.0, step=30):
    best_cc = -1
    best_angles = (0,0,0)
    best_coords = coords_moving.copy()
    for angles in itertools.product(range(0,360,step), repeat=3):
        coords_rot = rotate_coords(coords_moving - coords_moving.mean(axis=0), angles) + coords_moving.mean(axis=0)
        grid_rot = coords_to_grid(coords_rot, grid_size=grid_size_moving, box_size=box_size, sigma=sigma)
        _, cc = fft_cross_correlation(grid_target, grid_rot)
        if cc > best_cc:
            best_cc = cc
            best_angles = angles
            best_coords = coords_rot
    return best_coords, best_angles, best_cc

# -----------------------------
# 8. Fine rotation refinement
# -----------------------------
def refine_rotation(coords_moving, grid_target, grid_size_moving, best_angles, box_size=100.0, sigma=1.0, delta=15, step=5):
    best_cc = -1
    best_angles_final = best_angles
    best_coords_final = coords_moving.copy()
    angle_ranges = [range(max(a-delta,0), min(a+delta+1,360), step) for a in best_angles]
    for angles in itertools.product(*angle_ranges):
        coords_rot = rotate_coords(coords_moving - coords_moving.mean(axis=0), angles) + coords_moving.mean(axis=0)
        grid_rot = coords_to_grid(coords_rot, grid_size=grid_size_moving, box_size=box_size, sigma=sigma)
        _, cc = fft_cross_correlation(grid_target, grid_rot)
        if cc > best_cc:
            best_cc = cc
            best_angles_final = angles
            best_coords_final = coords_rot
    return best_coords_final, best_angles_final, best_cc

# -----------------------------
# 9. Sub-voxel translation refinement
# -----------------------------
def refine_translation(coords_aligned, coords_target):
    shift = coords_target.mean(axis=0) - coords_aligned.mean(axis=0)
    return coords_aligned + shift, shift

# -----------------------------
# 10. Convert coordinates to voxel grid
# -----------------------------
def coords_to_voxel(coords, grid_size=41, box_size=100.0):
    grid = np.zeros((grid_size, grid_size, grid_size))
    scale = (grid_size - 1) / box_size
    coords_centered = coords - coords.mean(axis=0) + (box_size / 2)
    indices = np.round(coords_centered * scale).astype(int)
    indices = np.clip(indices, 0, grid_size-1)
    np.add.at(grid, tuple(indices.T), 1)
    return grid

# -----------------------------
# 11. Save aligned PDB
# -----------------------------
def save_pdb(coords, out_path):
    structure = Structure.Structure("aligned_map")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    for i, (x,y,z) in enumerate(coords,1):
        atom_name = "C"
        atom = Atom.Atom(name=atom_name, coord=(x,y,z), bfactor=0.0, occupancy=1.0,
                         altloc=" ", fullname=" "+atom_name+" ", serial_number=i, element=atom_name[0])
        res_id = i if i<=999999 else 999999
        res = Residue.Residue((" ", res_id, " "), "ALA", "")
        res.add(atom)
        chain.add(res)
    model.add(chain)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    print(f"Saved aligned map PDB: {out_path}")

# -----------------------------
# 12. Global Pearson CC
# -----------------------------
def pearson_cc(grid1, grid2):
    g1 = grid1.flatten()
    g2 = grid2.flatten()
    return pearsonr(g1, g2)[0]

# -----------------------------
# 13. Phenix-style masked CC
# -----------------------------
def generate_mask_from_coords(coords, grid_size, box_size, mask_radius=2.0):
    mask = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    scale = (grid_size - 1) / box_size
    coords_centered = coords - coords.mean(axis=0) + (box_size / 2)
    indices = np.round(coords_centered * scale).astype(int)
    indices = np.clip(indices, 0, grid_size-1)

    for idx in indices:
        r_voxel = int(np.ceil(mask_radius * scale))
        slices = [slice(max(i-r_voxel,0), min(i+r_voxel+1, grid_size)) for i in idx]
        mask[slices[0], slices[1], slices[2]] = True
    return mask
# -----------------------------
# 14. Main
# -----------------------------
def main(map_pdb_path, target_pdb_path, out_path,
         grid_size_target=41, grid_size_map=31, box_size=100.0, sigma=1.0,
         coarse_step=30, refine_delta=15, refine_step=5):

    # --- Load coordinates ---
    coords_target = parse_pdb_atoms(target_pdb_path)
    coords_map = parse_pdb_atoms(map_pdb_path)

    # --- Target voxel grid ---
    grid_target = coords_to_grid(coords_target, grid_size=grid_size_target, box_size=box_size, sigma=sigma)

    # --- Coarse rotation ---
    coords_rot, best_angles, cc_coarse = coarse_rotation_align(coords_map, grid_target,
                                                               grid_size_map, box_size=box_size, sigma=sigma)
    print(f"Coarse rotation angles: {best_angles}, CC={cc_coarse:.3f}")

    # --- Refine rotation ---
    coords_refined, best_angles_refined, cc_refined = refine_rotation(coords_map, grid_target,
                                                                      grid_size_map, best_angles,
                                                                      box_size=box_size, sigma=sigma,
                                                                      delta=refine_delta, step=refine_step)
    print(f"Refined rotation angles: {best_angles_refined}, CC={cc_refined:.3f}")

    # --- FFT translation ---
    grid_rot = coords_to_grid(coords_refined, grid_size=grid_size_map, box_size=box_size, sigma=sigma)
    shift_vox, cc_translation = fft_cross_correlation(grid_target, grid_rot)
    coords_translated = apply_shift(coords_refined, shift_vox, grid_size_target, grid_size_map, box_size=box_size)
    print(f"FFT translation shift (voxels): {shift_vox}, CC={cc_translation:.3f}")

    # --- Sub-voxel refinement ---
    coords_aligned, shift_vec = refine_translation(coords_translated, coords_target)
    print(f"Sub-voxel translation applied: {shift_vec}")

    # --- Save aligned map PDB ---
    save_pdb(coords_aligned, out_path)
    save_pdb(coords_target, out_path.replace('.pdb', '_ref.pdb'))

    # --- Compute voxel correlation ---
    voxel_map_aligned = coords_to_voxel(coords_aligned, grid_size=grid_size_map, box_size=box_size)
    voxel_target = coords_to_voxel(coords_target, grid_size=grid_size_map, box_size=box_size)

    cc_global = pearson_cc(voxel_target, voxel_map_aligned)
    print(f"Final GLOBAL Pearson CC (full-atom target vs aligned map): {cc_global:.3f}")

    # --- Phenix-style masked CC (hard mask) ---
    mask_binary = voxel_target > 0
    print('mask:',mask_binary.sum())
    print('map_mask',voxel_map_aligned[mask_binary].sum())
    print('map',voxel_map_aligned.sum())
    print('target',voxel_target.sum())
    print('target_mask',voxel_target[mask_binary].sum())
    cc_masked_hard = pearson_cc(voxel_map_aligned[mask_binary], voxel_target[mask_binary])
    print(f"Final MASKED Pearson CC (hard binary mask): {cc_masked_hard:.3f}")

    # --- Phenix-style masked CC (soft mask, Gaussian blurred) ---
    mask_soft = gaussian_filter(mask_binary.astype(float), sigma=1.0)  # smooth edges
    mask_soft /= mask_soft.max()  # normalize 0–1

    # weighted Pearson CC
    v1 = voxel_map_aligned.flatten()
    v2 = voxel_target.flatten()
    w = mask_soft.flatten()

    mean1 = np.average(v1, weights=w)
    mean2 = np.average(v2, weights=w)
    cov = np.average((v1 - mean1) * (v2 - mean2), weights=w)
    std1 = np.sqrt(np.average((v1 - mean1) ** 2, weights=w))
    std2 = np.sqrt(np.average((v2 - mean2) ** 2, weights=w))
    cc_masked_soft = cov / (std1 * std2)

    print(f"Final MASKED Pearson CC (soft Gaussian mask): {cc_masked_soft:.3f}")

    # --- Phenix-style RSCC (real-space correlation coefficient) ---
    cc_rscc = real_space_volume_cc(voxel_target, voxel_map_aligned, mask=mask_binary)
    print(f"Final RSCC (real-space correlation, occupied voxels): {cc_rscc:.3f}")

    phi = real_space_volume_cc(voxel_target, voxel_map_aligned, mask=mask_binary)
    print(f"Final phi-cc (real-space correlation, occupied voxels): {phi:.3f}")

    return cc_global, cc_masked_hard, cc_masked_soft, cc_rscc, phi

# -----------------------------
# 15. Real-space correlation coefficient (RSCC)
# -----------------------------
# -----------------------------
# Real-space overlap CC (volume-based)
# -----------------------------
def real_space_volume_cc(voxel_target, voxel_map_aligned, mask=None):
    """
    Compute the fraction of overlapped volume normalized by the geometric mean of volumes.
    If mask is provided, only voxels within the mask are considered.
    """
    if mask is not None:
        v1 = voxel_target[mask]
        v2 = voxel_map_aligned[mask]
    else:
        v1 = voxel_target.flatten()
        v2 = voxel_map_aligned.flatten()

    numerator = np.sum(v1 * v2)
    denominator = np.sqrt(np.sum(v1**2) * np.sum(v2**2))
    return numerator / denominator if denominator != 0 else 0.0


def phi_coefficient(voxel_target, voxel_map_aligned, mask=None):
    """
    Compute the phi coefficient (Pearson correlation for binary data)
    between two binary voxel maps.
    """

    v1 = voxel_target.astype(int).flatten()
    v2 = voxel_map_aligned.astype(int).flatten()


    # Contingency table counts
    n11 = np.sum((v1 == 1) & (v2 == 1))  # both 1
    n00 = np.sum((v1 == 0) & (v2 == 0))  # both 0
    n10 = np.sum((v1 == 1) & (v2 == 0))  # v1=1, v2=0
    n01 = np.sum((v1 == 0) & (v2 == 1))  # v1=0, v2=1

    # Marginal totals
    n1_ = n11 + n10
    n0_ = n01 + n00
    n_1 = n11 + n01
    n_0 = n10 + n00

    # Phi coefficient formula
    denominator = np.sqrt(n1_ * n0_ * n_1 * n_0)
    phi = (n11 * n00 - n10 * n01) / denominator if denominator != 0 else 0.0

    return phi


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv)!=4:
        print("Usage: python align_map_to_pdb_3d_refined.py map.pdb full_atom.pdb output_aligned_map.pdb")
        sys.exit(1)

    map_pdb_path = sys.argv[1]
    target_pdb_path = sys.argv[2]
    out_path = sys.argv[3]

    main(map_pdb_path, target_pdb_path, out_path)
