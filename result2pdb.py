#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voxel_to_pdb_tools.py – pure-Python replacement for the old scitbx/sastbx helpers
-------------------------------------------------------------------------------
Requires:
    numpy        (array maths)
    biopython    (PDB reading/writing)
    mrcfile      (write CCP4/MRC map)   pip install mrcfile
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import os, shutil, tempfile
import numpy as np
import mrcfile
import gemmi
from Bio.PDB import PDBParser, PDBIO
from pathlib import Path
import numpy as np
from Bio.PDB import PDBIO, MMCIFIO
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

def voxel_to_pdb(vox: np.ndarray,
                 rmax: float,
                 pdb_file: str | Path,
                 atom_name_pos: str = "C",   # atom name for positive voxels
                 atom_name_neg: str = "O",   # atom name for negative voxels
                 residue: str = "DUM",
                 ccp4_file: str | Path | None = None,
                 pdb_reference_file: str | Path | None = None) -> None:

    idx = np.argwhere(vox != 0)
    if idx.size == 0:
        raise ValueError("voxel grid contains no foreground voxels")

    grid_size = vox.shape[0]
    box_size = 2 * rmax
    voxel_size = box_size / (grid_size - 1)
    coords = idx * voxel_size - rmax  # Convert voxel indices to Å coordinates centered at 0

    # Build structure
    model = Model(0)
    chain = Chain("A")
    for n, (x, y, z) in enumerate(coords, 1):
        val = vox[tuple(idx[n - 1])]
        if val == 1:
            atom_name = atom_name_pos
        elif val == -1:
            atom_name = atom_name_neg
        else:
            continue

        atom = Atom(
            name=atom_name,
            coord=(x, y, z),
            bfactor=1.0,
            occupancy=1.0,
            altloc=" ",
            fullname=f" {atom_name} ",
            serial_number=n,
            element=atom_name[0]
        )
        res_id = n if n <= 999999 else 999999  # Avoid massive IDs in CIF too
        res = Residue((" ", res_id, " "), residue, "")
        res.add(atom)
        chain.add(res)
    model.add(chain)

    # Choose output format
    pdb_file = Path(pdb_file)
    if len(coords) > 9999:
        cif_file = pdb_file.with_suffix('.cif')
        io = MMCIFIO()
        io.set_structure(model)
        io.save(str(cif_file))
        print(f"Saved as mmCIF: {cif_file}")
    else:
        io = PDBIO()
        io.set_structure(model)
        io.save(str(pdb_file))
        print(f"Saved as PDB: {pdb_file}")



def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return rotated-and-translated Q superposed onto P and the RMSD.
    """
    Pc = P.mean(axis=0); Qc = Q.mean(axis=0)
    P0, Q0 = P - Pc, Q - Qc
    C = Q0.T @ P0
    V, S, W = np.linalg.svd(C)
    U = V @ W
    if np.linalg.det(U) < 0:               # correct possible reflection
        V[:, -1] *= -1
        U = V @ W
    rot = Q0 @ U
    rmsd = np.sqrt(((rot - P0)**2).sum() / len(P))
    return rot + Pc, rmsd


def pdb_coords(fname: str | Path) -> np.ndarray:
    """xyz array (N × 3) of all ATOM/HETATM records."""
    atoms = PDBParser(QUIET=True).get_structure("x", fname).get_atoms()
    return np.array([a.coord for a in atoms], float)




def estimate_rmax_bounding_box(pdb_path: str | Path) -> float:
    coords = pdb_coords(pdb_path)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    box_center = (mins + maxs) / 2
    distances = np.linalg.norm(coords - box_center, axis=1)
    return distances.max()



# ───────────────────── high-level API (unchanged names) ───────────────────────
def write2pdb(group: np.ndarray,
              rmax: float,
              output_folder: str | Path,
              iq_file: str | Path | None = None,
              target_pdb: str | Path | None = None) -> None:
    """
    Average many voxel models, write out ·pdb and ·ccp4, align vs. target.
    """
    out_dir = Path(output_folder); out_dir.mkdir(parents=True, exist_ok=True)
    tmp2 = out_dir/"sub2"; tmp3 = out_dir/"sub3"
    tmp2.mkdir(exist_ok=True); tmp3.mkdir(exist_ok=True)

    # ── 1 write every voxel to PDB ───────────────────────────────
    for i, vox in enumerate(group):
        voxel_to_pdb(vox, rmax, tmp2/f"{i}.pdb")

    # ── 2 align each PDB onto #0 using Kabsch ────────────────────
    ref = pdb_coords(tmp2/"0.pdb")
    ave = np.zeros_like(group[0], float)
    for i in range(len(group)):
        mob_xyz = pdb_coords(tmp2/f"{i}.pdb")
        aligned, _ = kabsch_align(ref, mob_xyz)
        # overwrite file with aligned coords
        voxel_to_pdb(group[i], rmax, tmp3/f"{i}.pdb")   # coord scaling
        # shift coordinates in file
        atoms = PDBParser(QUIET=True).get_structure("m", tmp3/f"{i}.pdb")
        for a, new in zip(atoms.get_atoms(), aligned):
            a.set_coord(new)
        PDBIO().set_structure(atoms); PDBIO().save(tmp3/f"{i}.pdb")
        # back-map to voxel grid
        vox = pdb_to_voxel(tmp3/f"{i}.pdb", grid=group[0].shape, rmax=rmax)
        ave += vox

    ave /= len(group)                          # average density
    binarised = (ave > 0.3).astype(np.uint8)   # same threshold as before

    # save merged PDB & CCP4
    voxel_to_pdb(binarised, rmax, out_dir/"out.pdb")


    # ── 3 align optional target_pdb ~ quick CC ───────────────────
    if target_pdb:
        ref_xyz = pdb_coords(out_dir / f'{name}.pdb')
        mob_xyz = pdb_coords(target_pdb)
        _, rmsd = kabsch_align(ref_xyz, mob_xyz)
        print(f"RMSD(out ↔ target) = {rmsd:6.3f} Å")


def write_single_pdb(voxel: np.ndarray,
                     output_folder: str | Path,
                     name: str,
                     target_pdb: str | Path | None = None,
                     rmax_ref_pdb: str | Path | None = None,
                     rmax: float | None = None):


    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If no rmax is provided, estimate from reference PDB
    if rmax is None:
        if rmax_ref_pdb is None:
            raise ValueError("Either rmax or rmax_ref_pdb must be provided")
        rmax = estimate_rmax_bounding_box(rmax_ref_pdb)

    voxel_to_pdb(voxel, rmax, out_dir/name, ccp4_file=f'{out_dir}/{name}.ccp4')


    if target_pdb:
        ref_xyz = pdb_coords(out_dir/f'{name}.pdb')
        mob_xyz = pdb_coords(target_pdb)
        _, rmsd = kabsch_align(ref_xyz, mob_xyz)
        print(f"RMSD(out ↔ target) = {rmsd:6.3f} Å")



def pdb_to_voxel(pdb_file: str | Path,
                 grid=(31, 31, 31),
                 rmax: float = 15.0) -> np.ndarray:
    """
    Very coarse 'back projection': each atom occupies the nearest grid node.
    """
    coords = pdb_coords(pdb_file)
    centre = coords.mean(axis=0)
    scale  = (grid[0]//2) / rmax
    idx = np.rint((coords - centre) * scale + (grid[0]-1)/2).astype(int)
    idx = np.clip(idx, 0, grid[0]-1)
    vox = np.zeros(grid, np.uint8)
    vox[idx[:,0], idx[:,1], idx[:,2]] = 1
    return vox


def cal_cc(voxel_group: np.ndarray,
           rmax: float,
           output_folder: str | Path,
           target_pdb: str | Path) -> None:
    """
    Compute pair-wise (Kabsch) RMSD between each voxel model and a target PDB.
    Saves matrix as txt; prints progress.
    """
    out_dir = Path(output_folder); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=out_dir, prefix="temp_")
    target_xyz = pdb_coords(target_pdb)

    n, m, *_ = voxel_group.shape   # n replicates × m snapshots
    rmsd_mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            pdb_path = Path(tmp_dir)/f"{i}_{j}.pdb"
            voxel_to_pdb(voxel_group[i, j], rmax, pdb_path)
            xyz = pdb_coords(pdb_path)
            _, rmsd = kabsch_align(target_xyz, xyz)
            rmsd_mat[i, j] = rmsd
            print(f"{i:2d},{j:2d}  RMSD={rmsd:6.3f}")

    np.savetxt(out_dir/"cc_mat.txt", rmsd_mat, fmt="%.3f")
    shutil.rmtree(tmp_dir)
    print("✓ cal_cc finished   (matrix saved cc_mat.txt)")
