# -*- coding: utf-8 -*-
"""
result2pdb.py
-------------------------------------------------------------------------------
Converts grid to PDB file for visualization

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
from Bio.PDB import PDBParser, MMCIFParser

def voxel_to_pdb(vox: np.ndarray,
                 rmax: float,
                 pdb_file: str | Path,
                 atom_name_pos: str = "C",   # atom name for positive voxels
                 atom_name_neg: str = "O",   # atom name for negative voxels
                 residue: str = "ALA",
                 ccp4_file: str | Path | None = None,
                 pdb_reference_file: str | Path | None = None) -> None:
    # grid_size = vox.shape[0]
    if vox.shape[0] == 1:
        vox = vox.squeeze(0)  # removes the first dimension if it's 1
        grid_size = vox.shape[0]
    else:
        grid_size = vox.shape[0]

    idx = np.argwhere(vox != 0)
    if idx.size == 0:
        print(f"Warning: voxel grid contains no foreground voxels, skipping {pdb_file}")
        return  # Exit the function gracefully without error

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
        pdb_file = pdb_file.with_suffix('.pdb')
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
    """Return xyz array (N × 3) of all ATOM/HETATM records from PDB or CIF."""
    fname = Path(fname)
    if not fname.exists():
        raise FileNotFoundError(f"{fname} does not exist")

    if fname.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif fname.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file type: {fname.suffix}")

    structure = parser.get_structure("x", str(fname))
    atoms = list(structure.get_atoms())

    if len(atoms) == 0:
        raise ValueError(f"No atoms found in {fname}")

    return np.array([a.coord for a in atoms], float)




def estimate_rmax_bounding_box(pdb_path: str | Path) -> float:
    coords = pdb_coords(pdb_path)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    box_center = (mins + maxs) / 2
    distances = np.linalg.norm(coords - box_center, axis=1)
    return distances.max()


def write_bead(voxel_grid, threshold=0.5):
    labeled = np.zeros(voxel_grid.shape, dtype=int)
    labeled[voxel_grid >= threshold] = 1
    labeled[voxel_grid <= -threshold] = -1
    return labeled

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
    voxel = write_bead(voxel)
    voxel_to_pdb(voxel, rmax, out_dir/name, ccp4_file=f'{out_dir}/{name}.ccp4')

    if target_pdb:
        # Determine which output file exists: PDB or CIF
        out_pdb = out_dir / f"{name}.pdb"
        out_cif = out_dir / f"{name}.cif"

        if out_pdb.exists():
            ref_file = out_pdb
        elif out_cif.exists():
            ref_file = out_cif
        else:
            print(f"⚠ No output PDB/CIF found for {name}, skipping RMSD.")
            return

        # Only compute RMSD if the file contains atoms
        print(ref_file)
        try:
            ref_xyz = pdb_coords(ref_file)
            mob_xyz = pdb_coords(target_pdb)


        except ValueError as e:
            print(f"{e}, skipping RMSD")
            return

        if ref_xyz.shape[0] != mob_xyz.shape[0]:
            print(f"Skipping RMSD — atom count mismatch ({ref_xyz.shape[0]} vs {mob_xyz.shape[0]})")
        else:
            _, rmsd = kabsch_align(ref_xyz, mob_xyz)
            print(f"RMSD(out ↔ target) = {rmsd:6.3f} Å")


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
