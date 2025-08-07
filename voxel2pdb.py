#!/usr/bin/env python3
"""
Convert a 31×31×31 voxel grid (values 0/1/–1) to a dummy-atom PDB.
All non-zero voxels become one carbon atom each (ALA residue).
"""

from pathlib import Path
import sys
import numpy as np


def write_pdb(voxel: np.ndarray, output: str | Path, rmax: float, grid_size: int = 31) -> None:
    """
    Save a voxel grid as a PDB file made of dummy carbon atoms.

    Parameters
    ----------
    voxel : (N,N,N) numpy array
        Voxel map with non-zero entries marking atoms/beads.
    output : str or Path
        Path to the PDB file to write.
    rmax : float
        Desired molecular radius (Å) after scaling.
    grid_size : int, default 31
        Size of the cube in voxels (must match voxel.shape[0]).
    """
    rmax /= 0.9  # keep author’s original scaling convention

    # Find all occupied voxels
    coords = np.argwhere(voxel != 0)        # shape (num_atoms, 3)
    if coords.size == 0:
        raise ValueError("No occupied voxels found!")

    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1).max() / 0.9
    scale = rmax / radius

    with open(output, "w") as f:
        for atom_id, xyz in enumerate(coords, start=1):
            x, y, z = (xyz - center) * scale
            f.write(
                f"ATOM  {atom_id:5d}  C   ALA A{atom_id:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  1.00           C  \n"
            )
        f.write("END\n")


def voxel_to_pdb_as_string(voxel: np.ndarray, rmax: float, grid_size: int = 31) -> str:
    """
    Return a PDB string instead of writing a file.
    """
    rmax /= 0.9
    coords = np.argwhere(voxel != 0)
    if coords.size == 0:
        raise ValueError("No occupied voxels found!")

    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1).max() / 0.9
    scale = rmax / radius

    lines = []
    for atom_id, xyz in enumerate(coords, start=1):
        x, y, z = (xyz - center) * scale
        lines.append(
            f"ATOM  {atom_id:5d}  C   ALA A{atom_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  1.00           C  "
        )
    lines.append("END")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python voxel2pdb.py <voxel.npy> [output.pdb] [rmax=50]")

    voxel_file = Path(sys.argv[1]).expanduser()
    output_pdb = Path(sys.argv[2]) if len(sys.argv) > 2 else voxel_file.with_suffix(".pdb")
    rmax_value = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0

    voxel_grid = np.load(voxel_file)
    print(f"Loaded voxel grid {voxel_grid.shape} from {voxel_file}")

    write_pdb(voxel_grid, output_pdb, rmax_value)
    print(f"PDB written to {output_pdb}")
