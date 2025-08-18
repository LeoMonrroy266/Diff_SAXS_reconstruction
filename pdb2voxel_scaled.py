import sys
import numpy as np
from Bio.PDB import PDBParser
from scipy.ndimage import gaussian_filter
import os

def parse_pdb_atoms(pdbfile):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdbfile)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    for atom in residue:
                        atoms.append(atom.coord)
    return np.array(atoms)

def center_and_scale_coords(coords, target_box_size):
    coords_centered = coords - coords.mean(axis=0)
    min_coords = coords_centered.min(axis=0)
    max_coords = coords_centered.max(axis=0)
    max_extent = np.max(max_coords - min_coords)
    scale_factor = (target_box_size * 0.95) / max_extent  # leave small margin
    coords_scaled = coords_centered * scale_factor
    return coords_scaled, scale_factor


def voxelize_atoms(coords, grid_size=31, box_size=100.0, sigma=0.7, scale_factor=1.0):
    """
    Converts centered & scaled coordinates into voxel grid.
    sigma is adjusted so that the atom's effective radius in Å remains constant.
    """
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    scale = (grid_size - 1) / box_size  # Å to voxel index
    coords_shifted = coords + (box_size / 2)  # shift to (0, box_size)
    indices = np.round(coords_shifted * scale).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    np.add.at(voxel_grid, tuple(indices.T), 1.0)

    # Adjust sigma to maintain constant real-space smoothing
    sigma_scaled = sigma / scale_factor
    voxel_grid = gaussian_filter(voxel_grid, sigma=sigma_scaled)
    return voxel_grid


def write_bead(voxel_grid, threshold=0.5):
    labeled = np.zeros(voxel_grid.shape, dtype=int)
    labeled[voxel_grid >= threshold] = 1
    labeled[voxel_grid <= -threshold] = -1
    return labeled

def main(pdbfile, save_basename, outpath, grid_size=31, box_size=100.0, threshold=0.5):
    coords = parse_pdb_atoms(pdbfile)
    if coords.size == 0:
        print("No atoms found in PDB.")
        return

    scaled_coords, scale_factor = center_and_scale_coords(coords, box_size)
    voxel_grid = voxelize_atoms(scaled_coords, grid_size=grid_size, box_size=box_size,
                                sigma=0.7, scale_factor=scale_factor)
    labeled_cube = write_bead(voxel_grid, threshold=threshold)

    np.save(f'{outpath}/{save_basename}_scaled.npy', labeled_cube)
    np.save(f'{outpath}/{save_basename}_continuous_cube.npy', voxel_grid)
    # save the scale for later
    np.save(f'{outpath}/{save_basename}_scale_factor.npy', np.array([scale_factor], dtype=float))
    return labeled_cube


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdb_to_voxel.py input.pdb output_path")
        sys.exit(1)

    pdb_file = sys.argv[1]
    out_path = sys.argv[2]
    save_name = os.path.splitext(os.path.basename(pdb_file))[0]
    main(pdb_file, save_name, out_path)
