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

def center_coords(coords):
    centroid = coords.mean(axis=0)
    return coords - centroid  # center without scaling

def voxelize_atoms_fixed_box(coords, grid_size=31, box_size=100.0, sigma=0.7):
    """
    Converts centered coordinates into voxel grid without per-structure scaling.
    Assumes box_size is the total spatial span (in Å), e.g., 100 Å = ±50 Å radius.
    """
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    scale = (grid_size - 1) / box_size  # Å to voxel index
    coords_shifted = coords + (box_size / 2)  # shift to (0, box_size)
    indices = np.round(coords_shifted * scale).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    np.add.at(voxel_grid, tuple(indices.T), 1.0)

    voxel_grid = gaussian_filter(voxel_grid, sigma=sigma)
    #voxel_grid /= voxel_grid.max()
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
    #centered_coords = center_coords(coords)  # center only, no scaling
    voxel_grid = voxelize_atoms_fixed_box(coords, grid_size=grid_size, box_size=box_size)

    labeled_cube = write_bead(voxel_grid, threshold=threshold)

    # Save the labeled voxel grid
    np.save(f'{outpath}/{save_basename}_labeled_cube.npy', labeled_cube)
    np.save(f'{outpath}/{save_basename}_continous_cube.npy', voxel_grid)
    return labeled_cube


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdb_to_voxel.py input.pdb output_path")
        sys.exit(1)

    pdb_file = sys.argv[1]
    out_path = sys.argv[2]

    # Dynamically extract the basename (without extension)
    save_name = os.path.splitext(os.path.basename(pdb_file))[0]

    main(pdb_file, save_name, out_path)
