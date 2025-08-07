import os
import glob
import sys
from multiprocessing import Pool
from pdb2_voxel_no_tbx import *
from tqdm import tqdm


def process_file(args):
    pdbfile, outdir = args
    basename = os.path.splitext(os.path.basename(pdbfile))[0]

    labeled_path = os.path.join(outdir, f"{basename}_labeled_cube.npy")
    continuous_path = os.path.join(outdir, f"{basename}_continous_cube.npy")

    if os.path.exists(labeled_path) and os.path.exists(continuous_path):
        return f"Skipped {basename} (already exists)"

    main(pdbfile, basename, outdir)
    return f"Processed {basename}"


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    args = [(pdb, output_dir) for pdb in pdb_files]

    with Pool() as pool:
        # Wrap the iterator with tqdm
        for result in tqdm(pool.imap_unordered(process_file, args), total=len(args), desc="Processing PDBs"):
            pass  # you could also print(result) for logging each file
