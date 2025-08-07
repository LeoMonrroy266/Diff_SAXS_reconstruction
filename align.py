import sys
import os
from glob import glob
from pymol import cmd

def align_all(pdb_dir, ref_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cmd.load(ref_path, "reference")
    pdb_files = glob(os.path.join(pdb_dir, "*.pdb"))
    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file)
        cmd.load(pdb_file, "target")
        cmd.align("target", "reference")
        out_path = os.path.join(out_dir, name.replace(".pdb", "_aligned.pdb"))
        cmd.save(out_path, "target")
        cmd.delete("target")
    cmd.delete("reference")

def main():
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        args = sys.argv[idx+1:]
    else:
        args = []

    if len(args) != 3:
        print("Usage: pymol -cq align.py -- <pdb_dir> <ref_pdb> <out_dir>")
        sys.exit(1)

    pdb_dir, ref_pdb, out_dir = args
    align_all(pdb_dir, ref_pdb, out_dir)

if __name__ == "__main__":
    main()
