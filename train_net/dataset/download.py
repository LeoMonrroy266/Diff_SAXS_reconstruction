import os
import subprocess
from tqdm import tqdm

BASE_URL = "http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimer.pdb?"
CHAIN_SUFFIX = ":1,1"

input_file = "interfaces.txt"

with open(input_file) as fr:
    lines = fr.readlines()

for line in tqdm(lines, desc="Downloading PISA PDBs"):
    try:
        pdb_list = line.split("?")[1].strip().split(",")
    except IndexError:
        continue  # skip malformed lines

    for pdbcode in pdb_list:
        outname = f"{pdbcode}.pdb"

        if os.path.exists(outname):
            # Already downloaded
            continue

        url = f"{BASE_URL}{pdbcode}{CHAIN_SUFFIX}"
        cmd = ["wget", "-q", "-O", outname, url]

        try:
            subprocess.run(cmd, check=True)

            # Check if file is empty or invalid
            with open(outname) as fs:
                first_line = fs.readline().strip()
                if not first_line or first_line.startswith("***"):
                    os.remove(outname)
                    tqdm.write(f"  → {pdbcode} is invalid or empty, removed.")
        except Exception as e:
            tqdm.write(f"  → Failed to download {pdbcode}: {e}")
            if os.path.exists(outname):
                os.remove(outname)
