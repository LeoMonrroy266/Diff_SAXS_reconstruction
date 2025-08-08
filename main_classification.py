#!/usr/bin/env python3
# coding: utf-8
"""
Genetic-algorithm driver  —  TensorFlow-2 / Keras version
--------------------------------------------------------
"""
from functools import partial
from pathlib import Path
import argparse, time, threading, multiprocessing as mp, os
import numpy as np
import tensorflow as tf
import map2iq, region_search, result2pdb, pdb2_voxel_no_tbx, voxel2pdb
import matplotlib.pyplot as plt
import processSaxs as ps
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import classification_autoencoder_t as net         # <- helper you already have
#                         build_autoencoder() / decode_latent() / encode_decode()

# ──────────────────────────── CLI  /  constants ────────────────────────────
GPU_NUM     = 1                      # limit via CUDA_VISIBLE_DEVICES if needed
BATCH_SIZE  = 10
Z_DIM       = 200
np.set_printoptions(precision=10)

p = argparse.ArgumentParser()
p.add_argument("--model_path",   required=True)
p.add_argument("--iq_path",      required=True)
p.add_argument("--output_folder",required=True)
p.add_argument("--rmax",         type=float, default=0)
p.add_argument("--dark_model",   required=True)
p.add_argument("--rmax_start",   type=float,  default=10)
p.add_argument("--rmax_end",     type=float,  default=300)
p.add_argument("--max_iter",     type=int,    default=80)
#p.print_help()
args = p.parse_args()

out_dir = Path(args.output_folder)
out_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────── Load trained auto-encoder (GPU) ───────────────────────
ckpt = next(Path(args.model_path).glob("*.keras"), None) \
    or next(Path(args.model_path).glob("*.h5"))

AUTO, ENCODER, DECODER = net.build_autoencoder()
net._load_weights(AUTO, ckpt)        # sets weights for encoder+decoder
print(f"Loaded weights from {ckpt.name}")

# ─────────────────── Latent initialisation stats  ───────────────────────────
init_stats = np.loadtxt(Path(args.model_path) /
                        "genegroup_init_parameter_2.txt")
##################Might need to change########################################
# ───────────────────────── generic thread helper ────────────────────────────
class MyThread(threading.Thread):
    def __init__(self, fn, *a):
        super().__init__(); self.fn, self.a, self.ret = fn, a, None
    def run(self): self.ret = self.fn(*self.a)
    def result(self): return self.ret

# ────────────────────────────  GA class  ────────────────────────────────────
class Evolution:
    def __init__(self, target_path, dark_voxel, mode, rmin, rmax, max_iter=80):
        self.dark_voxel = dark_voxel  # store dark voxel grid
        self.target_path = target_path
        self.mode, self.rmin, self.rmax = mode, rmin, rmax
        self.max_iter   = max_iter
        self.gene_len   = Z_DIM
        self.group_num  = 300
        self.inherit    = 300
        self.remain     = 20
        self.stats_num  = 20
        self.xover_pts  = 2
        self.group      = self._init_group(self.group_num)
        self.score      = self._score(self.group)
        self._rank()
        if self.mode == "withoutrmax":
            self.top_r = self.group[:100,-1].copy()
        print("init top-5:", self.score[:5])

    # ───── population init ─────
    def _init_group(self, n):
        g = np.random.normal(init_stats[:, 0], init_stats[:, 1], (n, self.gene_len))
        if self.mode == "withoutrmax":
            r = np.random.randint(self.rmin, self.rmax, (n,1))
            g = np.hstack([g, r])
        return g.astype(np.float32)

    # ───── decode helpers (GPU Keras) ─────
    def _decode_batch(self, lat):
        v = DECODER.predict(lat, batch_size=BATCH_SIZE, verbose=0)
        v = (v > 0.1).astype(np.int8)[...,0]          # (N,31,31,31)
        return v[:,:31,:31,:31]

    def _decode_group(self, lat):
        out = [self._decode_batch(c)
               for c in np.array_split(lat, max(1,len(lat)//BATCH_SIZE))]
        return np.concatenate(out)

    # ───── region_process replaces sess.run ─────
    def _region_process(self, cubes):
        """Keep largest connected region; re-encode with ENCODER."""
        batch  = cubes.shape[0]
        vox4d  = np.zeros((batch,32,32,32,1), np.float32)
        vox4d[:,:31,:31,:31,0] = cubes
        z  = ENCODER.predict(vox4d, batch_size=BATCH_SIZE, verbose=0)
        # region filter
        clean = []; z_out = []
        for zi, vol in zip(z, cubes):
            vol, nreg = region_search.find_biggest_region(vol)
            clean.append(vol); z_out.append(zi)
        return np.vstack(z_out), np.stack(clean)

    # ───── fitness ─────
    def _score(self, group):
        probs = self._decode_group(group[:,:self.gene_len])
        classes = np.argmax(probs, axis=-1)
        vox = convert_class_indices_to_labels(classes)

        res = np.array(list(map(add_propagated_difference, [self.dark_voxel] * vox.shape[0], vox)))  #Extrapolate light gri
        # Unzip into two separate lists
        combined_vox, b_parts = zip(*res)

        # Convert to numpy arrays
        combined_vox = np.array(combined_vox)
        b_parts = np.array(b_parts)

        if self.mode == "withoutrmax":
            r = group[:,-1,None]
            X = np.hstack([vox.reshape(vox.shape[0],-1), r])
            with mp.Pool(20) as P:
                func = partial(map2iq.run_withoutrmax, iq_file=self.target_path,dark_model=dark_voxel)
                res = np.array(P.map(func, X))
            group[:,-1] = res[:,1]
            return res[:,0]
        else:
            with mp.Pool(20) as P:
                func = partial(map2iq.run_withrmax, iq_file=self.target_path, dark_model=dark_voxel)
                res = np.array(P.map(func, combined_vox))
                return res

    def _rank(self):
        #idx = np.argsort(self.score)# sorts based on lowest chi2
        idx = np.argsort(self.score)[::-1] # Sorts on best r²
        self.group, self.score = self.group[idx], self.score[idx]

    # ───── crossover ─────
    def _xover(self, g):
        np.random.shuffle(g)
        for i in range(0, self.inherit-self.remain, 2):
            pts = np.sort(np.random.randint(0,self.gene_len, 2*self.xover_pts))
            for j in range(self.xover_pts):
                a,b = pts[2*j:2*j+2]
                if np.random.rand() < 0.8:
                    g[i,a:b], g[i+1,a:b] = g[i+1,a:b].copy(), g[i,a:b].copy()

    # ───── mutation (unchanged, just numpy) ─────
    def _mutate(self, g):
        if self.mode=="withoutrmax":
            mu, sigma = self.top_r.mean(), self.top_r.std()
        for i in range(self.inherit-self.remain):
            r = np.random.rand(self.gene_len+1)
            for j in range(self.gene_len):
                if r[j] < 0.1: # changed from 0.05 to 0.1
                    g[i,j] = abs(np.random.normal(init_stats[j,0],
                                                  init_stats[j,1]))
            if self.mode=="withoutrmax" and r[-1] < 0.5:
                if np.random.rand()<0.5:
                    g[i,-1] = np.random.randint(self.rmin, self.rmax)
                else:
                    val = np.random.normal(mu, sigma)
                    while val <= 10: val = np.random.normal(mu, sigma)
                    g[i,-1] = val

    # ───── selection (tournament) ─────
    def _select(self):
        keep = self.remain
        out  = []
        while len(out) < self.inherit-keep:
            a,b = np.random.randint(0,self.group_num,2)
            if np.random.rand()>0.1:
                out.append(self.group[a] if a<b else self.group[b])
            else:
                out.append(self.group[b] if a<b else self.group[a])
        return np.array(out, np.float32)

    # ───── GA iteration loop (unchanged logic) ─────
    def iterate(self):
        counter = 0
        prev_best = 0
        for step in range(1, self.max_iter+1):
            sel = self._select()
            self._xover(sel); self._mutate(sel)
            self.group = np.vstack([sel, self.group[:self.remain]])
            self.score = self._score(self.group); self._rank()
            self.score[0]
            print(f"iter {step:3d} best={self.score[0]} "
                  f"mean={self.score[:20].mean():8.3f}")

            if step>1 and self.score[0]>=prev_best:
                counter+=1
            else:
                counter=0
            prev_best = self.score[0]
            if counter>10:
                break
        return self.group[0]     # best latent


@staticmethod
def add_propagated_difference(a, b):
    """
    Adds a + b but only adds parts of b connected (directly or indirectly) to a,
    propagating through connected voxels in b.
    b can have -1, 0, 1 values.

    Returns:
        result: a + connected part of b
        b_masked: only the connected part of b that was added
    """
    assert a.shape == b.shape
    result = np.copy(a)
    structure = generate_binary_structure(3, 1)

    # Voxels in b that are non-zero
    b_nonzero = (b != 0)

    # Start from voxels where a is non-zero and overlap b_nonzero (initial seeds)
    overlap = result != 0
    seeds = overlap & b_nonzero

    # Create mask to hold voxels in b connected to a
    connected_mask = np.zeros_like(b, dtype=bool)
    connected_mask[seeds] = True

    prev_sum = 0
    # Iteratively propagate connectedness through b_nonzero voxels
    while connected_mask.sum() != prev_sum:
        prev_sum = connected_mask.sum()
        # Dilate connected_mask within b_nonzero to add adjacent b voxels connected to a
        dilation = binary_dilation(connected_mask, structure=structure) & b_nonzero
        connected_mask |= dilation

    # Extract the part of b that will be added
    b_masked = np.zeros_like(b)
    b_masked[connected_mask] = b[connected_mask]

    # Add only the connected voxels from b to result
    result += b_masked

    return result, b_masked

@staticmethod
def convert_class_indices_to_labels(class_array: np.ndarray) -> np.ndarray:
    """
    Convert class indices {0,1,2} to labels {-1,0,1}.
    """
    label_map = {0: -1, 1: 0, 2: 1}
    # Vectorized mapping:
    vectorized_map = np.vectorize(label_map.get)
    return vectorized_map(class_array)


# ─────────────────── pre-process SAXS  ───────────────────────────
process_result = ps.process(args.iq_path)
if len(process_result)==2:
    estimate_rmax=process_result[1]

saxs = process_result[0]
np.savetxt(out_dir/"processed_saxs.iq", saxs)
target_path = out_dir/"processed_saxs.iq"
mode = "withoutrmax" if args.rmax==0 else "withrmax"

# ─────────────────── pre-process dark model   ───────────────────────────
dark_voxel = pdb2_voxel_no_tbx.main(args.dark_model, 'dark', out_dir)
np.save(f'{out_dir}/dark.npy', dark_voxel)
result2pdb.write_single_pdb(dark_voxel, out_dir, 'dark.pdb', rmax=50)
# ─────────────────── run GA, save best shape ─────────────────────
ga = Evolution(target_path, dark_voxel, mode, args.rmax_start, args.rmax_end+1, max_iter=args.max_iter)
best_lat = ga.iterate()
best_voxel = ga._decode_batch(best_lat[:Z_DIM][None,:])[0]
# ─────────────────── save best diff map ─────────────────────
np.save(f'{out_dir}/best_voxel.npy', best_voxel)
result2pdb.write_single_pdb(best_voxel, out_dir, 'diff.pdb',rmax=50)
# ─────────────────── save extrapolated light ───────────────
best_light, diff_mask = add_propagated_difference(dark_voxel,best_voxel)
np.save(f'{out_dir}/best_light.npy', best_light) #
result2pdb.write_single_pdb(best_light, out_dir, 'light.pdb',rmax=50)
result2pdb.write_single_pdb(diff_mask, out_dir, 'diff_mask.pdb',rmax=50)
print("GA finished.")
# Save final SAXS curve and a plot showing comparison   

# Load experimental SAXS data (already processed)
exp_data = map2iq.read_iq_ascii(target_path)
# Initialize ED_map with experimental data and dark voxel
ed_map = map2iq.ED_map(exp_data, dark_voxel, rmax=50)

# Compute SAXS profile for best_light voxel (extrapolated light)
final_saxs = ed_map.compute_saxs_profile(best_light, save_plot=True)

# Save final SAXS curve: columns = q, I_calc
final_curve_path = out_dir / "final_saxs_curve.txt"
np.savetxt(final_curve_path, np.column_stack((ed_map.exp_q, final_saxs)),
           header="q(A^-1)  I_calc", comments='')

print(f"Final SAXS curve saved to {final_curve_path}")

# Plot experimental vs calculated SAXS
plt.figure(figsize=(8, 5))
plt.errorbar(exp_data.q, exp_data.i, label='Experimental')
plt.plot(ed_map.exp_q, final_saxs, label='Calculated (best_light)')
plt.xlabel('q (Å⁻¹)')
plt.ylabel('I(q)')
plt.title('SAXS Curve: Experimental vs Calculated')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = out_dir / "final_saxs_comparison.png"
plt.savefig(plot_path)
plt.close()
print(f"Final SAXS comparison plot saved to {plot_path}")
