import sys
import glob
import argparse
from pathlib import Path
import numpy as np

import auto_encoder_t_py3 as net
import result2pdb
from result2pdb import *

p = argparse.ArgumentParser()
p.add_argument("--n_samples",type=int, default=100)
p.add_argument("--model_path",required=True)
p.add_argument("--output_folder",required=False)
args = p.parse_args()
def sample_latent_vectors_normal(n_samples, z_dim, std=0.5):
    # Normal distribution centered at 0 with some std,
    # then clipped to [-1, 1]
    samples = np.random.normal(loc=0.0, scale=std, size=(n_samples, z_dim))
    samples = np.clip(samples, -1.0, 1.0)
    return samples.astype(np.float32)

if __name__ == '__main__':
    ckpt = next(Path(args.model_path).glob("*.keras"), None) \
           or next(Path(args.model_path).glob("*.h5"))
    N = args.n_samples
    Z_DIM = 200
    latent_samples = sample_latent_vectors_normal(N, Z_DIM)

    ones_col = np.ones((1,Z_DIM), dtype=np.float32)  # shape (N, 1)
    neg_ones_col = -np.ones((1,Z_DIM), dtype=np.float32)  # shape (N, 1)
    latent_samples = np.concatenate([latent_samples, ones_col, neg_ones_col], axis=0)  # shape (N, 202)

    AUTO, ENCODER, DECODER = net.build_autoencoder()
    net._load_weights(AUTO, ckpt)  # sets weights for encoder+decoder
    diff_maps = DECODER.predict(latent_samples, batch_size=10, verbose=0)
    for n,i in enumerate(diff_maps):
        i = i[:31,:31,:31]
        i = i[...,0]
        result2pdb.write_single_pdb(i,name=f'{n}_sample.pdb',output_folder=args.output_folder, rmax=50)
        print(i.max(),i.min())


