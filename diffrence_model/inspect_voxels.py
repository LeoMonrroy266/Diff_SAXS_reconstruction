# -*- coding: utf-8 -*-
"""
Helper for inspecting training data from tfrecords
Randomly samples N grids and returns them as .pdb files for visualisation

* build_autoencoder() returns (auto, encoder, decoder)
* decode_latent()     : latent  → reconstruction
* encode_decode()     : voxel   → reconstruction
"""

from result2pdb import write_single_pdb
import os
import random
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm




def parse_tfrecord(example_proto):
       features = {
           "data": tf.io.FixedLenFeature([32 * 32 * 32], tf.int64),
           "index": tf.io.FixedLenFeature([1], tf.int64)
       }
       parsed = tf.io.parse_single_example(example_proto, features)
       data = tf.cast(parsed["data"], tf.int32)
       return data

def load_n_random_diffs_from_tfrecord(tfrecord_path, n_samples):
       # Read all examples
       raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
       parsed_dataset = raw_dataset.map(parse_tfrecord)

       all_arrays = []
       for array in tqdm(parsed_dataset, desc="Loading TFRecord"):
           # reshape to 3D
           all_arrays.append(array.numpy().reshape((32, 32, 32)))

       if n_samples > len(all_arrays):
           raise ValueError(f"Requested {n_samples} samples, but TFRecord only has {len(all_arrays)}")

       # pick n random arrays
       selected = random.sample(all_arrays, n_samples)

       return selected




# ─── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
   # ───── Example usage ─────
   tfrecord_file = "/home/leonardo/testing_saxs/training_data_LOV2/train.tfrecords"
   n = 100
   diff_arrays = load_n_random_diffs_from_tfrecord(tfrecord_file, n)
   print(np.where(diff_arrays[0] == -1)[0].sum())

   [write_single_pdb(voxel=voxel,output_folder='/home/leonardo/testing_saxs/inspected_diffs',name=str(n),rmax=50 ) for n, voxel in enumerate(diff_arrays)]