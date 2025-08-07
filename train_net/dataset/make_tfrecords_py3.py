import tensorflow as tf
import numpy as np
import os, glob, random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def pad_to_32(cube):
    if cube.shape == (31, 31, 31):
        return np.pad(cube, pad_width=((0, 1), (0, 1), (0, 1)), mode='constant')
    elif cube.shape == (32, 32, 32):
        return cube
    else:
        raise ValueError(f"Unexpected shape: {cube.shape}")

def load_npy_files_random(directory, n, seed=42):
    files = glob.glob(os.path.join(directory, "*.npy"))
    if len(files) < n:
        raise ValueError(f"Only {len(files)} files found, but {n} requested.")
    random.seed(seed)
    random.shuffle(files)
    return files[:n]

def write_bead(voxel_grid, threshold=0.5):
    labeled = np.zeros(voxel_grid.shape, dtype=int)
    labeled[voxel_grid >= threshold] = 1
    labeled[voxel_grid <= -threshold] = -1
    return labeled

def compute_pairwise_differences(arrays):
    diffs, pairs = [], []
    for i in tqdm(range(len(arrays)), desc="Computing pairwise diffs"):
        for j in range(i + 1, len(arrays)):
            diff = arrays[i] - arrays[j]
            diffs.append(write_bead(diff))
            pairs.append((i, j))
    return diffs, pairs

def serialize_example(flat_array, i, j):
    return tf.train.Example(features=tf.train.Features(feature={
        "data": tf.train.Feature(int64_list=tf.train.Int64List(value=flat_array)),
        "pair_i": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        "pair_j": tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
    })).SerializeToString()

def write_tfrecord(filename, examples):
    with tf.io.TFRecordWriter(filename) as writer:
        for ex in tqdm(examples, desc=f"Writing {filename}"):
            writer.write(ex)

def create_tfrecords_from_npy(directory, n_arrays=500, num_threads=8):
    print(f"Loading {n_arrays} arrays randomly from: {directory}")
    npy_files = load_npy_files_random(directory, n_arrays)

    print("Loading and padding arrays...")
    arrays = [pad_to_32(np.load(f)).astype(np.int64) for f in tqdm(npy_files, desc="Padding arrays")]

    print("Computing pairwise differences...")
    diffs, pairs = compute_pairwise_differences(arrays)

    print(f"Total number of pairs: {len(pairs)}")

    print("Serializing examples using multithreading...")

    examples_train = []
    examples_test = []

    def process_pair(idx):
        flat = diffs[idx].flatten()
        return (idx, serialize_example(flat, *pairs[idx]))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_pair, range(len(diffs))),
                            total=len(diffs), desc="Serializing TFRecords"))

    for idx, ex in results:
        if idx % 5 == 0:
            examples_test.append(ex)
        else:
            examples_train.append(ex)

    print("Writing TFRecords...")
    write_tfrecord("train.tfrecords", examples_train)
    write_tfrecord("test.tfrecords", examples_test)

    print(f"✅ Done: {len(examples_train)} training and {len(examples_test)} test samples.")

# ───────────────────────────
if __name__ == "__main__":
    create_tfrecords_from_npy(
        directory="/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/PISA_aligned/voxels",
        #directory="/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/voxels",
        n_arrays=1000,
        num_threads=8
    )
