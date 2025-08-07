import tensorflow as tf
import numpy as np
import os, glob, random, math
from tqdm import tqdm
from pathlib import Path

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

def serialize_example(flat_array, i, j):
    return tf.train.Example(features=tf.train.Features(feature={
        "data": tf.train.Feature(int64_list=tf.train.Int64List(value=flat_array)),
        "pair_i": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        "pair_j": tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
    })).SerializeToString()

def create_tfrecords_chunked(directory, output_dir="tfrecords", n_arrays=1000, chunk_size=100):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {n_arrays} random arrays from: {directory}")
    all_files = load_npy_files_random(directory, n_arrays)
    total_chunks = math.ceil(n_arrays / chunk_size)

    tf_idx_train, tf_idx_test = 0, 0

    for chunk_i in range(total_chunks):
        start_i = chunk_i * chunk_size
        end_i = min((chunk_i + 1) * chunk_size, n_arrays)
        files_i = all_files[start_i:end_i]

        arrays_i = [pad_to_32(np.load(f)).astype(np.int64) for f in tqdm(files_i, desc=f"Loading chunk {chunk_i+1}/{total_chunks}")]

        for chunk_j in range(chunk_i, total_chunks):
            start_j = chunk_j * chunk_size
            end_j = min((chunk_j + 1) * chunk_size, n_arrays)
            files_j = all_files[start_j:end_j]

            arrays_j = [pad_to_32(np.load(f)).astype(np.int64) for f in files_j]

            examples_train = []
            examples_test = []

            total_pairs = (len(arrays_i) * len(arrays_j))
            skip_self_and_duplicates = chunk_i == chunk_j

            with tqdm(total=total_pairs, desc=f"Chunk {chunk_i+1}-{chunk_j+1} pairwise diffs") as pbar:
                for i, arr_i in enumerate(arrays_i):
                    for j, arr_j in enumerate(arrays_j):
                        # Avoid duplicate and self-pairs
                        if skip_self_and_duplicates and j <= i:
                            pbar.update(1)
                            continue

                        diff = write_bead(arr_i - arr_j)
                        flat = diff.flatten()
                        serialized = serialize_example(flat, start_i + i, start_j + j)

                        if (i + j) % 5 == 0:
                            examples_test.append(serialized)
                        else:
                            examples_train.append(serialized)
                        pbar.update(1)

            if examples_train:
                path_train = os.path.join(output_dir, f"train_{tf_idx_train}.tfrecords")
                with tf.io.TFRecordWriter(path_train) as writer:
                    for ex in tqdm(examples_train, desc=f"Writing {path_train}"):
                        writer.write(ex)
                tf_idx_train += 1

            if examples_test:
                path_test = os.path.join(output_dir, f"test_{tf_idx_test}.tfrecords")
                with tf.io.TFRecordWriter(path_test) as writer:
                    for ex in tqdm(examples_test, desc=f"Writing {path_test}"):
                        writer.write(ex)
                tf_idx_test += 1

    print(f"✅ Done. TFRecords written to {output_dir}. Train: {tf_idx_train}, Test: {tf_idx_test}")


if __name__ == "__main__":
    create_tfrecords_chunked(
        directory="/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/PISA_aligned/voxels",
        output_dir="tfrecord_chunks",
        n_arrays=1000,
        chunk_size=250
    )
