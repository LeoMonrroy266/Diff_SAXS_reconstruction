import tensorflow as tf
import numpy as np
from pathlib import Path
import random

# Your TFRecord voxel shape
VOXEL_SHAPE = (32, 32, 32)

# Import your function
from result2pdb import write_single_pdb  # Replace with actual module name


def parse_fn(example_proto):
    feature_description = {
        'data': tf.io.FixedLenFeature([np.prod(VOXEL_SHAPE)], tf.int64),
        'pair_i': tf.io.FixedLenFeature([], tf.int64),
        'pair_j': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    voxel = tf.reshape(parsed['data'], VOXEL_SHAPE)
    return voxel, parsed['pair_i'], parsed['pair_j']


def count_tfrecord_examples(tfrecord_path):
    return sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))


def convert_random_samples_to_pdbs(tfrecord_path, output_dir, num_samples=100, seed=42):
    total = count_tfrecord_examples(tfrecord_path)
    print(f"🔎 Total examples in TFRecord: {total}")
    if num_samples > total:
        raise ValueError(f"TFRecord only has {total} samples, cannot pick {num_samples}.")

    # Choose random indices
    random.seed(seed)
    chosen_indices = set(random.sample(range(total), num_samples))

    dataset = tf.data.TFRecordDataset(tfrecord_path).map(parse_fn)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (voxel_tensor, pair_i, pair_j) in enumerate(dataset):
        if i not in chosen_indices:
            continue

        voxel = voxel_tensor.numpy()
        name = f"voxel_diff_pair_{pair_i.numpy()}_{pair_j.numpy()}.pdb"
        print(f"🔹 Writing PDB for {name}")

        write_single_pdb(
            voxel=voxel,
            output_folder=out_dir,
            name=name,
            rmax=50  # Will be estimated from reference PDB
        )

        if len(chosen_indices) == 1:
            break
        chosen_indices.remove(i)

    print(f"\n✅ Done: Converted 100 random voxel samples to PDBs in {out_dir}")


# ─── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    convert_random_samples_to_pdbs(
        tfrecord_path="/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/train.tfrecords",
        output_dir="/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/inspected_diffs",
        num_samples=100,
        seed=42
    )
