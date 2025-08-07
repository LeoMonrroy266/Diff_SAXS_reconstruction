#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference helpers for the 3-D voxel auto-encoder (TF-2 / Keras)

* build_autoencoder() returns (auto, encoder, decoder)
* decode_latent()     : latent  → reconstruction (probabilities or classes)
* encode_decode()     : voxel   → reconstruction
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

# ───────────────────────── configuration ─────────────────────────
Z_DIM      = 200        # latent size you asked for
VOX_SHAPE  = (32, 32, 32, 3)  # 3 classes output
K_INIT     = "he_normal"

# ─────────────────── model architecture (same as training) ──────
def _conv_block(x, filters, k=3, s=1):
    x = tf.keras.layers.Conv3D(filters, k, s, padding="same",
                               kernel_initializer=K_INIT)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def build_autoencoder():
    inp = tf.keras.Input(shape=(32, 32, 32, 1))

    # ---- encoder ----
    x = _conv_block(inp, 64)
    x = _conv_block(x, 64)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)

    x = _conv_block(x, 128)
    x = _conv_block(x, 128)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)

    x = _conv_block(x, 128)
    x = _conv_block(x, 128)
    x = _conv_block(x, 128)

    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(Z_DIM, activation="tanh",
                              kernel_initializer="he_normal",
                              name="latent")(x)

    # ---- decoder ----
    d = tf.keras.layers.Dense(8*8*8*32, activation="relu",
                              kernel_initializer="he_normal",
                              name="dense")(z)
    d = tf.keras.layers.Reshape((8, 8, 8, 32), name="reshape")(d)

    d = tf.keras.layers.Conv3DTranspose(64, 5, 2, padding="same",
                                        kernel_initializer="he_normal",
                                        name="conv3d_transpose")(d)
    d = tf.keras.layers.BatchNormalization(name="batch_norm")(d)
    d = tf.keras.layers.ReLU(name="relu")(d)

    d = tf.keras.layers.Conv3DTranspose(128, 5, 2, padding="same",
                                        kernel_initializer="he_normal",
                                        name="conv3d_transpose_1")(d)
    d = tf.keras.layers.BatchNormalization(name="batch_norm_1")(d)
    d = tf.keras.layers.ReLU(name="relu_1")(d)

    # Output layer with 3 channels (for 3 classes) + softmax activation
    d = tf.keras.layers.Conv3D(3, 3, padding="same",
                               kernel_initializer="he_normal",
                               name="conv3d_final")(d)
    out = tf.keras.layers.Activation("softmax", name="softmax")(d)

    # full autoencoder model
    auto = tf.keras.Model(inp, out, name="voxel_autoencoder")
    auto.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="categorical_crossentropy")

    # encoder model: input -> latent vector
    encoder = tf.keras.Model(inp, z, name="encoder")

    # decoder model: latent vector -> reconstruction
    latent_in = tf.keras.Input(shape=(Z_DIM,), name="latent_in")
    x = auto.get_layer("dense")(latent_in)
    x = auto.get_layer("reshape")(x)
    x = auto.get_layer("conv3d_transpose")(x)
    x = auto.get_layer("batch_norm")(x)
    x = auto.get_layer("relu")(x)
    x = auto.get_layer("conv3d_transpose_1")(x)
    x = auto.get_layer("batch_norm_1")(x)
    x = auto.get_layer("relu_1")(x)
    x = auto.get_layer("conv3d_final")(x)
    out_dec = auto.get_layer("softmax")(x)
    decoder = tf.keras.Model(latent_in, out_dec, name="decoder")

    return auto, encoder, decoder


# ─────────────────────── utility to convert classes ──────────────
def convert_class_indices_to_labels(class_array: np.ndarray) -> np.ndarray:
    """
    Convert class indices {0,1,2} to labels {-1,0,1}.
    """
    label_map = {0: -1, 1: 0, 2: 1}
    # Vectorized mapping:
    vectorized_map = np.vectorize(label_map.get)
    return vectorized_map(class_array)


# ─────────────────────── loading utilities ───────────────────────
def _load_weights(model, weights_path: Path | str):
    weights_path = Path(weights_path)
    if weights_path.suffix == ".keras":
        # full SavedModel / .keras
        loaded = tf.keras.models.load_model(weights_path, compile=False)
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(weights_path)


# ───────────────────────── API functions ─────────────────────────
def decode_latent(z_vectors: np.ndarray,
                  weights_path: str | Path,
                  dtype=np.float32,
                  hard_class: bool = False,
                  map_labels: bool = False) -> np.ndarray:
    """
    Decode latent vectors to voxel grid output.

    Parameters
    ----------
    z_vectors : (N, Z_DIM) latent array
    weights_path : path to .weights.h5 or .keras saved in training
    hard_class : if True, output hard class labels instead of probabilities
    map_labels : if True and hard_class=True, map class indices (0,1,2) to labels (-1,0,1)

    Returns
    -------
    voxels : (N, 32, 32, 32, [NUM_CLASSES or 1]) numpy array
        - if hard_class=False: shape (N, 32, 32, 32, 3) with probabilities
        - if hard_class=True and map_labels=False: shape (N, 32, 32, 32) with class indices 0..2
        - if hard_class=True and map_labels=True: shape (N, 32, 32, 32) with mapped labels (-1,0,1)
    """
    _, _, decoder = build_autoencoder()
    _load_weights(decoder, weights_path)
    probs = decoder.predict(z_vectors.astype(dtype), verbose=0)

    if not hard_class:
        return probs

    classes = np.argmax(probs, axis=-1)
    if map_labels:
        return convert_class_indices_to_labels(classes)
    return classes


def encode_decode(voxels: np.ndarray,
                  weights_path: str | Path,
                  dtype=np.float32) -> np.ndarray:
    """
    Run full auto-encoder on input voxels.
    voxels : (N, 32, 32, 32, 1) array in same scaling as training
    Returns reconstructions of same shape (N, 32, 32, 32, 3) as softmax probabilities.
    """
    auto, _, _ = build_autoencoder()
    _load_weights(auto, weights_path)
    return auto.predict(voxels.astype(dtype), verbose=0)


# ────────────────────────── example usage ────────────────────────
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Decode latent vectors or voxel grids.")
    parser.add_argument("weights", help="Path to weights (.weights.h5 or .keras)")
    parser.add_argument("--latent",  help="N×Z numpy file to decode")
    parser.add_argument("--voxels",  help="N×32×32×32×1 numpy file to encode+decode")
    parser.add_argument("--hard_class", action="store_true",
                        help="Return hard class indices instead of probabilities")
    parser.add_argument("--map_labels", action="store_true",
                        help="Map class indices {0,1,2} to labels {-1,0,1} (only if --hard_class used)")

    args = parser.parse_args()

    if args.latent:
        z = np.load(args.latent)
        rec = decode_latent(z, args.weights,
                            hard_class=args.hard_class,
                            map_labels=args.map_labels)
        # Choose output filename suffix depending on output type
        if args.hard_class and args.map_labels:
            suffix = ".recon_mapped_labels.npy"
        elif args.hard_class:
            suffix = ".recon_classes.npy"
        else:
            suffix = ".recon_probs.npy"
        out = Path(args.latent).with_suffix(suffix)
        np.save(out, rec)
        print(f"Decoded {z.shape[0]} latent vectors → {out}")

    elif args.voxels:
        vx = np.load(args.voxels)
        rec = encode_decode(vx, args.weights)
        out = Path(args.voxels).with_suffix(".recon_probs.npy")
        np.save(out, rec)
        print(f"Encoded+decoded {vx.shape[0]} grids → {out}")

    else:
        sys.exit("Provide --latent or --voxels input.")
