# -*- coding: utf-8 -*-
"""
Inference helpers for the 3-D voxel auto-encoder (TF-2 / Keras)

* build_autoencoder() returns (auto, encoder, decoder)
* decode_latent()     : latent  → reconstruction
* encode_decode()     : voxel   → reconstruction
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

# ───────────────────────── configuration ─────────────────────────
Z_DIM      = 200        # latent size you asked for
VOX_SHAPE  = (32, 32, 32, 1)
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

    d = tf.keras.layers.Conv3D(1, 3, padding="same",
                               kernel_initializer="he_normal",
                               name="conv3d_final")(d)  # changed name here
    out = tf.keras.layers.Activation("tanh", name="tanh")(d)

    # full autoencoder model
    auto = tf.keras.Model(inp, out, name="voxel_autoencoder")
    auto.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="mse")

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
    out_dec = auto.get_layer("tanh")(x)
    decoder = tf.keras.Model(latent_in, out_dec, name="decoder")

    return auto, encoder, decoder



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
                  dtype=np.float32) -> np.ndarray:
    """
    Parameters
    ----------
    z_vectors : (N, Z_DIM) latent array
    weights_path : path to .weights.h5 or .keras saved in training
    Returns
    -------
    recon : (N, 32, 32, 32, 1) numpy array in range [-1, 1]
    """
    _, _, decoder = build_autoencoder()
    _load_weights(decoder, weights_path)
    recon = decoder.predict(z_vectors.astype(dtype), verbose=0)
    return recon


def encode_decode(voxels: np.ndarray,
                  weights_path: str | Path,
                  dtype=np.float32) -> np.ndarray:
    """
    Run full auto-encoder on input voxels.
    voxels : (N, 32, 32, 32, 1) array in same scaling as training
    Returns reconstructions of same shape.
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
    args = parser.parse_args()

    if args.latent:
        z = np.load(args.latent)
        rec = decode_latent(z, args.weights)
        out = Path(args.latent).with_suffix(".recon.npy")
        np.save(out, rec)
        print(f"Decoded {z.shape[0]} latent vectors → {out}")

    elif args.voxels:
        vx = np.load(args.voxels)
        rec = encode_decode(vx, args.weights)
        out = Path(args.voxels).with_suffix(".recon.npy")
        np.save(out, rec)
        print(f"Encoded+decoded {vx.shape[0]} grids → {out}")

    else:
        sys.exit("Provide --latent or --voxels input.")

