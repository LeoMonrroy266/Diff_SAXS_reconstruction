# -*- coding: utf-8 -*-
"""
Inference helpers for the 3-D voxel auto-encoder (TF-2 / Keras)
Network matches training code (no BatchNorm, logits output),
with optional sigmoid post-processing for outputs.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

# ───────────────────────── configuration ─────────────────────────
Z_DIM = 200           # Latent space size
VOX_SHAPE = (32, 32, 32, 1)
K_INIT = "he_normal"

# ─────────────────── model architecture (training style) ──────
def build_encoder(z_dim=Z_DIM):
    inputs = tf.keras.Input(shape=VOX_SHAPE)
    x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_dim, activation='relu')(x)
    return tf.keras.Model(inputs, z, name='encoder')

def build_decoder(z_dim=Z_DIM):
    latent_inputs = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(8*8*8*32, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 8, 32))(x)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, padding='same', activation='relu')(x)
    logits = tf.keras.layers.Conv3D(1, 3, padding='same')(x)  # logits
    return tf.keras.Model(latent_inputs, logits, name='decoder')

def build_autoencoder(z_dim=Z_DIM):
    encoder = build_encoder(z_dim)
    decoder = build_decoder(z_dim)
    inputs = tf.keras.Input(shape=VOX_SHAPE)
    outputs = decoder(encoder(inputs))
    auto = tf.keras.Model(inputs, outputs, name='autoencoder')
    auto.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return auto, encoder, decoder

# ─────────────────────── loading utilities ───────────────────────
def _load_weights(model, weights_path: Path | str):
    weights_path = Path(weights_path)

    if weights_path.is_dir():
        keras_files = list(weights_path.glob("*.keras"))
        h5_files = list(weights_path.glob("*.h5"))
        if keras_files:
            weights_path = keras_files[0]
        elif h5_files:
            weights_path = h5_files[0]
        else:
            ckpt = tf.train.Checkpoint(model=model)
            latest = tf.train.latest_checkpoint(weights_path)
            if latest is None:
                raise FileNotFoundError(f"No checkpoint found in {weights_path}")
            ckpt.restore(latest).expect_partial()
            return

    elif weights_path.suffix == "":
        keras_candidate = weights_path.with_suffix(".keras")
        h5_candidate = weights_path.with_suffix(".h5")
        ckpt_candidate = weights_path
        if keras_candidate.exists():
            weights_path = keras_candidate
        elif h5_candidate.exists():
            weights_path = h5_candidate
        elif (ckpt_candidate.with_suffix(".index")).exists():
            reader = tf.train.load_checkpoint(str(ckpt_candidate))
            var_map = {var.name: var for var in model.variables}
            for tf1_name, var in var_map.items():
                try:
                    var.assign(reader.get_tensor(tf1_name))
                except Exception:
                    pass
            return
        else:
            raise FileNotFoundError(f"No valid weights found for {weights_path}")

    if weights_path.suffix == ".keras":
        loaded = tf.keras.models.load_model(weights_path, compile=False)
        model.set_weights(loaded.get_weights())
    elif weights_path.suffix == ".h5":
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    else:
        if (weights_path.with_suffix(".index")).exists():
            reader = tf.train.load_checkpoint(str(weights_path))
            var_map = {var.name: var for var in model.variables}
            for tf1_name, var in var_map.items():
                try:
                    var.assign(reader.get_tensor(tf1_name))
                except Exception:
                    pass
        else:
            raise ValueError(f"Unsupported weights file format: {weights_path.suffix}")

# ───────────────────────── API functions ─────────────────────────
def decode_latent(z_vectors: np.ndarray,
                  weights_path: str | Path,
                  use_sigmoid=True,
                  dtype=np.float32) -> np.ndarray:
    _, _, decoder = build_autoencoder()
    _load_weights(decoder, weights_path)
    logits = decoder.predict(z_vectors.astype(dtype), verbose=0)
    return tf.sigmoid(logits).numpy() if use_sigmoid else logits

def encode_voxels(voxels: np.ndarray,
                  weights_path: str | Path,
                  dtype=np.float32) -> np.ndarray:
    _, encoder, _ = build_autoencoder()
    _load_weights(encoder, weights_path)
    return encoder.predict(voxels.astype(dtype), verbose=0)

def encode_decode(voxels: np.ndarray,
                  weights_path: str | Path,
                  use_sigmoid=True,
                  dtype=np.float32) -> np.ndarray:
    auto, _, _ = build_autoencoder()
    _load_weights(auto, weights_path)
    logits = auto.predict(voxels.astype(dtype), verbose=0)
    return tf.sigmoid(logits).numpy() if use_sigmoid else logits
