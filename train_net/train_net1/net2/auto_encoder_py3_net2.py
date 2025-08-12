# coding: utf-8
"""
3-D voxel auto-encoder – step 2 (Z_DIM=200)
-------------------------------------------
* Loads encoder weights from previous Z_DIM=3000 model
* Trains with the same dataset setup and logging style as step 1
"""
import os, glob
import numpy as np
import tensorflow as tf

# ───── config ────────────────────────────────────────────────────
BATCH_SIZE        = 8
NUM_EPOCHS        = 15
SEED              = 56297
Z_DIM             = 200         # step 2 latent size
OLD_Z_DIM         = 3000        # step 1 latent size
TRAIN_SAMPLES     = 300_000     # per training epoch
TEST_SAMPLES      = 10_000
AUTOTUNE          = tf.data.AUTOTUNE

cur_path              = os.getcwd()
folder_to_save_model  = os.path.join(cur_path, 'model_normal_pisa_step2')
folder_to_save_log    = os.path.join(cur_path, 'log')
os.makedirs(folder_to_save_model, exist_ok=True)
os.makedirs(folder_to_save_log,   exist_ok=True)
log_path = os.path.join(folder_to_save_log, 'log.txt')

# ───── dataset helpers ───────────────────────────────────────────
def parse_example(serialized_example):
    feats = tf.io.parse_single_example(
        serialized_example,
        features={'data': tf.io.FixedLenFeature([32*32*32], tf.int64)})
    x = tf.reshape(feats['data'], (32, 32, 32, 1))
    x = tf.cast(x, tf.float32)
    return x, x  # (input, label) for autoencoder

def get_dataset_from_dir(tfrecord_dir, batch, shuffle_buf=6400):
    tfrecord_files = glob.glob(os.path.join(tfrecord_dir, "*.tfrecords"))
    if not tfrecord_files:
        raise ValueError(f"No .tfrecords files found in {tfrecord_dir}")

    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(batch, drop_remainder=True).prefetch(AUTOTUNE)
    return ds

train_dir = "/home/leonardo/SAXS_reconstruction/changed_network/train_net/tfrecord_chunks/train"
test_dir  = "/home/leonardo/SAXS_reconstruction/changed_network/train_net/tfrecord_chunks/test"

train_ds = get_dataset_from_dir(train_dir, BATCH_SIZE)
test_ds  = get_dataset_from_dir(test_dir,  BATCH_SIZE)

# ───── model ─────────────────────────────────────────────────────
def conv_block(x, filters, k=3, s=1):
    x = tf.keras.layers.Conv3D(filters, k, s, padding="same",
                               kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def build_autoencoder(z_dim):
    inp = tf.keras.Input(shape=(32, 32, 32, 1))

    # ---- encoder ----
    x = conv_block(inp, 64)
    x = conv_block(x, 64)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_dim, activation="tanh",
                              kernel_initializer="he_normal")(x)

    # ---- decoder ----
    x = tf.keras.layers.Dense(8*8*8*32, activation="relu",
                              kernel_initializer="he_normal")(z)
    x = tf.keras.layers.Reshape((8, 8, 8, 32))(x)

    x = tf.keras.layers.Conv3DTranspose(64, 5, 2, padding="same",
                                        kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv3DTranspose(128, 5, 2, padding="same",
                                        kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv3D(1, 3, padding="same",
                               kernel_initializer="he_normal")(x)
    out = tf.keras.activations.tanh(x)

    model = tf.keras.Model(inp, out, name="voxel_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="mse")
    return model

# Build current model (Z_DIM=200)
model = build_autoencoder(Z_DIM)

# Load old model to copy encoder weights
old_model = build_autoencoder(OLD_Z_DIM)
old_model.load_weights('/home/leonardo/SAXS_reconstruction/changed_network/train_net/train_net1/net1/model/weights_best_epoch_15.weights.h5')

# Copy encoder weights where names match
for l1, l2 in zip(old_model.layers, model.layers):
    if l1.name == l2.name:
        try:
            l2.set_weights(l1.get_weights())
            print(f"Copied weights for layer: {l1.name}")
        except Exception as e:
            print(f"Skipping layer {l1.name} due to mismatch: {e}")

model.summary()

# ───── training loop ─────────────────────────────────────────────
steps_per_train_epoch = TRAIN_SAMPLES // BATCH_SIZE
steps_per_test        = TEST_SAMPLES  // BATCH_SIZE

best_test_loss = float('inf')

with open(log_path, "a") as logf:
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        hist = model.fit(
            train_ds,
            epochs=1,
            steps_per_epoch=steps_per_train_epoch,
            verbose=1
        )

        train_loss = hist.history["loss"][0]
        logf.write(f"Epoch {epoch}  Train-loss: {train_loss:.6f}\n")

        test_loss = model.evaluate(
            test_ds,
            steps=steps_per_test,
            verbose=1
        )
        print(f"Epoch {epoch}  Test loss: {test_loss:.6f}")
        logf.write(f"Epoch {epoch}  Test-loss : {test_loss:.6f}\n")

        if test_loss < best_test_loss:
            if 'best_ckpt_path' in locals() and os.path.exists(best_ckpt_path):
                try:
                    os.remove(best_ckpt_path)
                    os.remove(best_full_path)
                    print(f"Removed previous best model: {best_full_path}")
                    print(f"Removed previous best weights: {best_ckpt_path}")
                except Exception as e:
                    print(f"Warning: could not remove old best model: {e}")

            best_test_loss = test_loss
            epoch_tag = f"best_epoch_{epoch:02d}"
            full_path = os.path.join(folder_to_save_model, f"model_{epoch_tag}.keras")
            ckpt_path = os.path.join(folder_to_save_model, f"weights_{epoch_tag}.weights.h5")

            model.save(full_path)
            model.save_weights(ckpt_path)

            best_full_path = full_path
            best_ckpt_path = ckpt_path

            print(f" *** Saved new best model with test loss {best_test_loss:.6f} ***")
            print(f"     Full model: {full_path}")
            print(f"     Weights   : {ckpt_path}")
