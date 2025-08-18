# coding=utf-8
import tensorflow as tf
import numpy as np
import glob
import os

# ────────────────────────────  CONFIG  ────────────────────────────
BATCH_SIZE        = 16
NUM_EPOCHS        = 15
SEED              = 56297
Z_DIM             = 3000
# Need to adjust sample size based on training data
TRAIN_SAMPLES     = 50_000    # number of samples per training epoch
TEST_SAMPLES      = 10_000   # number of samples for test evaluation
AUTOTUNE          = tf.data.AUTOTUNE

# ─────────────────────  I/O paths (create if needed) ──────────────
cur_path              = os.getcwd()
folder_to_save_model  = os.path.join(cur_path, 'model_normal_pisa')
folder_to_save_log    = os.path.join(cur_path, 'log')
os.makedirs(folder_to_save_model, exist_ok=True)
os.makedirs(folder_to_save_log,   exist_ok=True)
log_path = os.path.join(folder_to_save_log, 'log.txt')

# ────────────────────────  TF-Record parsing  ─────────────────────
def parse_example(serialized_example):
    feats = tf.io.parse_single_example(
        serialized_example,
        features={'data': tf.io.FixedLenFeature([32*32*32], tf.int64)})
    x = tf.reshape(feats['data'], (32, 32, 32, 1))
    x = tf.cast(x, tf.float32)
    return x, x  # autoencoder ⇒ (input, label)

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

# ─────────────────────────  Model building  ───────────────────────
def conv_block(x, filters, k=3, s=1):
    x = tf.keras.layers.Conv3D(filters, k, s, padding="same",
                               kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def build_model():
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
    z = tf.keras.layers.Dense(Z_DIM, activation="relu",
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
    #out = tf.keras.activations.sigmoid(x)  # Used when training on normal data binary
    out = tf.keras.activations.tanh(x)   # Used when training on diff data, 0, 1 and -1
    model = tf.keras.Model(inp, out, name="voxel_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy")
    return model

# ───────────────────────────  Data  ───────────────────────────────
train_dir = "/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/PISA_aligned/voxels_structures/train"
test_dir  = "/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/PISA_aligned/voxels_structures/test"

train_ds = get_dataset_from_dir(train_dir, BATCH_SIZE)
test_ds  = get_dataset_from_dir(test_dir,  BATCH_SIZE)

# ──────────────────────────  Training  ────────────────────────────
model = build_model()
model.summary()

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

        # Save only if test loss improved
        if test_loss < best_test_loss:
            # Remove previous best weights if they exist
            if 'best_ckpt_path' in locals() and os.path.exists(best_ckpt_path):
                try:
                    os.remove(best_ckpt_path)
                    os.remove(best_full_path)
                    print(f"Removed previous best model: {best_full_path}")
                    print(f"Removed previous best weights: {best_ckpt_path}")
                except Exception as e:
                    print(f"Warning: could not remove old best model: {e}")

            # Update best loss
            best_test_loss = test_loss
            epoch_tag = f"best_epoch_{epoch:02d}"
            full_path = os.path.join(folder_to_save_model, f"model_{epoch_tag}.keras")
            ckpt_path = os.path.join(folder_to_save_model, f"weights_{epoch_tag}.weights.h5")

            # Save new best model and weights
            model.save(full_path)
            model.save_weights(ckpt_path)

            # Keep track of paths for deletion next time
            best_full_path = full_path
            best_ckpt_path = ckpt_path

            print(f" *** Saved new best model with test loss {best_test_loss:.6f} ***")
            print(f"     Full model: {full_path}")
            print(f"     Weights   : {ckpt_path}")

