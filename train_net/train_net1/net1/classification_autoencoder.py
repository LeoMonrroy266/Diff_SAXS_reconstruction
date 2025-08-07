# coding=utf-8
import tensorflow as tf
import numpy as np
import os

# ────────────────────────────  CONFIG  ────────────────────────────
BATCH_SIZE     = 8
NUM_EPOCHS     = 15
SEED           = 56297
Z_DIM          = 3000          # size of latent space
AUTOTUNE       = tf.data.AUTOTUNE

# ─────────────────────  I/O paths (create if needed) ──────────────
cur_path              = os.getcwd()
folder_to_save_model  = os.path.join(cur_path, 'classification_model')
folder_to_save_log    = os.path.join(cur_path, 'classification_log')
os.makedirs(folder_to_save_model, exist_ok=True)
os.makedirs(folder_to_save_log,   exist_ok=True)
log_path = os.path.join(folder_to_save_log, 'log.txt')

# ────────────────────────  TF-Record parsing  ─────────────────────
def parse_example(serialized_example):
    feats = tf.io.parse_single_example(
        serialized_example,
        features={'data': tf.io.FixedLenFeature([32*32*32], tf.int64)})  # shape 32x32x32 as in original
    x = tf.reshape(feats['data'], (32, 32, 32, 1))
    x = tf.cast(x, tf.float32)
    # Shift labels from (-1,0,1) to (0,1,2) for classification if needed,
    # but here we assume input == label, adjust if you have separate label logic
    return x, tf.cast((x + 1), tf.int32)

def get_dataset(tfrecord, batch):
    ds = tf.data.TFRecordDataset(tfrecord)
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(6400, seed=SEED).batch(batch).prefetch(AUTOTUNE)
    return ds

# ─────────────────────────  Model building  ───────────────────────
def conv_block(x, filters, k=3, s=1):
    x = tf.keras.layers.Conv3D(filters, k, s, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def build_classifier_autoencoder():
    inp = tf.keras.Input(shape=(32, 32, 32, 1))

    # Encoder: same as original AE
    x = conv_block(inp, 64)
    x = conv_block(x, 64)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)  # 32->16

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)  # 16->8

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(Z_DIM, activation="tanh", kernel_initializer="he_normal")(x)

    # Decoder: same shape output as original (32,32,32)
    x = tf.keras.layers.Dense(8*8*8*32, activation="relu", kernel_initializer="he_normal")(z)
    x = tf.keras.layers.Reshape((8, 8, 8, 32))(x)

    # Upsample back to (16,16,16)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Upsample back to (32,32,32)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Final layer: output 3 channels for classification, no activation here (logits)
    logits = tf.keras.layers.Conv3D(3, 3, padding="same", kernel_initializer="he_normal")(x)
    out = tf.keras.layers.Softmax(axis=-1)(logits)

    model = tf.keras.Model(inp, out, name="voxel_classifier_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

    return model

# ───────────────────────────  Data  ───────────────────────────────
train_ds = get_dataset("/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/train.tfrecords", BATCH_SIZE)
test_ds  = get_dataset("/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/test.tfrecords", BATCH_SIZE)

# ──────────────────────────  Training  ────────────────────────────
model = build_classifier_autoencoder()
model.summary()

best_test_loss = float('inf')   # initialize best loss as very large

with open(log_path, "a") as logf:
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        hist = model.fit(train_ds,
                         epochs=1,
                         steps_per_epoch=50000 // BATCH_SIZE,
                         verbose=1)

        train_loss = hist.history["loss"][0]
        logf.write(f"Epoch {epoch}  Train-loss: {train_loss:.6f}\n")

        test_loss = model.evaluate(test_ds,
                                   steps=10000 // BATCH_SIZE,
                                   verbose=1)
        print(f"Epoch {epoch}  Test loss: {test_loss:.6f}")
        logf.write(f"Epoch {epoch}  Test-loss : {test_loss:.6f}\n")

        # Save only if test loss improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epoch_tag = f"best_epoch_{epoch:02d}"
            full_path = os.path.join(folder_to_save_model, f"model_{epoch_tag}.keras")
            ckpt_path = os.path.join(folder_to_save_model, f"weights_{epoch_tag}.weights.h5")

            # Save full model & weights
            model.save(full_path)
            model.save_weights(ckpt_path)

            print(f" *** Saved new best model with test loss {best_test_loss:.6f} ***")
            print(f"     Full model: {full_path}")
            print(f"     Weights   : {ckpt_path}")

print("Training complete.")
