# coding: utf-8
"""
3-D voxel classifier auto-encoder – TensorFlow 2/Keras re-implementation
------------------------------------------------------------
* Reads 32×32×32 voxel grids from TFRecord files
* Encoder–decoder with 3-class output per voxel (-1,0,1) classification
* Loads encoder weights from a bigger pretrained model (latent size 3000)
* Trains for NUM_EPOCHS, keeps best checkpoint
"""
import os, tensorflow as tf

# ───── config ────────────────────────────────────────────────────
BATCH_SIZE     = 8
NUM_EPOCHS     = 15
SEED           = 56297
Z_DIM          = 200  # latent space size for this smaller model
OLD_Z_DIM      = 3000 # latent size of pretrained bigger model
AUTOTUNE       = tf.data.AUTOTUNE

cur_path              = os.getcwd()
folder_to_save_model  = os.path.join(cur_path, 'classification_model')
folder_to_save_log    = os.path.join(cur_path, 'classification_log')
os.makedirs(folder_to_save_model, exist_ok=True)
os.makedirs(folder_to_save_log,   exist_ok=True)
log_path = os.path.join(folder_to_save_log, 'log.txt')

train_tfrecord = "/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/train.tfrecords"
test_tfrecord  = "/home/leonardo/SAXS_reconstruction/changed_network/train_net/dataset/LOV2/test.tfrecords"

# ───── dataset helpers ───────────────────────────────────────────
def _parse(serial):
    feat = tf.io.parse_single_example(
        serial, {"data": tf.io.FixedLenFeature([32*32*32], tf.int64)})
    x = tf.reshape(feat["data"], (32, 32, 32, 1))
    # Convert from float to int and shift classes: -1,0,1 -> 0,1,2 for classification
    x_class = tf.cast(x, tf.int32) + 1
    x_class = tf.clip_by_value(x_class, 0, 2)  # safety clamp
    return tf.cast(x, tf.float32), x_class  # input float, label classes

def make_ds(tfr, batch):
    ds = tf.data.TFRecordDataset(tfr)
    ds = ds.map(_parse,  num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(6400, seed=SEED).batch(batch).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_tfrecord, BATCH_SIZE)
test_ds  = make_ds(test_tfrecord,  BATCH_SIZE)

# ───── model ─────────────────────────────────────────────────────
def conv_block(x, filters, k=3, s=1):
    x = tf.keras.layers.Conv3D(filters, k, s, padding="same",
                               kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def build_classifier_autoencoder(z_dim):
    inp = tf.keras.Input(shape=(32, 32, 32, 1))

    # ---- encoder ----
    x = conv_block(inp, 64); x = conv_block(x, 64)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)

    x = conv_block(x, 128); x = conv_block(x, 128)
    x = tf.keras.layers.MaxPooling3D(2, padding="same")(x)

    x = conv_block(x, 128); x = conv_block(x, 128); x = conv_block(x, 128)

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

    # Output layer: 3 channels = 3 classes logits per voxel
    logits = tf.keras.layers.Conv3D(3, 3, padding="same",
                                    kernel_initializer="he_normal")(x)
    out = tf.keras.layers.Softmax(axis=-1)(logits)

    model = tf.keras.Model(inp, out, name="voxel_classifier_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    return model

# ───── build and compile small model ──────────────────────────────
model = build_classifier_autoencoder(Z_DIM)
model.summary()

# ───── load encoder weights from old bigger model ────────────────
old_model = build_classifier_autoencoder(OLD_Z_DIM)
old_weights_path = '/home/leonardo/SAXS_reconstruction/changed_network/train_net/train_net1/net1/model/weights_best_epoch_15.weights.h5'  # update path accordingly
old_model.load_weights(old_weights_path)

# Copy weights where layer names match
for old_layer, new_layer in zip(old_model.layers, model.layers):
    if old_layer.name == new_layer.name:
        try:
            new_layer.set_weights(old_layer.get_weights())
            print(f"Copied weights for layer: {old_layer.name}")
        except Exception as e:
            print(f"Skipping layer {old_layer.name} due to mismatch: {e}")

# ───── training loop with manual logging ──────────────────────────
best_test_loss = float('inf')   # initialize best loss as very large
with open(log_path, "a") as logf:
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        hist = model.fit(train_ds,
                         epochs          = 1,
                         steps_per_epoch = 50000 // BATCH_SIZE,
                         verbose         = 1)
        train_loss = hist.history["loss"][0]
        logf.write(f"Epoch {epoch}  Train-loss {train_loss:.6f}\n")

        test_loss = model.evaluate(test_ds,
                                   steps   = 10000 // BATCH_SIZE,
                                   verbose = 1)
        print(f"Epoch {epoch}  Test-loss {test_loss:.6f}")
        logf.write(f"Epoch {epoch}  Test-loss  {test_loss:.6f}\n")

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

print("Training finished.")
