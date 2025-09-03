import tensorflow as tf
import os
import sys

# -------------------
# Hyperparameters
# -------------------
BATCH_SIZE = 64
NUM_EPOCHS = 15
SEED = 56297
z_dim = 3000
input_shape = (32, 32, 32, 1)
LEARNING_RATE = 0.0005

cur_path = "/home/leonardo/testing_saxs/training_data_LOV2"
folder_to_save_model = os.path.join(cur_path, 'model_1')
folder_to_save_log = os.path.join(cur_path, 'log')
os.makedirs(folder_to_save_model, exist_ok=True)
os.makedirs(folder_to_save_log, exist_ok=True)
log_path = os.path.join(folder_to_save_log, 'log.txt')

# -------------------
# TFRecord parser
# -------------------
def parse_tfrecord(example_proto):
    features = {'data': tf.io.FixedLenFeature([32768], tf.int64)}
    parsed = tf.io.parse_single_example(example_proto, features)

    # Input for network
    data = tf.cast(tf.reshape(parsed['data'], input_shape), tf.float32)

    # Labels for SparseCategoricalCrossentropy
    label = tf.cast(tf.reshape(parsed['data'], input_shape), tf.int32)
    label = label + 1  # -1->0, 0->1, 1->2
    label = tf.squeeze(label, axis=-1)  # remove channel dimension for labels

    return data, label

def load_dataset(tfrecord_path, batch_size, shuffle_buffer=3200):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer, seed=SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

# -------------------
# Encoder
# -------------------
def build_encoder():
    inputs = tf.keras.Input(shape=input_shape)
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

# -------------------
# Decoder
# -------------------
def build_decoder():
    latent_inputs = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(8 * 8 * 8 * 32, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 8, 32))(x)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, padding='same', activation='relu')(x)

    # 3 channels for 3 voxel classes
    outputs = tf.keras.layers.Conv3D(3, 3, padding='same')(x)  # logits
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

# -------------------
# Load datasets
# -------------------
train_set = os.path.join(cur_path, 'train_net/train_data/train/train.tfrecords')
test_set = os.path.join(cur_path, 'train_net/train_data/test/test.tfrecords')

train_dataset = load_dataset(train_set, BATCH_SIZE)
test_dataset = load_dataset(test_set, BATCH_SIZE)

# Count samples (one per tfrecord entry)
num_train_samples = sum(1 for _ in tf.data.TFRecordDataset(train_set))
num_test_samples  = sum(1 for _ in tf.data.TFRecordDataset(test_set))

steps_per_epoch   = num_train_samples // BATCH_SIZE
validation_steps  = num_test_samples // BATCH_SIZE


# -------------------
# Build Autoencoder
# -------------------
encoder = build_encoder()
decoder = build_decoder()
inputs = tf.keras.Input(shape=input_shape)
outputs = decoder(encoder(inputs))
autoencoder = tf.keras.Model(inputs, outputs, name="autoencoder")

# -------------------
# Compile
# -------------------
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

)

# -------------------
# Logging callback (per epoch)
# -------------------
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch+1}, Training Loss: {logs.get('loss',0):.4f}, Validation Loss: {logs.get('val_loss',0):.4f}"
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

# -------------------
# Checkpoint callback (best weights)
# -------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(folder_to_save_model, "best_model_weights.h5"),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    save_format='h5'
)

# -------------------
# Train the model
# -------------------
autoencoder.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=NUM_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[EpochLogger(), checkpoint_cb]
)

