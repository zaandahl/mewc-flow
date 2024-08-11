import absl.logging
import os
# Suppress TensorFlow and absl logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
import subprocess
import numpy as np

def warmup_gpu(batch_size=1, img_size=224, num_classes=1000):
    # Create a dummy input with the same shape as your training data
    dummy_input = np.random.random((batch_size, img_size, img_size, 3)).astype(np.float32)
    dummy_labels = np.random.randint(0, num_classes, (batch_size,))
    # Build a simple model for warming up
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    # Compile the model with a simple loss and optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    # Run a few epochs with the dummy data
    model.fit(dummy_input, dummy_labels, epochs=5, verbose=0)
    print("GPU warmup completed.")

# Print TensorFlow and CUDA versions
print("TensorFlow Version:", tf.__version__)
print("CUDA Version:", tf.sysconfig.get_build_info()['cuda_version'])
print("CUDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])

# Check NUMA nodes
try:
    numa_info = subprocess.check_output(['numactl', '--hardware'])
    print("NUMA Node Information:\n", numa_info.decode())
except FileNotFoundError:
    print("numactl is not installed or not found.")

# Check GPUs
gpus = tf.config.list_physical_devices('GPU')
print('Num GPUs Available:', len(gpus))
for gpu in gpus:
    print('GPU:', gpu)

# Perform the GPU warmup
if gpus:
    warmup_gpu()

print("Done.")
