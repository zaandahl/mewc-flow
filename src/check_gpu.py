import tensorflow as tf
import subprocess

# Print TensorFlow and CUDA versions
print("TensorFlow Version:", tf.__version__)
print("CUDA Version:", tf.sysconfig.get_build_info()['cuda_version'])

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

print("Done.")

