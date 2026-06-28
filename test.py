import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs available:", len(gpus))

if gpus:
    print("GPU is AVAILABLE")
    for gpu in gpus:
        print("  -", gpu.name)
else:
    print("GPU is NOT available (running on CPU)")
