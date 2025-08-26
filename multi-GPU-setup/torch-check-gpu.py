import torch
# import tensorflow as tf
import time
import torch as tf
import keras
import pyopencl as cl


# keras.backend.set_backend("torch")

"""
import keras
keras.backend.set_backend("torch") 

try this
import torch as tf
"""

if __name__ == "__main__":
    # Check if a GPU is available
    print("Is CUDA available:", torch.cuda.is_available())
    # if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))


    def check_gpu_availability():
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                if device.type == cl.device_type.GPU:
                    print(f"GPU available: {device.name}")


        print("No GPU available.")
        return False


    def pytorch_function():
        x = torch.randn(50000, 50000).to('cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randn(50000, 50000).to('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.mm(x, y)


    def tensorflow_function():
        x = tf.random.normal((500, 500))
        y = tf.random.normal((500, 500))
        return tf.matmul(x, y)

    check_gpu_availability()


    # Timing the PyTorch function
    start_time = time.time()
    pytorch_result = pytorch_function()
    pytorch_duration = time.time() - start_time

    # Timing the TensorFlow function
    start_time = time.time()
    tensorflow_result = tensorflow_function()
    tensorflow_duration = time.time() - start_time

    print("PyTorch Duration:", pytorch_duration)
    print("TensorFlow Duration:", tensorflow_duration)