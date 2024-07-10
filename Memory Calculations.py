import numpy as np
import psutil


def estimate_memory_requirements(shape, dtype=np.uint8, num_images=1, overhead_factor=1.1):
    """
    shape               (int)            Desired input schape in height width, channels
    num_images          (int)            The number of images
    overhead_factor     (float)          The factor that should be used to calulate more memory for safety    
    """ 
    item_size = np.dtype(dtype).itemsize
    single_image_size = np.prod(shape) * item_size
    total_size = single_image_size * np.int64(num_images)  
    total_size_with_overhead = total_size * overhead_factor
    return total_size_with_overhead

# Example usage for an RGB image with shape (height, width, channels)
image_shape = (120, 160, 3)
num_images = 603315
total_memory_required = estimate_memory_requirements(image_shape, num_images=num_images)
print(f"Estimated total memory required for {num_images} images: {total_memory_required / (1024 ** 3):.2f} GB")

# Check available memory
available_memory = psutil.virtual_memory().available
print(f"Available system memory: {available_memory / (1024 ** 3):.2f} GB")

# Determine if the available memory is sufficient
if total_memory_required > available_memory:
    print("Warning: Not enough available memory to load all images at once.")
else:
    print("Sufficient memory available to load all images.")
