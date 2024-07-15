import numpy as np
import logging
import os

# Setting up logging
logging.basicConfig(filename='normalize_data.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: RSS = {mem_info.rss / (1024 * 1024)} MB, VMS = {mem_info.vms / (1024 * 1024)} MB")

def normalize_array(input_file, output_file):
    # Load the numpy array
    logger.info(f"Loading data from {input_file}")
    data = np.load(input_file)
    log_memory_usage()

    # Normalize the array
    logger.info(f"Normalizing data")
    data_normalized = data.astype(np.float32) / 255.0
    log_memory_usage()

    # Save the normalized array
    logger.info(f"Saving normalized data to {output_file}")
    np.save(output_file, data_normalized)
    log_memory_usage()

# Example usage
input_file = "/home/u993985/Thesis/Data/200,300/test.npy"
output_file = '/home/u993985/Thesis/Data/200,300/test_normalized.npy'  
normalize_array(input_file, output_file)