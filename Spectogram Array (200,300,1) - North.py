import os
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_image(file_path: str) -> np.ndarray:
    """
    Load an image, resize it to 300x200, convert it to grayscale, and return it as a numpy array.

    Parameters:
    file_path (str): Path to the image file.

    Returns:
    np.ndarray: The processed image as a numpy array.
    """
    with Image.open(file_path) as img:
        img_resized = img.resize((160, 120))
        img_gray = img_resized.convert('L')
        img_array = np.array(img_gray)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array

def process_and_save_batch(file_paths_batch, batch_index, batch_dir):
    """
    Process a batch of images and save them to a temporary file.

    Parameters:
    file_paths_batch (list): List of image file paths to process.
    batch_index (int): Index of the current batch.
    batch_dir (str): Directory to save the temporary batch files.
    """
    with ThreadPoolExecutor() as executor:
        images_batch = list(tqdm(executor.map(process_image, file_paths_batch), total=len(file_paths_batch), desc=f"Processing batch {batch_index}"))
    
    batch_array = np.array(images_batch)
    batch_file = os.path.join(batch_dir, f"batch_{batch_index}.npy")
    np.save(batch_file, batch_array)

def combine_batches(batch_dir, output_file, num_batches):
    """
    Combine saved batches into a single .npy file.

    Parameters:
    batch_dir (str): Directory containing the batch files.
    output_file (str): Path to the final combined .npy file.
    num_batches (int): Number of batches to combine.
    """
    combined_array = []

    for i in range(num_batches):
        batch_file = os.path.join(batch_dir, f"batch_{i}.npy")
        batch_array = np.load(batch_file)
        
        combined_array.append(batch_array)
    
    final_array = np.concatenate(combined_array, axis=0)
    np.save(output_file, final_array)

def convert_images_to_numpy_array(image_dir: str, output_file: str, batch_size: int, max_workers: int = 4) -> None:
    """
    Convert all PNG images in the specified directory to a numpy array in batches and save it.

    Parameters:
    image_dir (str): Path to the directory containing the PNG images.
    output_file (str): Path where the numpy array will be saved.
    batch_size (int): Number of images to process in each batch.
    max_workers (int, optional): Maximum number of worker threads. Defaults to 4.
    """
    print(f"The following spectrogram directory is used: {image_dir}")
    print(f"The following output directory is used: {output_file}")
    # Create a temporary directory to store batch files
    batch_dir = os.path.join(image_dir, "Batch")
    os.makedirs(batch_dir, exist_ok=True)

    # Get a list of all PNG files in the directory containing "North" in their filename
    file_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png') and "North" in filename]
    file_paths = sorted(file_paths)
    print("First five filenames:", file_paths[:5])

    len_file_paths = len(file_paths)
    print(f'The number of files are: {len_file_paths}')

    num_batches = (len(file_paths) + batch_size - 1) // batch_size

    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(file_paths))
        file_paths_batch = file_paths[batch_start:batch_end]
        process_and_save_batch(file_paths_batch, batch_index, batch_dir)
    
    
    combine_batches(batch_dir, output_file, num_batches)

    # Remove the temporary directory and batch files
    for filename in os.listdir(batch_dir):
        os.remove(os.path.join(batch_dir, filename))
    os.rmdir(batch_dir)

    print(f"Images saved to {output_file}")
    

# Example usage
image_directory = "D:/Menno/Thesis/Expirement/Spectograms/Test/"
output_file_path = "D:/Menno/Thesis/Expirement/Spectograms/Denoising/test.npy"

batch_size = 200000  # Adjust based on available memory and number of images
convert_images_to_numpy_array(image_directory, output_file_path, batch_size=batch_size)
