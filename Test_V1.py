import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import random
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from keras.models import load_model

###............     Setting up Logging       ............ 

logging.basicConfig(filename='Validation_evaluation_Version 1.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###............     Configuration       ............ 

# Defining directory paths
ROOT_FOLDER     = r"/home/u993985/Thesis/"
DATA_FOLDER     = ROOT_FOLDER + r"Data/"
WEIGHTS_FOLDER  = ROOT_FOLDER + r"Weights/"
RESULTS_FOLDER  = ROOT_FOLDER + r"Results/"
ANNOTATIONS_FOLDER = ROOT_FOLDER +  r"Annotations/"

# Spectrogram specifics
spectogram_h, spectogram_w, spectogram_d = 120, 160, 1
block_height, block_width, block_depth = 60, 80, 1

# Version description
modelname = str(f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}")
version = "Version 1"
weights_filename = str(f"{modelname}-{version}" + ".h5.keras")
weights_path = os.path.join(WEIGHTS_FOLDER, weights_filename)
cluster_filename = F'cluster_results - {version}.txt'

###............     Loading Data       ............ 

# Load the validation data
xy_test = np.load(os.path.join(DATA_FOLDER, "test.npy"))

# Normalize the validation data
def normalizing(dataset):
    dataset = dataset.astype(np.float32)
    dataset_normalized = np.array(dataset) / 255.0
    logger.info("Dataset normalized successfully.")
    return dataset_normalized

xy_test_normalized = normalizing(xy_test)


###............     Load Pre-trained Model       ............ 

# Load the trained model
model = load_model(weights_path)
logger.info("Model loaded successfully.")

###............     Splitting Spectrograms into Blocks       ............ 

def separating_spectrograms(dataset, block_height, block_width):
    block_dataset = []
    for img in dataset:
        if img.shape[0] % block_height != 0 or img.shape[1] % block_width != 0:
            logging.error("Warning: Image dimensions not divisible by block size. Adjusting block extraction.")

        for r in range(0, img.shape[0], block_height):  # Iterate over rows
            for c in range(0, img.shape[1], block_width):  # Iterate over columns
                if r + block_height <= img.shape[0] and c + block_width <= img.shape[1]:
                    block = img[r:r + block_height, c:c + block_width]
                    block_dataset.append(block)
                else:
                    logging.error("Warning: Block extraction out of bounds. Adjusting block extraction.")

    block_dataset = np.array(block_dataset)
    logger.info("Images split into blocks successfully.")
    return block_dataset

block_test = separating_spectrograms(xy_test_normalized, block_height, block_width)

###............     Predict and Reconstruct       ............ 

def predicting_full(block_dataset_normalized, model, spectogram_h, spectogram_w, block_height, block_width, block_depth):
    block_predicting = model.predict(block_dataset_normalized)
    N_blocks = (spectogram_h // block_height) * (spectogram_w // block_width)
    N_images = block_predicting.shape[0] // N_blocks

    block_predicting = block_predicting.reshape(N_images, N_blocks, block_height, block_width, block_depth)

    full_pred = []
    for n in block_predicting:
        result = np.zeros((spectogram_h, spectogram_w, block_depth))
        i = 0 
        for y in range(0, result.shape[0], block_height):
            for x in range(0, result.shape[1], block_width):
                res = n[i]
                result[y:y + block_height, x:x + block_width] = res
                i += 1
        full_pred.append(result)

    full_pred = np.array(full_pred) * 255.0
    logger.info("Full predictions are made.")
    
    return full_pred, block_predicting

# Perform prediction
full_pred, block_pred = predicting_full(block_test, model, spectogram_h, spectogram_w, block_height, block_width, block_depth)
results_path = os.path.join(RESULTS_FOLDER, f"predictions - {version}.npy")
np.save(results_path, full_pred)

"""
full_pred = np.load(os.path.join(RESULTS_FOLDER, f"predictions - {version}.npy"))
"""
###............     Calculate MSE       ............ 
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_total_mse(dataset, full_pred):
    total_mse = []
    for img1, img2 in zip(dataset, full_pred):
        mse_value = mse(img1, img2)
        total_mse.append(float(mse_value/255.0))
    return np.mean(total_mse)

mean_mse = calculate_total_mse(block_test, block_pred)
logger.info(f"Mean MSE of blocks: {mean_mse}")


mean_mse = calculate_total_mse(xy_test, full_pred)
logger.info(f"Mean MSE of full spectograms: {mean_mse}")

