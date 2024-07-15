# Import necessary dependencies
import os
import librosa
import numpy as np
import librosa.display
import pylab
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import shutil
import random
import logging


# Setting Paths
ROOT_FOLDER = r"D:/Menno/Thesis/Expirement/"
DATA_FOLDER = r"Data/"
MIC_FOLDER = r"mic_south/"
SPECTOGRAM_FOLDER  = r'Spectograms/'

# Setting Logging function
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Defining necessary paths and define global directory paths
def set_root_paths(ROOT_FOLDER, DATA_FOLDER, SPECTOGRAM_FOLDER, MIC_FOLDER ):
    """
    ......................................................................
    Parameters:
    - ROOT_FOLDER       = directory     -> to the main enviroment
    - DATA_FOLDER       = string        -> folder name, where the orginal 3 second audio files are located in
    - SPECTOGRAM_FOLDER = string        -> folder name, where the spectograms are located
    - MIC_FOLDER        = string        -> folder name, where the orginal 3 second audio files are located for a specific mic position
    ......................................................................
    
    """
        
    global ROOT_PATH, DATA_PATH , SPECTOGRAM_PATH, MIC_PATH
    
    ROOT_PATH           = str(ROOT_FOLDER)
    DATA_PATH           = ROOT_PATH + str(DATA_FOLDER)
    MIC_PATH            = DATA_PATH + str(MIC_FOLDER)
    SPECTOGRAM_PATH     = ROOT_PATH + str(SPECTOGRAM_FOLDER)

    logger.info("Root paths are set.")
    
set_root_paths( ROOT_FOLDER, DATA_FOLDER, SPECTOGRAM_FOLDER, MIC_FOLDER )

# Creat a function that splits wav audio files in train and test
# The split should be 70 % and 30 % 

import shutil
import random

def split_spectrograms(source_folder, val_folder, test_folder, train_ratio=0.5, seed = 49):
    
    random.seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all files from the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle the files randomly
    random.shuffle(all_files)
    logger.info("All files are shuffeld.")
    
    # Calculate split index
    split_index = int(len(all_files) * train_ratio)
    
    # Split files into training and testing sets
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]
    logger.info("Split is realised.")
    
    # Move files to respective directories
    for file in train_files:
        output_train_file_path = os.path.join(val_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(val_folder, file))
        print(f"Moved {file} to {output_train_file_path}")
    
    for file in test_files:
        output_test_file_path = os.path.join(test_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))
        print(f"Moved {file} to {output_test_file_path }")
    
    print(f"Moved {len(train_files)} files to {val_folder}")
    print(f"Moved {len(test_files)} files to {test_folder}")

# Define paths

SPECTOGRAM_VAL_PATH = SPECTOGRAM_PATH + "Validation/"
SPECTOGRAM_TEST_PATH  = SPECTOGRAM_PATH + "Test/"

SPECTOGRAM_NORTH_PATH  = SPECTOGRAM_PATH + "North/"
SPECTOGRAM_SOUTH_PATH  = SPECTOGRAM_PATH + "South/"

# Split the spectrograms
split_spectrograms(SPECTOGRAM_NORTH_PATH, SPECTOGRAM_VAL_PATH, SPECTOGRAM_TEST_PATH , train_ratio=0.5, seed = 49)

split_spectrograms(SPECTOGRAM_SOUTH_PATH, SPECTOGRAM_VAL_PATH, SPECTOGRAM_TEST_PATH , train_ratio=0.5, seed = 49)
