
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


# Make from these audio files spectograms

def create_mel_spec(MIC_PATH, SPECTOGRAM_PATH):   
    
    # Process each audio file in the directory
    for file in os.listdir(MIC_PATH):
        if file.endswith('.wav'):
            
            # Directory path to audio files
            audio_path = os.path.join(MIC_PATH, file)  # Proper path handling
            
            # Load the audio file with librosa
            sig, sr = librosa.load(audio_path)

            # Create spectograms
            S = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=1024, fmin=800, fmax = (sr//2),  n_mels=128 , hop_length=512)   
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Set up the figure for plotting with no axes
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])
            
            # Define the filename for the output image
            name_prefix = file[:-4]
            output_name = f"{name_prefix}.png"
            output_file_path = os.path.join(SPECTOGRAM_PATH, output_name)
            print(f"Moved {file} to {output_file_path }")
            # Save the figure
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, format='png')
            plt.close()  # Close the figure to free up memory
            
    logging.info(f" All the Mel spectrograms are saved: {SPECTOGRAM_PATH}")

create_mel_spec(MIC_PATH, SPECTOGRAM_PATH) 



# Creat a function that splits wav audio files in train and test
# The split should be 70 % and 30 % 

import shutil
import random

def split_spectrograms(source_folder, train_folder, test_folder, train_ratio=0.7, seed = 49):
    
    random.seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
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
        output_train_file_path = os.path.join(train_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))
        print(f"Moved {file} to {output_train_file_path}")
    
    for file in test_files:
        output_test_file_path = os.path.join(test_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))
        print(f"Moved {file} to {output_test_file_path }")
    
    print(f"Moved {len(train_files)} files to {train_folder}")
    print(f"Moved {len(test_files)} files to {test_folder}")

# Define paths
SPECTOGRAM_TRAIN_PATH = SPECTOGRAM_PATH + "Train/"
SPECTOGRAM_TEST_PATH  = SPECTOGRAM_PATH + "Test/"

# Split the spectrograms
split_spectrograms(SPECTOGRAM_PATH, SPECTOGRAM_TRAIN_PATH, SPECTOGRAM_TEST_PATH , train_ratio=0.70, seed = 49)


# Defining function for splitting the test subset into two subsets of validation and test.
def split_spectograms_validation(source_folder, validation_folder, validation_ratio=0.5, seed = 49):
    """
    ......................................................................
    Parameters:
    - source_folder         = directory -> where the spectograms are located, in this case the test folder
    - validation_folder     = directory -> where the validations file need to be located to
    - validation_ratio      = float     -> The ratio's of split wihtin the test folder after the training and test files are splitted
    ......................................................................
    
    """  
    
    #Setting seed
    random.seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(validation_folder, exist_ok=True)

    # Get all files from the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle the files randomly
    random.shuffle(all_files)
    logger.info("All files are shuffeld.")
    
    # Calculate split index
    split_index = int(len(all_files) * validation_ratio)
    
    # Split files into validation set
    validation_files = all_files[split_index:]
    
    logger.info("Split is realised.")
    # Move files to respective directories
    for file in validation_files:
        output_val_file_path = os.path.join(validation_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(validation_folder, file))
        print(f"Moved {file} to {output_val_file_path}")
        
    print(f"Moved {len(validation_files)} files to {validation_folder}")

# Define paths
SPECTOGRAM_TEST_PATH  = SPECTOGRAM_PATH + "Test/"
SPECTOGRAM_VALIDATION_PATH = SPECTOGRAM_PATH + "Validation/"

# Splitting the spectograms into validation and test subset form the percentage split that is placed in test folder.
split_spectograms_validation(SPECTOGRAM_TEST_PATH, SPECTOGRAM_VALIDATION_PATH, validation_ratio=0.5, seed = 49)


def split_audio_files(source_folder, train_folder, test_folder, train_ratio=0.7, seed = 49):
    
    random.seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
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
        output_train_file_path = os.path.join(train_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))
        print(f"Moved {file} to {output_train_file_path}")
    
    
    for file in test_files:
        output_test_file_path = os.path.join(test_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))
        print(f"Moved {file} to {output_test_file_path }")
    
    print(f"Moved {len(train_files)} files to {train_folder}")
    print(f"Moved {len(test_files)} files to {test_folder}")

# Define paths
DATA_TRAIN_PATH = DATA_PATH + "Train/"
DATA_TEST_PATH  = DATA_PATH + "Test/"

# Split the spectrograms into two subsets into two subsets of triaing and test.
split_audio_files(MIC_PATH, DATA_TRAIN_PATH , DATA_TEST_PATH , train_ratio=0.70, seed = 49)


# Defining function for splitting the test subset into two subsets of validation and test.
def split_audio_files_validation(source_folder, validation_folder, validation_ratio=0.5, seed = 49):
    
    """
    ................................................................
    Parameters:
    - source_folder         = directory -> where the files are located
    - validation_folder     = directory -> where the validations file need to be located to
    - validation_ratio      = float     -> The ratio's of split wihtin the test folder after the training and test files are splitted
    ......................................................................
    
    """
    
    #Setting seed
    random.seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(validation_folder, exist_ok=True)

    # Get all files from the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle the files randomly
    random.shuffle(all_files)
    logger.info("All files are shuffeld.")
    
    # Calculate split index
    split_index = int(len(all_files) * validation_ratio)
    
    # Split files into validation set
    validation_files = all_files[split_index:]
    
    logger.info("Split is realised.")
    # Move files to respective directories
    for file in validation_files:
        output_val_file_path = os.path.join(validation_folder, file)
        shutil.move(os.path.join(source_folder, file), os.path.join(validation_folder, file))
        print(f"Moved {file} to {output_val_file_path}")
        
    print(f"Moved {len(validation_files)} files to {validation_folder}")

# Define paths
DATA_TEST_PATH  = DATA_PATH + "Test/"
DATA_VALIDATION_PATH = DATA_PATH + "Validation/"

# Splitting the spectograms into validation and test subset form the percentage split that is placed in test folder.
split_audio_files_validation(DATA_TEST_PATH, DATA_VALIDATION_PATH, validation_ratio=0.5, seed = 49) 