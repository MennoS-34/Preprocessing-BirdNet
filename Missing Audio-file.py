import os
import librosa
import numpy as np
import logging
from scipy.io.wavfile import write

# Setting up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define global paths and variables
ROOT_FOLDER = r"D:/Menno/Thesis/Expirement/"
AUDIO_FOLDER = r"Audio/mic_north/"
DATA_FOLDER_1 = r"Data/Train/"

SPECTOGRAM_FOLDER = r'Spectograms/'

global_segment_id = 1

def set_root_paths(root_folder, data_folders, audio_folder):
    
    global ROOT_PATH, DATA_PATHS, AUDIO_PATH
    ROOT_PATH = str(root_folder)
    DATA_PATHS = [os.path.join(ROOT_PATH, str(data_folder)) for data_folder in data_folders]
    AUDIO_PATH = os.path.join(ROOT_PATH, str(AUDIO_FOLDER))

    logger.info("Root paths are set.")
    
set_root_paths(ROOT_FOLDER, [DATA_FOLDER_1], AUDIO_FOLDER)

def split_audio_files(audio_path, data_paths, output_data_path, segment_duration=3):
    global global_segment_id
    
    # Ensure output directory exists
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    
    # Process each audio file in the directory
    for filename in os.listdir(audio_path):
        if filename.endswith('.wav'):
            # Load the WAV audio file
            file_path = os.path.join(audio_path, filename)
            y, sr = librosa.load(file_path, sr=None)  # Use the native sampling rate

            # Calculate the number of samples per segment
            split_length = segment_duration * sr

            # Split the audio into segments
            num_splits = len(y) // split_length
            for i in range(num_splits):
                # Define start and end samples for each split
                start_sample = i * split_length
                end_sample = (i + 1) * split_length
                
                # Extract the segment
                split_segment = y[start_sample:end_sample]
                
                # Calculate start and end time in seconds for file naming
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                # Format file name with first 11 characters of the original file and time interval
                name_prefix = filename[:11]
                output_file = f"ID {global_segment_id} - {name_prefix} - Sec {start_time}-{end_time} - North.wav"
                
                # Check if the specific split file already exists in any of the data paths
                file_already_exists = False
                for data_path in data_paths:
                    if os.path.exists(os.path.join(data_path, output_file)):
                        file_already_exists = True
                        break
                
                if file_already_exists:
                    logger.info(f"Segment {output_file} already exists. Skipping...")
                else:
                    # Save to the specified output data path
                    output_file_path = os.path.join(output_data_path, output_file)
                    
                    # Export the split segment as a new WAV file
                    write(output_file_path, sr, split_segment)
                
                # Increment the global segment ID
                global_segment_id += 1
    
    logger.info("Audio 3 second chunks are created")

# Example usage:
OUTPUT_DATA_PATH = r"D:/Menno/Thesis/Expirement/Data/mic_north"

split_audio_files(AUDIO_PATH, DATA_PATHS, OUTPUT_DATA_PATH, segment_duration=3)
