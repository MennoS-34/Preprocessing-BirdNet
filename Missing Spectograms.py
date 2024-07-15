import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import logging

ROOT_FOLDER = r"D:/Menno/Thesis/Expirement/"
DATA_FOLDER = r"Data/"
MIC_FOLDER = r"mic_north/"
SPECTOGRAM_FOLDER  = r'Spectograms/'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_root_paths(ROOT_FOLDER, DATA_FOLDER, SPECTOGRAM_FOLDER, ):
    global ROOT_PATH, DATA_PATH , SPECTOGRAM_PATH, MIC_PATH
    ROOT_PATH           = str(ROOT_FOLDER)
    DATA_PATH           = ROOT_PATH + str(DATA_FOLDER)
    MIC_PATH            = DATA_PATH + str(MIC_FOLDER)
    SPECTOGRAM_PATH     = ROOT_PATH + str(SPECTOGRAM_FOLDER)

    logger.info("Root paths are set.")
    
set_root_paths(ROOT_FOLDER, DATA_FOLDER, SPECTOGRAM_FOLDER)



def create_mel_spec(MIC_PATH, SPECTOGRAM_PATH):  
    
    # Process each audio file in the directory
    for songname in os.listdir(MIC_PATH):
        if songname.endswith('.wav'):
            audio_path = os.path.join(MIC_PATH, songname)  # Proper path handling
            name_prefix = songname[:-4]
            output_name = f"{name_prefix}.png"
            output_file_path = os.path.join(SPECTOGRAM_PATH, output_name)
            
            # Check if the spectrogram already exists
            if not os.path.exists(output_file_path):
                sig, sr = librosa.load(audio_path)  # Load the audio file with librosa
                
                S = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=1024, fmin=800, fmax=(sr//2), n_mels=128, hop_length=512)   
                S_dB = librosa.power_to_db(S, ref=np.max)

                # Set up the figure for plotting with no axes
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
                plt.axis('off')
                plt.gca().set_position([0, 0, 1, 1])
                
                # Save the figure
                plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, format='png')
                plt.close()  # Close the figure to free up memory
                logging.info(f"Mel spectrogram saved: {output_file_path}")
            else:
                logging.info(f"Mel spectrogram already exists: {output_file_path}")

create_mel_spec(MIC_PATH, SPECTOGRAM_PATH)