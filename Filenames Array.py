#Import necessary dependencies
import os
import numpy as np
import logging

###.............................................
# Set up logging
logging.basicConfig(filename='CAE-Baseline.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###.............................................
# Setting Paths
ROOT_FOLDER = r"D:/Menno/Thesis/Expirement/"
DATA_FOLDER = r"Data/"
TRAIN_FOLDER = r"Train/"
SPECTOGRAM_FOLDER  = r'Spectograms/'

###.............................................
# Defining necessary paths and define global directory paths
ROOT_PATH           = str(ROOT_FOLDER)
DATA_PATH           = ROOT_PATH + str(DATA_FOLDER)
SPECTOGRAM_PATH     = ROOT_PATH + str(SPECTOGRAM_FOLDER)
TRAIN_PATH          = SPECTOGRAM_PATH + str(TRAIN_FOLDER)
    
logger.info("Root paths are set.")

def get_filenames(train_path, save_path):
    # Haal alle bestandsnamen op in de directory
    filenames = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f)) and 'North' in f]
    
    # Sorteer de bestandsnamen om consistentie te garanderen
    filenames.sort()
    
    # Print de eerste vijf bestandsnamen
    print("First five filenames:", filenames[:5])
    
    # Sla de bestandsnamen op naar het gespecificeerde pad
    np.savetxt(save_path, filenames, delimiter='\n', fmt='%s')


# Voorbeeldpaden (pas deze aan naar jouw situatie)
TRAIN_PATH = "D:\\Menno\\Thesis\\Expirement\\Spectograms\\Test"
SAVE_PATH = "D:\\Menno\\Thesis\\Expirement\\1. Python files\\filenames_test_north.txt"

# Roep de functie aan met de paden
get_filenames(TRAIN_PATH, SAVE_PATH)