import os
import numpy as np
import tensorflow as tf
import logging
from keras.models import load_model, Model

###............     Setting up Logging       ............ 

logging.basicConfig(filename='Test_evaluation_Variation 2.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


###............     GPU Testing for usages       ............ 

# Optional: Test GPU usage
def test_gpu_usage():
    try:
        a = tf.constant(1.0)
        b = tf.constant(2.0)
        c = a + b

        with tf.device('/GPU:0'):  
            for _ in range(10000):
                result = c.numpy()

        logger.info("GPU test completed. Result of c: {}".format(result))
    except Exception as e:
        logger.error("Error during GPU test: {}".format(e))

test_gpu_usage()


###............     Configuration screen       ............ 

# Defining directory paths
ROOT_FOLDER = r"/home/u993985/Thesis/"
DATA_FOLDER = r"Data/200,300/"
MODEL_FOLDER = r"Models/" 
WEIGHTS_FOLDER = r"Weights/"
RESULTS_FOLDER = r"Results/"


# Test Parameter
batch_size = 64

# Filenames
spectogram_h, spectogram_w, spectogram_d = 200, 300, 1
modelname = str(f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}")
version = "Version 2"

###............     Setting Directory Paths        ............ 

ROOT_PATH = str(ROOT_FOLDER)
DATA_PATH = ROOT_PATH + str(DATA_FOLDER)
MODEL_PATH = ROOT_PATH + str(MODEL_FOLDER)
WEIGHTS_PATH = ROOT_PATH + str(WEIGHTS_FOLDER)
RESULTS_PATH = ROOT_PATH + str(RESULTS_FOLDER)

###............     Loading and normalizing dataset        ............ 
# Load test data
xy_test = np.load(os.path.join(DATA_PATH,'test_normalized.npy'))

###............     Loading model and generating predictions       ............ 
# Load the trained model
weights_filename = str(f"{modelname}-{version}" + ".h5.keras")
model_path = os.path.join(WEIGHTS_PATH, weights_filename)

model = load_model(model_path)
logging.info(f"Model loaded from {model_path}")

# Generate predictions (latent space representations) using the encoder model
full_pred = model.predict(xy_test, batch_size=batch_size)
logging.info("Predictions made on test data using the encoder model.")

# Save or print the results
results_path = os.path.join(RESULTS_PATH, f"predictions-{version}.npy")
logging.info(f'Predictions shape is: {full_pred.shape}')

np.save(results_path, full_pred)
logging.info(f"Predictions saved to {results_path}")


###............     Calculate MSE       ............ 
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_total_mse(dataset, full_pred):
    total_mse = []
    for img1, img2 in zip(dataset, full_pred):
        mse_value = mse(img1, img2)
        total_mse.append(mse_value)
    return np.mean(total_mse)

mean_mse = calculate_total_mse(xy_test, full_pred)
logger.info(f"Mean MSE: {mean_mse}")

