import numpy as np
import os
import tensorflow as tf
import logging
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D, Cropping2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model, Sequence 
import psutil
import math
import random
import matplotlib.pyplot as plt



###............     Setting up Logging       ............ 

logging.basicConfig(filename='Train-Version 3.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info() 
    logger.info(f"Memory usage: RSS = {mem_info.rss / (1024 * 1024)} MB, VMS = {mem_info.vms / (1024 * 1024)} MB")


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
ROOT_FOLDER     = r"/home/u993985/Thesis/"
DATA_FOLDER     = r"Data/V2/"
MODEL_FOLDER    = r"Models/" 
WEIGHTS_FOLDER  = r"Weights/"
RESULTS_FOLDER  = r"Results/"

# Spectogram specifics and spectogram seperation specifics
spectogram_h, spectogram_w, spectogram_d = 200, 300, 1

seperation = True
block_height, block_width, block_depth = 100, 150, 1

# Version description 
modelname = str(f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}")
version = "Version 3"

###............     Defining Hyperparamters      ............ 

# Setting Model Checkpoint parameters              
monitor = 'val_loss'                       # Metric to monitor
verbose = 0                                # Verbosity mode
save_best_only = True                      # Only save the model if the monitored quantity improves
save_weights_only = False                  # If True, only the model's weights will be saved (not the entire model)
save_freq = 'epoch'                        # Frequency to save the model, 'epoch' means after each epoch

# Setting EarlyStopping parameters
min_delta = 0.0001                         # Minimum change to qualify as an improvement
patience = 20                              # Number of epochs with no improvement after which training will be stopped
mode = 'auto'                              # Whether to minimize or maximize the monitored quantity
baseline = None                            # Baseline value for the monitored quantity
restore_best_weights = True                # Whether to restore model weights from the epoch with the best value of the monitored quantity

# Training Parameters
batch_size = 128
epochs = 50

###............     Setting Directory Paths        ............ 

def set_root_paths(ROOT_FOLDER, DATA_FOLDER,  MODEL_FOLDER, WEIGHTS_FOLDER, RESULTS_FOLDER):
    ROOT_PATH           = str(ROOT_FOLDER)
    DATA_PATH           = ROOT_PATH + str(DATA_FOLDER)
    MODEL_PATH          = ROOT_PATH + str(MODEL_FOLDER)
    WEIGHTS_PATH        = ROOT_PATH + str(WEIGHTS_FOLDER)
    RESULTS_PATH        = ROOT_PATH + str(RESULTS_FOLDER)
    logger.info("Root paths are set.")
    return ROOT_PATH, DATA_PATH, MODEL_PATH, WEIGHTS_PATH, RESULTS_PATH

ROOT_PATH, DATA_PATH, MODEL_PATH, WEIGHTS_PATH, RESULTS_PATH = set_root_paths(ROOT_FOLDER, DATA_FOLDER, MODEL_FOLDER, WEIGHTS_FOLDER, RESULTS_FOLDER)

###............     Loading and normalizing data     ............ 

# Loading necessary numpy arrays
# Memory-mapped loading of large arrays
xy_val_normalized = np.load(DATA_PATH + "val_normalized.npy")
log_memory_usage()
logger.info("Validation set is loaded in.")

xy_train_normalized = np.load(DATA_PATH + "train_normalized.npy")
log_memory_usage()
logger.info("Training set is loaded in.")

###............     Spectogram seperation and Generator    ............ 

def spectogram_seperation():
    if seperation:
        return block_height, block_width, block_depth
    else:
        return spectogram_h, spectogram_w, spectogram_d

input_height, input_width, input_depth = spectogram_seperation()

def seperating_spectograms(dataset, block_height, block_width):
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

class DataGenerator(Sequence):
    def __init__(self, data, batch_size=128, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]

        if batch_data.shape[1:] != (input_height, input_width, input_depth):
            raise ValueError(f"Batch data dimensions are incorrect: {batch_data.shape}")

        return batch_data, batch_data  # Since we are doing autoencoder, x and y are same

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

if seperation:
    block_train = seperating_spectograms(xy_train_normalized, block_height, block_width)
    logger.info("The training dataset is transformed into block size dataset.")
    block_val = seperating_spectograms(xy_val_normalized, block_height, block_width)
    logger.info("The validation dataset is transformed into block size dataset.")
    del xy_train_normalized
    del xy_val_normalized

    train_generator = DataGenerator(block_train, batch_size=batch_size)
    val_generator = DataGenerator(block_val, batch_size=batch_size, shuffle=False)
else:
    train_generator = DataGenerator(xy_train_normalized, batch_size=batch_size)
    val_generator = DataGenerator(xy_val_normalized, batch_size=batch_size, shuffle=False)

###............     Models Architecture CAE       ............ 

def models_architecture(model_path, modelname, version, input_height, input_width, input_depth):
    input_img = Input(shape=(input_height, input_width, input_depth))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name = 'encoded')(x)
    logger.info("Encoded is defined.")
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)    
    x = Cropping2D(cropping=((2, 2), (1, 1)))(x) 
    decoded = Conv2D(block_depth, (3, 3), activation='sigmoid', padding='same', name = 'decoded')(x)
    logger.info("Decoded is defined.") 
    model = Model(input_img, decoded)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    logger.info("Model is compiled.") 
    
    architecture_file = os.path.join(model_path, f'model_architecture-{modelname}-{version}.png')
    plot_model(model, to_file=architecture_file, show_shapes=True, show_layer_names=True)
    logger.info(f"Model architecture saved as {architecture_file}")
    model.summary(print_fn=lambda x: logger.info(x))
    
    return model, encoded, decoded

model, encoded, decoded= models_architecture(MODEL_PATH, modelname, version, input_height, input_width, input_depth)


###............     Defining Callbacks        ............ 

class CustomModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if logs is not None:
            logger.info(f"Epoch {epoch}: val_loss improved to {logs.get('val_loss'):.4f}, saving model to {self.filepath}")

class CustomEarlyStopping(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if logs is not None:
            logger.info(f"Epoch {epoch}: {logs}")

weights_filename = str(f"{modelname}-{version}" + ".h5.keras")
filepath = os.path.join(WEIGHTS_PATH, weights_filename)

model_checkpoint_callback = CustomModelCheckpoint(
    filepath = filepath,
    monitor = monitor,
    verbose = 1,
    save_best_only = save_best_only,
    mode = mode,
    save_weights_only = save_weights_only,
    save_freq = save_freq,
)

early_stopping_callback = CustomEarlyStopping(
    monitor = monitor,
    min_delta = min_delta,
    patience = patience,
    verbose = verbose,
    mode = mode,
    baseline = baseline,
    restore_best_weights = restore_best_weights 
)

log_memory_usage()

###............     Training Convolutional Autoencoder        ............ 

try:
    logger.info(f"Starting training with batch_size = {batch_size}, epochs = {epochs}, min_delta = {min_delta}, patience = {patience}")
    log_memory_usage()
    history = model.fit(
        train_generator,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 1,
        validation_data = val_generator,
        callbacks = [model_checkpoint_callback, early_stopping_callback]
    )
    log_memory_usage()
    
    
    logger.info(f"Training finished with batch_size = {batch_size}, epochs = {epochs}, min_delta = {min_delta}, patience = {patience}")
    
    logger.info(f"Training Loss: {history.history['loss']}")
    logger.info(f"Validation Loss: {history.history['val_loss']}")

    model.save(filepath)
    logger.info(f"Model saved at {filepath}")
    logger.info("Model training complete.")  
except Exception as e:
    logger.error(f"Error during training: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error args: {e.args}")


###............     Prediction and Reconstruction   ............ 


def prediciting_full(block_dataset_normalized, extraction_model, spectogram_h, spectogram_w, block_height, block_width, block_depth):
    block_predicting = extraction_model.predict(block_dataset_normalized)
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

    full_pred = np.array(full_pred)
    full_pred = full_pred * 255.0
    logger.info("Full predictions are made.")
    return full_pred

full_pred = prediciting_full(block_val, model, spectogram_h, spectogram_w, block_height, block_width, block_depth)

###............     Plotting Predictions    ............ 
del block_val
del xy_val_normalized

xy_val = np.load(DATA_PATH + "val.npy")
log_memory_usage()
logger.info("Validation set is loaded in again but not normalized.")


def plots_with_titles_full(dataset, full_pred, number_of_plots=10):
    random_indices = random.sample(range(len(dataset)), number_of_plots)
    for idx in random_indices:
        img1 = dataset[idx]
        img2 = full_pred[idx]

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img1.squeeze(), cmap='gray')
        ax[0].axis("off")
        ax[0].set_title("Original")

        ax[1].imshow(img2.squeeze().astype(np.uint8), cmap='gray')
        ax[1].axis("off")
        ax[1].set_title("Predicted")

        original_filename = f"image_{idx}"
        plt.savefig(os.path.join(RESULTS_FOLDER, f"result_{original_filename}.png"))
        plt.close()
        logger.info("Full spectrogram plots are saved.")

# Plot original and predicted full spectrograms


plots_with_titles_full(xy_val, full_pred, number_of_plots=10)


###............     Evaluation    ............ 

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_total_mse(dataset, full_pred):
    total_mse = []
    for img1, img2 in zip(dataset, full_pred):
        mse_value = mse(img1, img2)
        total_mse.append(mse_value)
    return total_mse, np.mean(total_mse)

total_mse_values, mean_mse = calculate_total_mse(xy_val, full_pred)
logger.info(f"Total MSE: {total_mse_values}")
logger.info(f"Mean MSE: {mean_mse}")