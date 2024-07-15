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



###............     Setting up Logging       ............ 

logging.basicConfig(filename='Train-Version 2.log', level=logging.INFO,
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

seperation = False
block_height, block_width, block_depth = 100, 150, 1

# Version description 
modelname = str(f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}")
version = "Version 2"

###............     Defining Hyperparamters      ............ 

# Setting Model Checkpoint parameters              
monitor = 'val_loss'                       # Metric to monitor
verbose = 1                                # Verbosity mode
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
xy_val = np.load(DATA_PATH + "val_normalized.npy")
log_memory_usage()
logger.info("Validation set is loaded in.")

xy_train = np.load(DATA_PATH + "train_normalized.npy")
log_memory_usage()
logger.info("Training set is loaded in.")

###............       Block seperation            ............ 

def block_seperation():
    if seperation:
        return block_height, block_width, block_depth
    else:
        return spectogram_h, spectogram_w, spectogram_d

input_height, input_width, input_depth = block_seperation()

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
    x = Cropping2D(cropping=((0, 0), (0, 1)))(x) 
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)    
    
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

class DataGenerator(Sequence):
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]

        # Check dimensions of the batch
        if batch_data.shape[1:] != (200, 300, 1):
            raise ValueError(f"Batch data dimensions are incorrect: {batch_data.shape}")

        return batch_data, batch_data  # Since we are doing autoencoder, x and y are same

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

            
logging.info(f"xy_train shape: {xy_train.shape}")
logging.info(f"xy_val shape: {xy_val.shape}")
            
train_generator = DataGenerator(xy_train, batch_size=batch_size)
val_generator = DataGenerator(xy_val, batch_size=batch_size, shuffle=False)

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
    del xy_train
    del xy_val
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
