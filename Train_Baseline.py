import numpy as np
import os
import tensorflow as tf
import logging
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

###............     Setting up Logging       ............ 

logging.basicConfig(filename='Train-Baseline.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


###............     GPU Testing for usages       ............ 

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
DATA_FOLDER     = r"Data/"
MODEL_FOLDER    = r"Models/" 
WEIGHTS_FOLDER  = r"Weights/"
RESULTS_FOLDER  = r"Results/"

# Spectogram specifics and spectogram seperation specifics
spectogram_h, spectogram_w, spectogram_d = 120, 160, 1

seperation = False
block_height, block_width, block_depth = 120, 160, 1

# Version description 
modelname = str(f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}")
version = "Baseline"

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

ROOT_PATH = str(ROOT_FOLDER)
DATA_PATH = ROOT_PATH + str(DATA_FOLDER)
MODEL_PATH = ROOT_PATH + str(MODEL_FOLDER)
WEIGHTS_PATH = ROOT_PATH + str(WEIGHTS_FOLDER)
RESULTS_PATH = ROOT_PATH + str(RESULTS_FOLDER)

###............     Loading and normalizing data     ............ 

def normalizing(block_dataset):
    """
    Normalize the dataset by converting it to float32 and scaling it to [0, 1].

    Parameters:
    block_dataset (numpy array): The dataset to normalize.

    Returns:
    numpy array: The normalized dataset.
    """
    
    dataset = block_dataset.astype(np.float32) 
    dataset_normalized = np.array(dataset) / 255.0
    logger.info("Dataset normalized successfully.") 
    return dataset_normalized

# Loading necessary numpy arrays
xy_train = np.load(DATA_FOLDER + "train.npy")
xy_val = np.load(DATA_FOLDER + "val.npy")

# Normalizing the datasets
xy_train = normalizing(xy_train)
xy_val = normalizing(xy_val)

###............       Block seperation            ............ 

def block_seperation():
    """
    Define the dimensions of the input based on the separation flag.

    Returns:
    tuple: The input height, width, and depth.
    """
    
    if seperation:
        return block_height, block_width, block_depth
    else:
        return spectogram_h, spectogram_w, spectogram_d

input_height, input_width, input_depth = block_seperation()


###............     Models Architecture CAE       ............ 

def models_architecture(model_path, modelname, version, input_height, input_width, input_depth):
    """
    Define and compile the Convolutional Autoencoder (CAE) model, then save its architecture as an image.

    Parameters:
    model_path (str): The path to save the model architecture image.
    modelname (str): The model name.
    version (str): The version description.
    input_height (int): The input height of the model.
    input_width (int): The input width of the model.
    input_depth (int): The input depth of the model.

    Returns:
    tuple: The compiled model, the encoded layer, and the decoded layer.
    """
    
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


###............     Training Convolutional Autoencoder        ............ 
try:
    logger.info(f"Starting training with batch_size = {batch_size}, epochs = {epochs}, min_delta = {min_delta}, patience = {patience}")

    history = model.fit(
        xy_train, xy_train,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 1,
        validation_data = (xy_val, xy_val),
        callbacks = [model_checkpoint_callback, early_stopping_callback]
    )

    logger.info(f"Training finished with batch_size = {batch_size}, epochs = {epochs}, min_delta = {min_delta}, patience = {patience}")
    
    logger.info(f"Training Loss: {history.history['loss']}")
    logger.info(f"Validation Loss: {history.history['val_loss']}")

    model.save(filepath)
    logger.info(f"Model saved at {filepath}")
    logger.info("Model training complete.")  
except Exception as e:
    logger.error(f"Error during training: {e}")
