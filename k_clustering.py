import os
import numpy as np
import gc  # Garbage collection
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from keras.models import load_model, Model
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids  
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

###............     Setting up Logging       ............ 

logging.basicConfig(filename='K-Hyperparameter-Gridsearch.log', level=logging.INFO,
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

###............     Configuration       ............ 

ROOT_FOLDER = r"/home/u993985/Thesis/"
DATA_FOLDER = r"Data/"
MODEL_FOLDER = r"Models/" 
WEIGHTS_FOLDER = r"Weights/"
RESULTS_FOLDER = r"Results/"
ANNOTATIONS_FOLDER = r"Annotations/"

batch_size = 128
max_iterations = 50

spectogram_h, spectogram_w, spectogram_d = 120, 160, 1
modelname = f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}"
version = "Baseline"

cluster_filename = f'Cluster_Predictions-Hyperparameter-Gridsearch-K'
sorted_files = 'filenames_test.txt'

def set_root_paths(ROOT_FOLDER, DATA_FOLDER, MODEL_FOLDER, WEIGHTS_FOLDER, RESULTS_FOLDER, ANNOTATIONS_FOLDER):
    ROOT_PATH = str(ROOT_FOLDER)
    DATA_PATH = ROOT_PATH + str(DATA_FOLDER)
    MODEL_PATH = ROOT_PATH + str(MODEL_FOLDER)
    WEIGHTS_PATH = ROOT_PATH + str(WEIGHTS_FOLDER)
    RESULTS_PATH = ROOT_PATH + str(RESULTS_FOLDER)
    ANNOTATIONS_PATH = ROOT_PATH + str(ANNOTATIONS_FOLDER)
    
    logging.info("Root paths are set.")
    return ROOT_PATH, DATA_PATH, MODEL_PATH, WEIGHTS_PATH, RESULTS_PATH, ANNOTATIONS_PATH

ROOT_PATH, DATA_PATH, MODEL_PATH, WEIGHTS_PATH, RESULTS_PATH, ANNOTATIONS_PATH = set_root_paths(ROOT_FOLDER, DATA_FOLDER, MODEL_FOLDER, WEIGHTS_FOLDER, RESULTS_FOLDER, ANNOTATIONS_FOLDER)

###............     Loading and Normalizing Dataset        ............ 

xy_test = np.load(os.path.join(DATA_PATH, 'test.npy'))
filenames_path = os.path.join(DATA_PATH, sorted_files)
with open(filenames_path, 'r') as file:
    filenames = file.read().splitlines()

def normalize(dataset):
    dataset = dataset.astype(np.float16)
    dataset_normalized = dataset / 255.0
    logging.info("Dataset normalized successfully.")
    return dataset_normalized

xy_test = normalize(xy_test)

###............     Loading Model and Generating Predictions       ............ 

weights_filename = f"{modelname}-{version}.h5.keras"
model_path = os.path.join(WEIGHTS_PATH, weights_filename)

model = load_model(model_path)
logging.info(f"Model loaded from {model_path}")

encoder_model = Model(inputs=model.input, outputs=model.get_layer('encoded').output)
logging.info("Encoder model extracted from full model.")

predictions_latent_space = encoder_model.predict(xy_test, batch_size=batch_size)
logging.info("Predictions made on test data using the encoder model.")
del xy_test  # Free up memory by deleting the test data array
gc.collect()  # Explicitly run garbage collection

results_path = os.path.join(RESULTS_PATH, "predictions.npy")
np.save(results_path, predictions_latent_space)
logging.info(f"Predictions saved to {results_path}")

###............     Annotating 1% of the Test Data        ............ 

subset_annotations_file = os.path.join(ANNOTATIONS_FOLDER, 'subset_annotations.txt')

selected_filenames = []
true_labels_dict = {}

with open(subset_annotations_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if ': ' in line:
            try:
                filename, annotation = line.split(': ', 1)
                selected_filenames.append(filename)
                true_labels_dict[filename] = annotation
            except ValueError as e:
                logging.error("Error processing line '{}': {}".format(line, e))
        else:
            logging.warning("Line skipped (incorrect format): {}".format(line))

logging.info("Subset annotations loaded from {}".format(subset_annotations_file))

###............     K Medoids        ............ 

def apply_kmedoids_clustering(predictions_latent_space, max_iterations, num_clusters):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    flat_images = flat_images.astype('float16') / 255.0

    kmedoids_instance = KMedoids(n_clusters=num_clusters, init='k-medoids++', max_iter=max_iterations, random_state=49)
    kmedoids_instance.fit(flat_images)

    clusters = kmedoids_instance.labels_
    medoids = kmedoids_instance.medoid_indices_

    logging.info("KMedoids clustering completed.")
    return clusters, medoids, kmedoids_instance.inertia_

def save_cluster_assignments(filenames, clusters, cluster_filename, RESULTS_PATH):
    cluster_assignments_path = os.path.join(RESULTS_PATH, f"{cluster_filename}.txt")
    with open(cluster_assignments_path, 'w') as f:
        for filename, cluster in zip(filenames, clusters):
            f.write(f"{filename}: {cluster}\n")
    logging.info(f"Cluster assignments saved to {cluster_assignments_path}")

def read_cluster_assignments(cluster_assignments_path):
    cluster_to_filenames = {}
    with open(cluster_assignments_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                filename, cluster = line.split(': ')
                cluster = int(cluster)
                if cluster not in cluster_to_filenames:
                    cluster_to_filenames[cluster] = []
                cluster_to_filenames[cluster].append(filename)
    logging.info("Cluster assignments read successfully.")
    return cluster_to_filenames

def retrieve_annotated_clusters(selected_filenames, cluster_assignments_path):
    cluster_to_filenames = read_cluster_assignments(cluster_assignments_path)
    annotated_clusters = {filename: cluster for cluster, files in cluster_to_filenames.items() for filename in files if filename in selected_filenames}
    return annotated_clusters

###............     Evaluation Metrics Calculation        ............ 

def calculate_metrics(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi

from sklearn.preprocessing import LabelEncoder

annotation_mapping = {"Bird": 0, "Noise": 1}  # Add more mappings as needed

def map_annotations_to_labels(annotations, annotation_mapping):
    return [annotation_mapping[annotation.strip()] for annotation in annotations]

K_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
metrics_results = {}
wss_values = []

for K in K_values:
    clusters, medoids, wss = apply_kmedoids_clustering(predictions_latent_space, max_iterations, K)
    save_cluster_assignments(filenames, clusters, f"{cluster_filename}-K{K}", RESULTS_PATH)
    cluster_to_filenames = read_cluster_assignments(os.path.join(RESULTS_PATH, f"{cluster_filename}-K{K}.txt"))
    logging.info("The functions are executed")
    
    wss_values.append(wss)
    logger.info(f"WSS for K={K}: {wss}")
    
    # Free up memory by deleting the kmedoids_instance and other large variables, then calling garbage collector
    del wss, clusters, medoids
    gc.collect()

    true_labels = []
    pred_labels = []
    for cluster, files in cluster_to_filenames.items():
        for file in files:
            if file in selected_filenames:
                true_labels.append(true_labels_dict[file])  # Get annotation (string)
                pred_labels.append(cluster)  # Cluster label

    true_labels_int = map_annotations_to_labels(true_labels, annotation_mapping)

    ari, nmi = calculate_metrics(true_labels_int, pred_labels)
    metrics_results[K] = {'ARI': ari, 'NMI': nmi}

    logging.info(f"Metrics for K={K}: ARI={ari}, NMI={nmi}")

    # Explicitly free up memory
    del true_labels, pred_labels, cluster_to_filenames, true_labels_int
    gc.collect()
    
logging.info(metrics_results)
logging.info(wss_values)
