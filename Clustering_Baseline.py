import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import logging
from keras.models import load_model, Model
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids  
import random

###............     Setting up Logging       ............ 

logging.basicConfig(filename='Clustering_evaluation_Baseline.log', level=logging.INFO,
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
ROOT_FOLDER = r"/home/u993985/Thesis/"
DATA_FOLDER = r"Data/"
MODEL_FOLDER = r"Models/" 
WEIGHTS_FOLDER = r"Weights/"
RESULTS_FOLDER = r"Results/"
ANNOTATIONS_FOLDER = r"Annotations/"

# K-Medoids Parameter
num_clusters = 8
max_iterations = 50

# Filenames
version = "Baseline"
cluster_filename = f'Cluster_Predicitions-{version}-K{num_clusters}.txt'
sorted_files = 'filenames_test.txt'

###............     Setting Directory Paths        ............ 
   
ROOT_PATH = str(ROOT_FOLDER)
DATA_PATH = ROOT_PATH + str(DATA_FOLDER)
MODEL_PATH = ROOT_PATH + str(MODEL_FOLDER)
WEIGHTS_PATH = ROOT_PATH + str(WEIGHTS_FOLDER)
RESULTS_PATH = ROOT_PATH + str(RESULTS_FOLDER)
ANNOTATIONS_PATH = ROOT_PATH + str(ANNOTATIONS_FOLDER)

logging.info("Root paths are set.")

###............     Loading and normalizing dataset        ............ 

xy_test = np.load(os.path.join(DATA_PATH,'test.npy'))

def normalize(dataset):
    """
    Normalize the dataset of spectograms.

    Parameters:
    dataset (numpy array): The numpy array that contains the dataset of spectograms.

    Returns:
    numpy array: Normalized dataset.
    """ 
    
    dataset = dataset.astype(np.float32)
    dataset_normalized = dataset / 255.0
    logging.info("Dataset normalized successfully.")
    return dataset_normalized

xy_test = normalize(xy_test)


###............     Reading corresponding filenames    ............

filenames_path = os.path.join(DATA_PATH, sorted_files)
with open(filenames_path, 'r') as file:
    filenames = file.read().splitlines()

###............     Loading predictions       ............ 

results_path = os.path.join(RESULTS_PATH, f"predictions-{version}.npy")
predictions_latent_space = np.load(results_path)
logging.info(f'Predictions shape is: {predictions_latent_space.shape}')

###............     K-Medoids Clustering        ............ 

def apply_kmedoids_clustering(predictions_latent_space, max_iterations, num_clusters):
    """
    Apply K-Medoids clustering to the predicted latent space.

    Parameters:
    predictions_latent_space (numpy array): Numpy array containing the predicted latent spaces.
    max_iterations (int): The number of iterations the K-Medoids should perform.
    num_clusters (int): The number of clusters the K-Medoids should apply.

    Returns:
    tuple: Cluster labels and medoid indices.
    """ 
    
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    flat_images = flat_images.astype('float32') / 255.0

    kmedoids_instance = KMedoids(n_clusters=num_clusters, init='k-medoids++', max_iter= max_iterations, random_state=42)
    kmedoids_instance.fit(flat_images)

    clusters = kmedoids_instance.labels_
    medoids = kmedoids_instance.medoid_indices_

    logging.info("KMedoids clustering completed.")
    return clusters, medoids

clusters, medoids = apply_kmedoids_clustering(predictions_latent_space, max_iterations, num_clusters)

def plot_clustered_images(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path='results/'):
    """
    Plot the clustered images using K-Medoids clustering results.

    Parameters:
    predictions_latent_space (numpy array): Numpy array containing the predicted latent spaces.
    clusters (numpy array): Array containing the cluster assignments for each data point.
    medoids (numpy array): Array containing the indices of the medoids.
    cluster_filename (str): Filename to save the plot.
    num_clusters (int): The number of clusters used in K-Medoids.
    save_path (str): Directory path where the plot will be saved.

    Returns:
    None
    """

    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(flat_images)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters)) 
    
    for cluster_label in np.unique(clusters):
        cluster_points = reduced_features[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label+1}', color=colors[cluster_label], s=50)

    plt.scatter(reduced_features[medoids, 0], reduced_features[medoids, 1], s=150, c='black', marker='X', label='Medoids')

    plt.title('Clustering Results (KMedoids)', fontsize=20, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=18)
    plt.ylabel('Principal Component 2', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
  
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = cluster_filename + "-plot.png"
        filepath = os.path.join(save_path, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logging.info(f"The clustering plot is saved at: {filepath}")
    else:
        plt.show()
        logging.info("The clustering plot is displayed.")

plot_clustered_images(predictions_latent_space, clusters, medoids, cluster_filename , num_clusters, save_path=RESULTS_PATH)

del predictions_latent_space

cluster_assignments_path = RESULTS_PATH + cluster_filename
with open(cluster_assignments_path, 'w') as f:
    for filename, cluster in zip(filenames, clusters):
        f.write(f"{filename}: {cluster+1}\n")

logging.info(f"Cluster assignments saved to {cluster_assignments_path}")


###............     Clustering Annotations        ............ 

def read_cluster_assignments(cluster_assignments_path):
    """
    Read cluster assignments from file.

    Parameters:
    cluster_assignments_path (str): Path to the file containing cluster assignments.

    Returns:
    dict: Dictionary mapping cluster numbers to lists of filenames.
    """ 
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

def select_filenames_for_annotation(cluster_to_filenames, percentage_to_select):
    """
    Select filenames for annotation based on a given percentage.

    Parameters:
    cluster_to_filenames (dict): Dictionary mapping cluster numbers to lists of filenames.
    percentage_to_select (float): Percentage of files to select for annotation from each cluster.

    Returns:
    dict: Dictionary mapping cluster numbers to lists of selected filenames.
    """ 
    selected_filenames = {}
    for cluster, filenames in cluster_to_filenames.items():
        num_files_to_select = max(1, int(len(filenames) * percentage_to_select))
        selected_files = random.sample(filenames, num_files_to_select)
        selected_filenames[cluster] = selected_files
    logging.info("Selected filenames for manual annotation.")
    return selected_filenames

def save_selected_filenames(selected_filenames, ANNOTATIONS_PATH):
    """
    Save selected filenames for annotation to text files.

    Parameters:
    selected_filenames (dict): Dictionary mapping cluster numbers to lists of selected filenames.
    ANNOTATIONS_PATH (str): Path to save the annotation files.

    Returns:
    None
    """ 
    
    
    if not os.path.exists(ANNOTATIONS_PATH):
        os.makedirs(ANNOTATIONS_PATH)

    for cluster, filenames in selected_filenames.items():
        annotation_file = os.path.join(ANNOTATIONS_PATH, f"annotations_cluster{cluster}.txt")
        annotation = "undetermined" 
        with open(annotation_file, 'w') as f:
            for filename in filenames:
                f.write(f"{filename}: {annotation}\n")
        logging.info(f"Cluster {cluster}: Selected {len(filenames)} filenames for manual annotation")
        logging.info(f"Annotations saved to {annotation_file}")

cluster_to_filenames = read_cluster_assignments(cluster_assignments_path)
selected_filenames = select_filenames_for_annotation(cluster_to_filenames, percentage_to_select=0.01)
save_selected_filenames(selected_filenames, ANNOTATIONS_PATH)
