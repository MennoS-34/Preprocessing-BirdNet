import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import random
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from keras.models import load_model

###............     Setting up Logging       ............ 

logging.basicConfig(filename='Clustering_evaluation_Version 1.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###............     Configuration       ............ 

# Defining directory paths
ROOT_FOLDER     = r"/home/u993985/Thesis 2/"
DATA_FOLDER     = ROOT_FOLDER + r"Data/200,300/"
WEIGHTS_FOLDER  = ROOT_FOLDER + r"Weights/"
RESULTS_FOLDER  = ROOT_FOLDER + r"Results/"
ANNOTATIONS_FOLDER = ROOT_FOLDER +  r"Annotations/"

# Spectrogram specifics
spectogram_h, spectogram_w, spectogram_d = 200, 300, 1
block_height, block_width, block_depth = 100, 150, 1

# Version description
modelname = str(f"Spectogram {spectogram_h}, {spectogram_w}, {spectogram_d}")
version = "Version 3"
cluster_filename = f'cluster_results - {version}'

# Parameters
max_iterations = 50
num_clusters = 8
annotations_path = os.path.join(ROOT_FOLDER, ANNOTATIONS_FOLDER)

# Apply KMedoids clustering
def apply_kmedoids_clustering(predictions_latent_space, max_iterations, num_clusters):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    flat_images = flat_images.astype('float16') / 255.0

    kmedoids_instance = KMedoids(n_clusters=num_clusters, init='k-medoids++', max_iter=max_iterations, random_state=42)
    kmedoids_instance.fit(flat_images)

    clusters = kmedoids_instance.labels_
    medoids = kmedoids_instance.medoid_indices_

    logging.info("KMedoids clustering completed.")
    return clusters, medoids

# Plot clustered images
def plot_clustered_images(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path='results/'):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(flat_images)
    
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

# Read cluster assignments from file
def read_cluster_assignments(cluster_assignments_path):
    cluster_to_filenames = {}
    with open(cluster_assignments_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:  # Ensure the line is not empty
                filename, cluster = line.split(': ')
                cluster = int(cluster)  # Convert cluster to integer
                if cluster not in cluster_to_filenames:
                    cluster_to_filenames[cluster] = []
                cluster_to_filenames[cluster].append(filename)
    logging.info("Cluster assignments read successfully.")
    return cluster_to_filenames

# Select the percentage of that need to be annotated
def select_filenames_for_annotation(cluster_to_filenames, percentage_to_select):
    selected_filenames = {}
    for cluster, filenames in cluster_to_filenames.items():
        num_files_to_select = max(1, int(len(filenames) * percentage_to_select))
        selected_files = random.sample(filenames, num_files_to_select)
        selected_filenames[cluster] = selected_files
    logging.info("Selected filenames for manual annotation.")
    return selected_filenames

# Create directories and save selected filenames
def save_selected_filenames(selected_filenames, ANNOTATIONS_PATH):
    if not os.path.exists(ANNOTATIONS_PATH):
        os.makedirs(ANNOTATIONS_PATH)

    for cluster, filenames in selected_filenames.items():
        annotation_file = os.path.join(ANNOTATIONS_PATH, f"annotations_cluster{cluster}.txt")

        # Simulate adding labels "bird", "nothing", or "noise" to each filename
        # Replace with your actual labeling logic
        annotation = "undetermined"  # Example label, replace as needed

        # Save annotation information to text file
        with open(annotation_file, 'w') as f:
            for filename in filenames:
                f.write(f"{filename}: {annotation}\n")

        logging.info(f"Cluster {cluster}: Selected {len(filenames)} filenames for manual annotation")
        logging.info(f"Annotations saved to {annotation_file}")

# Main execution
if __name__ == "__main__":
    # Load the predictions_latent_space
    predictions_latent_space = np.load('/home/u993985/Thesis 2/Results/predictions - Version 3.npy')  # Adjust the path accordingly

    # Apply KMedoids clustering
    clusters, medoids = apply_kmedoids_clustering(predictions_latent_space, max_iterations, num_clusters)

    # Save cluster assignments to file
    cluster_assignments_path = os.path.join(RESULTS_FOLDER, cluster_filename + '.txt')
    
    sorted_files = "filenames_test.txt"
    filenames_path = os.path.join(DATA_FOLDER, sorted_files)
    with open(filenames_path, 'r') as file:
        filenames = [line for line in file.read().splitlines() if 'North' in line]

    
    with open(cluster_assignments_path, 'w') as f:
        for filename, cluster in zip(filenames, clusters):
            f.write(f"{filename}: {cluster+1}\n")

    logging.info(f"Cluster assignments saved to {cluster_assignments_path}")

    # Plot clustering results
    plot_clustered_images(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path=RESULTS_FOLDER)
    del predictions_latent_space
    # Read cluster assignments from file
    cluster_to_filenames = read_cluster_assignments(cluster_assignments_path)

    # Select filenames for annotation
    selected_filenames = select_filenames_for_annotation(cluster_to_filenames, percentage_to_select=0.05)

    # Save selected filenames for annotation
    save_selected_filenames(selected_filenames, annotations_path)

    
    