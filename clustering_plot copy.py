import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Setting up Logging
logging.basicConfig(filename='Clustering_evaluation_Version_1.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ROOT_FOLDER = r"/home/u993985/Thesis/"
RESULTS_FOLDER = ROOT_FOLDER + r"Results/"

# Spectrogram specifics
spectrogram_h, spectrogram_w, spectrogram_d = 200, 300, 1

# Version description
version = "Version 3"
cluster_filename = f'cluster_results - {version}'

# Parameters
max_iterations = 100
num_clusters = 8

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

# Plot clustered images in 2D PCA
def plot_2d_pca(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path='results/'):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(flat_images)
    
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    for cluster_label in np.unique(clusters):
        cluster_points = reduced_features[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label+1}', color=colors[cluster_label], s=50)

    plt.scatter(reduced_features[medoids, 0], reduced_features[medoids, 1], s=150, c='black', marker='X', label='Medoids')

    plt.title('Clustering Results (2D PCA)', fontsize=20, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=18)
    plt.ylabel('Principal Component 2', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = cluster_filename + "-2D_PCA_plot.png"
        filepath = os.path.join(save_path, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logging.info(f"The 2D PCA clustering plot is saved at: {filepath}")
    else:
        plt.show()
        logging.info("The 2D PCA clustering plot is displayed.")

# Plot clustered images in 3D PCA
def plot_3d_pca(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path='results/'):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(flat_images)
    
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for cluster_label in np.unique(clusters):
        cluster_points = reduced_features[clusters == cluster_label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_label+1}', color=colors[cluster_label], s=50)

    ax.scatter(reduced_features[medoids, 0], reduced_features[medoids, 1], reduced_features[medoids, 2], s=150, c='black', marker='X', label='Medoids')

    ax.set_title('Clustering Results (3D PCA)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Principal Component 1', fontsize=18)
    ax.set_ylabel('Principal Component 2', fontsize=18)
    ax.set_zlabel('Principal Component 3', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', linewidth=0.5)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = cluster_filename + "-3D_PCA_plot.png"
        filepath = os.path.join(save_path, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logging.info(f"The 3D PCA clustering plot is saved at: {filepath}")
    else:
        plt.show()
        logging.info("The 3D PCA clustering plot is displayed.")

# Plot clustered images using t-SNE
def plot_tsne(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path='results/'):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(flat_images)
    
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    for cluster_label in np.unique(clusters):
        cluster_points = reduced_features[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label+1}', color=colors[cluster_label], s=50)

    plt.scatter(reduced_features[medoids, 0], reduced_features[medoids, 1], s=150, c='black', marker='X', label='Medoids')

    plt.title('Clustering Results (t-SNE)', fontsize=20, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=18)
    plt.ylabel('t-SNE Component 2', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = cluster_filename + "-tSNE_plot.png"
        filepath = os.path.join(save_path, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logging.info(f"The t-SNE clustering plot is saved at: {filepath}")
    else:
        plt.show()
        logging.info("The t-SNE clustering plot is displayed.")

# Plot Silhouette Plot
def plot_silhouette(predictions_latent_space, clusters, cluster_filename, num_clusters, save_path='results/'):
    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    silhouette_avg = silhouette_score(flat_images, clusters)
    sample_silhouette_values = silhouette_samples(flat_images, clusters)
    
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    for i in range(num_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1), fontsize=12)
        y_lower = y_upper + 10
    
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.title('Silhouette Plot', fontsize=20, fontweight='bold')
    plt.xlabel('Silhouette Coefficient', fontsize=18)
    plt.ylabel('Cluster', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = cluster_filename + "-Silhouette_plot.png"
        filepath = os.path.join(save_path, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logging.info(f"The Silhouette plot is saved at: {filepath}")
    else:
        plt.show()
        logging.info("The Silhouette plot is displayed.")

# Main execution
if __name__ == "__main__":
    # Load the predictions_latent_space
    predictions_latent_space = np.load('/home/u993985/Thesis/Results/predictions - Version 3.npy')  # Adjust the path accordingly

    # Apply KMedoids clustering
    clusters, medoids = apply_kmedoids_clustering(predictions_latent_space, max_iterations, num_clusters)

    # Plot clustering results
    plot_2d_pca(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path=RESULTS_FOLDER)
    plot_3d_pca(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path=RESULTS_FOLDER)
    plot_tsne(predictions_latent_space, clusters, medoids, cluster_filename, num_clusters, save_path=RESULTS_FOLDER)
    plot_silhouette(predictions_latent_space, clusters, cluster_filename, num_clusters, save_path=RESULTS_FOLDER)
