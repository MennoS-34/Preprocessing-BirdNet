import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

# Function to read annotations from text files
def read_annotations(annotation_folder, num_clusters):
    annotations = []
    for cluster in range(1, num_clusters + 1):
        filename = os.path.join(annotation_folder, f"cluster{cluster}_annotations.txt")
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                filename, annotation = line.strip().split(":")
                annotations.append((filename, annotation.strip()))
    return annotations

# Function to plot 2D PCA with annotations
def plot_2d_pca_with_annotations(reduced_features, clusters, medoids, annotations, cluster_filename, num_clusters, save_path='results/'):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
    # Plot clusters
    for cluster_label in np.unique(clusters):
        cluster_points = reduced_features[clusters == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label+1}', color=colors[cluster_label], s=50)
    
    # Plot medoids
    plt.scatter(reduced_features[medoids, 0], reduced_features[medoids, 1], s=150, c='black', marker='X', label='Medoids')

    # Add annotations
    for filename, annotation in annotations:
        # Assuming filenames correspond to indices, this logic may need adjustment
        x = int(filename.split(':')[-1])  # Extracting the index from the filename
        cluster_label = clusters[x]
        color = colors[cluster_label]
        
        if annotation == 'Bird':
            marker = 'o'  # Bird
        elif annotation == 'Noise':
            marker = 's'  # Noise
        else:
            continue
        
        plt.scatter(reduced_features[x, 0], reduced_features[x, 1], marker=marker, color=color, s=100, edgecolors='black')
        plt.text(reduced_features[x, 0], reduced_features[x, 1], f' {annotation}', fontsize=12, ha='right', va='top')

    plt.title('Clustering Results (2D PCA with Annotations)', fontsize=20, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=18)
    plt.ylabel('Principal Component 2', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = cluster_filename + "-2D_PCA_with_annotations.png"
        filepath = os.path.join(save_path, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"The 2D PCA with annotations plot is saved at: {filepath}")
    else:
        plt.show()
        print("The 2D PCA with annotations plot is displayed.")

# Main script
if __name__ == "__main__":
    # Configuration
    ROOT_FOLDER = "/home/u993985/Thesis/"
    ANNOTATIONS_FOLDER = os.path.join(ROOT_FOLDER, "Annotations")
    RESULTS_FOLDER = os.path.join(ROOT_FOLDER, "Results")
    version = "Version 3"
    cluster_filename = f'cluster_results - {version}'
    num_clusters = 8

    # Load predictions_latent_space and apply clustering
    predictions_latent_space = np.load(os.path.join(ROOT_FOLDER, 'Results', 'predictions - Version 3.npy'))

    flat_images = predictions_latent_space.reshape(predictions_latent_space.shape[0], -1)
    flat_images = flat_images.astype('float16') / 255.0

    kmedoids_instance = KMedoids(n_clusters=num_clusters, init='k-medoids++', max_iter=100, random_state=42)
    kmedoids_instance.fit(flat_images)

    clusters = kmedoids_instance.labels_
    medoids = kmedoids_instance.medoid_indices_

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(flat_images)

    # Read annotations
    annotations = read_annotations(ANNOTATIONS_FOLDER, num_clusters)

    # Plot 2D PCA with annotations
    plot_2d_pca_with_annotations(reduced_features, clusters, medoids, annotations, cluster_filename, num_clusters, save_path=RESULTS_FOLDER)
