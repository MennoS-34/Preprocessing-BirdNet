import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def count_species_above_threshold(file_path, threshold=0.80):
    species_count = Counter()
    if os.stat(file_path).st_size == 0:
        return species_count

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # Check if line is not empty
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        species = parts[0].split(',')[1].strip()  # Assuming species is in the first field after the comma
                        score = float(parts[1])  # Assuming score is in the second field
                        if score > threshold:
                            species_count[species] += 1
                    except (ValueError, IndexError):
                        continue  # Skip lines where the score is not a valid float or index errors occur
    return species_count

def process_folder_for_species_count(input_folder, threshold=0.80):
    total_species_counts = Counter()
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    for f in files:
        file_path = os.path.join(input_folder, f)
        species_count = count_species_above_threshold(file_path, threshold)
        total_species_counts.update(species_count)

    return total_species_counts

def save_species_counts_to_file(species_counts, output_file):
    with open(output_file, 'w') as file:
        for species, count in species_counts.items():
            file.write(f"{species}: {count}\n")

def save_species_counts_to_csv(species_counts, csv_file):
    df = pd.DataFrame(species_counts.items(), columns=['Species', 'Count'])
    df.to_csv(csv_file, index=False)

def plot_histogram(species_counts, output_image, min_count=5):
    filtered_counts = {species: count for species, count in species_counts.items() if count > min_count}
    species = list(filtered_counts.keys())
    counts = list(filtered_counts.values())
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(species, counts, color='skyblue', edgecolor='black')  # Change bar color here

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
                 ha='center', va='bottom', fontsize=12, fontname='Times New Roman')  # Use Times New Roman font here

    plt.title(f'Distribution of Species Counts Above Threshold: {min_count}', fontsize=20, fontweight='bold', fontname='Times New Roman')  # Use Times New Roman font here
    plt.xlabel('Species', fontsize=18, fontname='Times New Roman')  # Use Times New Roman font here
    plt.ylabel('Count', fontsize=18, fontname='Times New Roman')  # Use Times New Roman font here
    plt.xticks(rotation=90, fontsize=14, fontname='Times New Roman')  # Use Times New Roman font here
    plt.yticks(fontsize=14, fontname='Times New Roman')  # Use Times New Roman font here
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()

# Example usage
input_folder = 'D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Wiener/0.80/B'  # Replace with your input folder path
output_file = 'D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Wiener/0.80/species_counts.txt'  # File to save species counts
csv_file = 'D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Wiener/0.80/species_counts.csv'  # CSV file to save species counts
output_image = 'D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Wiener/0.80/species_histogram.png'  # File to save the histogram

# Process folder and get species counts
species_counts = process_folder_for_species_count(input_folder)

# Save counts to a file
save_species_counts_to_file(species_counts, output_file)

# Save counts to a CSV file
save_species_counts_to_csv(species_counts, csv_file)

# Plot and save histogram
plot_histogram(species_counts, output_image)

print("Script executed successfully!")
