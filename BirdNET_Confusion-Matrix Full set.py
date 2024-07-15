import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def read_annotations(file_path):
    annotations = {}
    print(f"Reading annotations from {file_path}")

    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    parts = line.strip().rsplit(': ', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        annotation = parts[1].strip()

                        # Replace .png with .BirdNET.results.txt
                        filename = filename.replace('.png', '.BirdNET.results.txt')

                        # Map annotation to 1 for 'Bird' and 0 for 'Noise'
                        if annotation == "Bird":
                            annotations[filename] = 1
                        else:
                            annotations[filename] = 0

                        print(f"Read annotation: {filename} -> {annotations[filename]}")
                    else:
                        raise ValueError("Invalid line format")

                except Exception as e:
                    print(f"Error reading line: {line.strip()}. Error: {e}")

    except Exception as e:
        print(f"Error opening or reading file {file_path}. Error: {e}")
        return None

    return annotations

def check_for_bird(file_path):
    print(f"Checking for 'Bird' in file {file_path}")
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            if "Bird" in file_content:
                print(f"'Bird' found in {file_path}")
                return True
            else:
                print(f"'Bird' not found in {file_path}")
                return False
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

def evaluate_predictions(annotations, predictions_dir):
    TP = FP = TN = FN = 0
    y_true = []
    y_pred = []

    annotated_birds = sum(annotations.values())
    predicted_birds = 0

    for filename, annotation in annotations.items():
        # Ensure filename matches what's in predictions_dir
        filename_in_predictions = os.path.basename(filename)
        file_path = os.path.join(predictions_dir, filename_in_predictions)
        
        print(f"Evaluating file: {filename_in_predictions}")

        if os.path.exists(file_path):
            is_bird_predicted = check_for_bird(file_path)

            y_true.append(annotation)
            y_pred.append(1 if is_bird_predicted else 0)

            if is_bird_predicted and annotation == 1:
                TP += 1
            elif is_bird_predicted and annotation == 0:
                FP += 1
            elif not is_bird_predicted and annotation == 1:
                FN += 1
            elif not is_bird_predicted and annotation == 0:
                TN += 1

            # Count predicted birds based on presence of "Bird" in file
            if is_bird_predicted:
                predicted_birds += 1

        else:
            print(f"Warning: {filename_in_predictions} not found in predictions directory")

    print(f"Annotated birds: {annotated_birds}")
    print(f"Predicted birds: {predicted_birds}")
    print(f"Evaluation results - TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    return TP, FP, TN, FN, y_true, y_pred

def compute_metrics(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    return precision, recall, accuracy

def plot_confusion_matrix(y_true, y_pred, predictions_dir):
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No predictions to plot.")
        return

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"$\\bf{{{v1}}}$\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(10, 8))
    sns.set(style="white", font="Times New Roman", font_scale=1.4)  # Times New Roman font

    # Use a formal beige palette
    cmap = sns.light_palette("#f5deb3", as_cmap=True)  # Wheat color, close to beige

    sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, cbar=True, annot_kws={"size": 16}, 
                linewidths=.5, linecolor='gray', square=True, cbar_kws={"shrink": .75})
    
    # Add labels, title and adjust appearance
    plt.xlabel('Predicted', fontsize=18, labelpad=20, fontname='Times New Roman')
    plt.ylabel('Actual', fontsize=18, labelpad=20, fontname='Times New Roman')
    plt.title('Confusion Matrix', fontsize=20, pad=20, fontname='Times New Roman', weight='bold')
    plt.xticks(ticks=[0.5, 1.5], labels=['Noise', 'Bird'], fontsize=16, fontname='Times New Roman')
    plt.yticks(ticks=[0.5, 1.5], labels=['Noise', 'Bird'], fontsize=16, rotation=0, fontname='Times New Roman')
    

    plt.tick_params(axis='both', which='both', length=0)
    output_file = os.path.join(predictions_dir, "Confusion_Matrix_Baseline.png")
    plt.savefig(output_file)
    plt.show()


annotation_files = [
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster1.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster2.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster3.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster4.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster5.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster6.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster7.txt',
    'D:\\Menno\\Thesis\\Expirement\\Clustering - Varation 3\\Annotations\\annotations_cluster8.txt'
]

predictions_dir = "D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Noise Reduce/B/"

total_TP = total_FP = total_TN = total_FN = 0
combined_y_true = []
combined_y_pred = []

# Process each annotation file
for annotation_file in annotation_files:
    annotations = read_annotations(annotation_file)
    TP, FP, TN, FN, y_true, y_pred = evaluate_predictions(annotations, predictions_dir)
    
    # Accumulate the results
    total_TP += TP
    total_FP += FP
    total_TN += TN
    total_FN += FN
    combined_y_true.extend(y_true)
    combined_y_pred.extend(y_pred)

# Compute and print metrics
precision, recall, accuracy = compute_metrics(total_TP, total_FP, total_TN, total_FN)
print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")
print(f"Overall Accuracy: {accuracy}")

# Plot the combined confusion matrix
plot_confusion_matrix(combined_y_true, combined_y_pred, predictions_dir)
