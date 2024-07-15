import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def modify_text_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        if os.stat(input_file).st_size == 0:
            f_out.write("Noise")
            return

        lines = f_in.readlines()
        output_lines = []

        for line in lines:
            if line.strip():  # Check if line is not empty
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    score = float(parts[3])  # Assuming score is in the fourth field
                    species_info = parts[2]  # Assuming species info is in the third field
                    noise_labels = ["Gun", "Engine", "Siren", "Fireworks", "Dog"]
                    if score >= 0.80 and species_info not in noise_labels:
                        label = 'Bird'
                    else:
                        label = 'Noise'
                    #label = 'Bird' if score >= 0.80 and species_info is not ["Gun", "Siren"] else 'Noise'
                    
                    modified_line = f'{species_info}\t{score}\t{label}\n'
                    output_lines.append(modified_line)

        f_out.writelines(output_lines)

def process_files_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for f in files:
        input_file = os.path.join(input_folder, f)
        output_file = os.path.join(output_folder, f)
        modify_text_file(input_file, output_file)


trail_folder = "B"
confidence = "0.80"
input_folder = f'D:/Menno/Thesis/Expirement/BirdNet Output/Noise Reduce/{trail_folder}/'
output_folder = f"D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Noise Reduce/{trail_folder}/"

process_files_in_folder(input_folder, output_folder)



"""
trail_folder = "C"
confidence = "0.80"
input_folder = f'D:/Menno/Thesis/Expirement/BirdNet_Output/Segmentation (4-6)/{trail_folder}/'
output_folder = f"D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Segmentation (4-6)/{confidence}/{trail_folder}/"

process_files_in_folder(input_folder, output_folder)
"""
