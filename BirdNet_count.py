import os

def count_noise_and_birds(folder_path):
    total_noise_count = 0
    total_bird_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path): 
            contains_bird = False
            
            with open(file_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if "Bird" in line:
                        contains_bird = True
                        break  #It should stop if it reads the string bird  
            
            if contains_bird:
                total_bird_count += 1
            else:
                total_noise_count += 1         

    print(f"Total counts across all files for: {trail}")
    print(f"    Total noise count: {total_noise_count}")
    print(f"    Total bird count: {total_bird_count}")
    print(f"    Entire total: {total_bird_count + total_noise_count}")




"""
trail_folder = "A"
confidence = "0.80"
output_folder = f"D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Segmentation (4-6)/{confidence}/{trail_folder}/"
trail = f"Output Baseline {trail_folder} "
count_noise_and_birds(output_folder)
"""

trail_folder = "B"
output_folder = f"D:/Menno/Thesis/Expirement/BirdNet_Labbeling/Noise Reduce/{trail_folder}/"
trail = f"Output Baseline {trail_folder} "
count_noise_and_birds(output_folder)

