import json
import os
import shutil
from collections import Counter

def get_dataset(json_path):
    # Base directory for the KL grade folders
    base_dir = "./datasets/processed_dataset"  # Adjust this to your desired path
    
    # Read the JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Create directories for each KL grade if they don't already exist
    kl_grade_dirs = {str(k): os.path.join(base_dir, str(k)) for k in range(5)}
    for dir_path in kl_grade_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Copy images to the corresponding KL grade directory
    for item in data:
        kl_grade = str(item["KL_GRADE"])
        source_image_path = item["IMAGE"]
        destination_dir = kl_grade_dirs[kl_grade]
        shutil.copy(source_image_path, destination_dir)
    
    # Count and print the number of images in each directory
    for grade, dir_path in kl_grade_dirs.items():
        num_images = len(os.listdir(dir_path))
        print(f"KL grade {grade} has {num_images} images.")

get_dataset('prompts.json')
