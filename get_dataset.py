import json
import os
import shutil
from PIL import Image
from tqdm import tqdm

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
    
    loop = tqdm(data, total=len(data), leave=False)
    for item in loop:
        kl_grade = str(item["KL_GRADE"]).split('.')[0]
        side = item["SIDE"]
        try:
            source_image_path = item["IMAGE"]
        except KeyError:
            continue
        
        # Open and split the image
        img = Image.open(source_image_path)
        width, height = img.size
        center = width // 2
        
        # Determine which part to keep based on the side
        if side == 1:
            # SIDE=1 means it's a right knee, but it appears on the left side of the image
            side_to_save = img.crop((0, 0, center, height))
            suffix = "_R"
        elif side == 2:
            # SIDE=2 means it's a left knee, but it appears on the right side of the image
            side_to_save = img.crop((center, 0, width, height))
            suffix = "_L"
        
        # Save the relevant half to the corresponding KL grade directory
        filename = os.path.basename(source_image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        new_filename = filename_without_ext + suffix + '.jpg'
        destination_path = os.path.join(kl_grade_dirs[kl_grade], new_filename)
        side_to_save.save(destination_path)

    # Count and print the number of images in each directory
    for grade, dir_path in kl_grade_dirs.items():
        num_images = len(os.listdir(dir_path))
        print(f"KL grade {grade} has {num_images} images.")

get_dataset('prompts.json')
