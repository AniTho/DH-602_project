import zipfile
import os
from tqdm import tqdm

def unzip_file_with_progress(zip_file_path, extract_to_folder):
    # Make sure the output directory exists
    if not os.path.exists(extract_to_folder):
        os.makedirs(extract_to_folder)
    
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get a list of all archived file names from the zip
        all_files = zip_ref.namelist()
        
        # Loop through each file
        for file in tqdm(all_files, desc="Extracting files"):
            # Extract each file to another directory
            zip_ref.extract(member=file, path=extract_to_folder)

# Specify your zip file path and the extraction directory
zip_file_path = 'project/datasets/OAI12Month/downloads/results/12m.zip'
extract_to_folder = 'project/datasets/OAI12Month/downloads/results/extract'

unzip_file_with_progress(zip_file_path, extract_to_folder)
