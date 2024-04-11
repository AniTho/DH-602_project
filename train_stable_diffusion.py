import json
import os
import shutil
from tqdm import tqdm
from PIL import Image

def parse_json_and_prepare_data(json_path):
    # Read JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Base directory for the KL grade folders
    base_dir = "./datasets/processed_dataset"  

    concepts_list = []

    # Prepare directories and concepts list
    loop = tqdm(data, total=len(data), leave=False)
    for item in loop:
        kl_grade = str(int(float(item["KL_GRADE"])))  # Convert KL grade to integer and then to string
        prompt = item["PROMPT"]
        
        # Create directories if they don't already exist
        class_data_dir = os.path.join(base_dir, kl_grade)

        # Add concept for this instance if not already added
        concept = {
            "class_prompt": prompt,  
            "instance_prompt": prompt,
            "instance_data_dir": class_data_dir,
            "class_data_dir": class_data_dir
        }
        if concept not in concepts_list:
            concepts_list.append(concept)
    
    # print(f'Concepts list: {concepts_list}')

    # Write the concepts list to a JSON file
    with open(os.path.join("concepts_list.json"), "w") as f:
        json.dump(concepts_list, f, indent=4)

def train_model():
    # Assuming all necessary parameters and environment are set
    os.system(f"accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path='stabilityai/stable-diffusion-2' \
      --pretrained_vae_name_or_path='stabilityai/sd-vae-ft-mse' \
      --output_dir='./stable_diffusion_weights' \
      --revision='fp16' \
      --with_prior_preservation --prior_loss_weight=1.0 \
      --seed=1337 \
      --resolution=512 \
      --train_batch_size=1 \
      --train_text_encoder \
      --mixed_precision='fp16' \
      --use_8bit_adam \
      --gradient_accumulation_steps=1 \
      --gradient_checkpointing \
      --learning_rate=2e-6 \
      --lr_scheduler='constant' \
      --lr_warmup_steps=0 \
      --sample_batch_size=128 \
      --max_train_steps=100 \
      --save_interval=500 \
      --not_cache_latents \
      --concepts_list='concepts_list.json'")

# Setup parameters
json_path = "prompts.json"
output_dir = "./stable_diffusion_weights"

# Prepare data and train the model
# parse_json_and_prepare_data(json_path)
train_model()
