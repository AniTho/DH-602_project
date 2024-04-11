import pandas as pd
import json
import pathlib as pl
import os
import re

def generate_prompts(txt_path, json_path):
    txt_path = pl.Path(txt_path)
    json_path = pl.Path(json_path)
    
    df = pd.read_csv(txt_path, sep='|')
    df = df.dropna(subset=['ID', 'SIDE', 'V01XRKL', 'V01XRJSL', 'V01XRJSM'])
    prompts = []
    for _, row in df.iterrows():
        id = row['ID']
        side = row['SIDE']
        
        kl_grade = str(row['V01XRKL'])
        jsn_lateral = str(row['V01XRJSL'])
        jsn_medial = str(row['V01XRJSM'])
        
        descriptions = {
        "0": "Healthy knee image.",
        "1": "Doubtful joint narrowing with possible osteophytic lipping.",
        "2": "Definite presence of osteophytes and possible joint space narrowing.",
        "3": "Multiple osteophytes, definite joint space narrowing, with mild sclerosis.",
        "4": "Large osteophytes, significant joint narrowing, and severe sclerosis."
        }
        kl_description = descriptions[str(int(float(kl_grade)))]
#         print(f"ID: {id}, SIDE: {side}, KL Grade: {kl_grade}, JSN Lateral: {jsn_lateral}, JSN Medial: {jsn_medial}, KL Description: {kl_description}")

        # prompt_text = f"Generate a high-resolution X-ray image of the knee showing detailed anatomy with the following pathological features based on Kellgren and Lawrence grade (KL Grade): {kl_grade}, depicting {kl_description} Joint space narrowing in the lateral compartment: {jsn_lateral}, and medial compartment: {jsn_medial} reflecting cartilage degradation. Ensure clear visibility of bone contours, joint space, and any relevant pathological features indicative of the specified grades."

        prompt_text = f"Generate a high-resolution X-ray image of the knee showing detailed anatomy with the following pathological features based on Kellgren and Lawrence grade (KL Grade): {kl_grade}, depicting {kl_description} Ensure clear visibility of bone contours, joint space, and any relevant pathological features indicative of the specified grades."

        prompt_data = {
            "ID": id,
            "SIDE": side,
            "PROMPT": prompt_text
        }
        prompts.append(prompt_data)

    with open(json_path, 'w') as f:
        json.dump(prompts, f, indent=4)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        print(f"Prompts have been saved to {json_path}, total prompts: {len(data)}")
    
    return prompts

def search_jpg_files(directory):
    if not directory.exists():
        return None 
    
    for file_path in directory.iterdir():
        if file_path.is_dir():
            for sub_file in file_path.iterdir():
                if sub_file.is_file() and re.search(r"1x1\.jpg$", sub_file.name):
                    return sub_file
    return None

def searchTextAndUpdateJson(json_path):

    with open(json_path, 'r') as f:
        prompts = json.load(f)
    
    image_count = 0  

    for prompt in prompts:
        patient_id = prompt['ID']

        path_1 = pl.Path(f"./datasets/OAI12Month/downloads/results/1.C.2/{patient_id}")
        path_2 = pl.Path(f"./datasets/OAI12Month/downloads/results/1.E.1/{patient_id}")

        image_path = search_jpg_files(path_1)
        
        if not image_path:
            image_path = search_jpg_files(path_2)
        
        if image_path:
            prompt['Image'] = str(image_path)
            image_count += 1  

    with open(json_path, 'w') as f:
        json.dump(prompts, f, indent=4)

    print(f"Updated prompts have been saved to {json_path}, total image paths added: {image_count}")
    


prompts = generate_prompts("kxr_sq_bu01.txt", "prompts.json")
json_path = "prompts.json"   
searchTextAndUpdateJson(json_path)


