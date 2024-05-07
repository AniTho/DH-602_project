import streamlit as st
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.models as models
import torch.nn as nn
import timm

import omegaconf
from data.kneeOA import KneeOADataset
from utils.utils import *
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from models.generative_model import build_autoencoder, build_diffusion_model, \
                                    build_scheduler, AutoEncoderSampler, \
                                    build_controlnet
import torch
from transformers import CLIPTextModel, CLIPTokenizer
import gc

cfg = omegaconf.OmegaConf.load('./configs/cfg.yaml')
set_determinism(seed=cfg.TRAIN.SEED)

img_transform = get_transforms(cfg)
train_dataset = KneeOADataset(cfg, 'train', img_transform)
scaling_factor = ldm_scaling_factor(cfg, train_dataset)
cfg.ldm.scale_factor = scaling_factor.item()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gc.collect()
torch.cuda.empty_cache()

# Loading Model
autoencoder = build_autoencoder(cfg)
autoencoder.load_state_dict(torch.load(os.path.join(cfg.TRAIN.MODEL_PATH, 
                                                    cfg.stage1.BEST_MODEL_NAME)))
auto_sampler = AutoEncoderSampler(autoencoder)
auto_sampler.eval()

diffusion_model = build_diffusion_model(cfg)
diffusion_model.load_state_dict(torch.load(os.path.join(cfg.TRAIN.MODEL_PATH, 
                                                        cfg.ldm.BEST_MODEL_NAME)))
diffusion_model.eval()

controlnet = build_controlnet(cfg)

# Copy weights from the DM to the controlnet
controlnet.load_state_dict(torch.load(os.path.join(cfg.TRAIN.MODEL_PATH,
                                                   cfg.controlnet.CHECKPOINT_NAME))["controlnet"])
controlnet.eval()
    
scheduler = build_scheduler(cfg)
text_encoder = CLIPTextModel.from_pretrained(cfg.text_encoder.model_name, 
                                                subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(cfg.text_encoder.model_name, 
                                            subfolder="tokenizer")

autoencoder = autoencoder.to(device)
diffusion_model = diffusion_model.to(device)
controlnet = controlnet.to(device)
text_encoder = text_encoder.to(device)

@torch.no_grad()
def transform_to_kl_grade(image, current_kl_grade, target_kl_grade,
                          controlnet_scale=1.0, guidance_scale=2.0):
    image = img_transform(image)
    descriptions = {
        "0": "Healthy knee image",
        "1": "Doubtful joint narrowing with possible osteophytic lipping",
        "2": "Definite presence of osteophytes and possible joint space narrowing",
        "3": "Multiple osteophytes, definite joint space narrowing, with mild sclerosis",
        "4": "Large osteophytes, significant joint narrowing, and severe sclerosis"
        }
    prompt_text = f"Generate a high-resolution X-ray image of the knee with \
Kellgren and Lawrence grade (KL Grade): {target_kl_grade}, depicting \
{descriptions[target_kl_grade]} from a conditioned image of Kellgren and \
Lawrence grade (KL Grade): {current_kl_grade}, depicting {descriptions[current_kl_grade]}"

    caption = [prompt_text]
    prompt_input = tokenizer(caption, padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,return_tensors="pt")['input_ids']
    prompt_input = prompt_input.to(device)
    condition = image[0:1,:,:].unsqueeze(0).to(device)
    prompt_embeds = text_encoder(prompt_input.squeeze(1))
    prompt_embeds = prompt_embeds[0]
        
    latent = torch.randn((len(condition),) + (3,28,28))
    latent = latent.to(device)
    uncond_prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
    uncond_prompt_embeds = uncond_prompt_embeds.to(device)
    uncond_prompt_embeds = text_encoder(uncond_prompt_embeds.squeeze(1))
    uncond_prompt_embeds = uncond_prompt_embeds[0]
    uncond_prompt_embeds = uncond_prompt_embeds.expand(len(condition), *uncond_prompt_embeds.shape[1:])
    for t in scheduler.timesteps:
        down_block_res_samples, mid_block_res_sample = controlnet(
                                x=latent,
                                timesteps=torch.asarray((t,)).to(device).long(),
                                context=prompt_embeds,
                                controlnet_cond=condition,
                                conditioning_scale=controlnet_scale,
                            )
        model_output = diffusion_model(
                                x=latent,
                                timesteps=torch.asarray((t,)).to(device).long(),
                                context=prompt_embeds,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            )
        model_output_uncond = diffusion_model(
                                x=latent,
                                timesteps=torch.asarray((t,)).to(device).long(),
                                context=uncond_prompt_embeds,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            )
        noise_pred_uncond, noise_pred_text = model_output, model_output_uncond
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latent, _ = scheduler.step(noise_pred, t, latent)
    x_hat = auto_sampler.model.decode(latent / scaling_factor)
    x_hat = torch.clip(x_hat, 0, 1)
    x_hat = x_hat.cpu().numpy()[0][0]
    # convert x_hat to PIL image
    x_hat = Image.fromarray(np.uint8(x_hat*255))
    return x_hat

# Streamlit page configuration
icon = Image.open("./knee_OA_dl_app/app/img/iitb_logo.png")
st.set_page_config(page_title="Knee OA Severity Analysis", page_icon=icon)    
# UI Sidebar
with st.sidebar:
    st.image(icon)
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>Team PADMA</h2>", unsafe_allow_html=True)
    st.caption("===DH 602 Course Project===")
    gen_kl_grade=st.text_input("Write the KL Grade you want to generate")
    input_kl_grade=st.text_input("Write the KL Grade of input image")
    uploaded_file = st.file_uploader("Choose X-Ray image")
    
st.header("Knee Osteoarthritis X-Ray Image Generation")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":camera: Input X-Ray Image")
        st.image(img, use_column_width=True)

        # with col2:
    with col2:
        st.subheader(":mag: Generated X-Ray Image")
        st.image(transform_to_kl_grade(img, input_kl_grade, gen_kl_grade), use_column_width=True)
                

