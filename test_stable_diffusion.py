import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os

model_path = './stable_diffusion_weights/1000'            
imgs_dir = "./generated_imgs"
os.system(f"rm -rf {imgs_dir}/*")

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16, low_cpu_mem_usage=False).to("cuda")

g_cuda = torch.Generator(device='cuda')
seed = 0 #@param {type:"number"}
g_cuda.manual_seed(seed)

prompt = "Generate a high-resolution image of a knee" #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 100 #@param {type:"number"}
height = 1024 #@param {type:"number"}
width = 1024 #@param {type:"number"}

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for i, img in enumerate(images):
    img.save(f"./generated_imgs/{i}.png")