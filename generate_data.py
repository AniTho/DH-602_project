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

cfg = omegaconf.OmegaConf.load('configs/cfg.yaml')
set_determinism(seed=cfg.TRAIN.SEED)

img_transform = get_transforms(cfg)
train_dataset = KneeOADataset(cfg, 'train', img_transform)
scaling_factor = ldm_scaling_factor(cfg, train_dataset)
cfg.ldm.scale_factor = scaling_factor.item()
train_loader = DataLoader(train_dataset, batch_size=4, 
                        shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                        pin_memory=cfg.TRAIN.PIN_MEMORY)
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

class_0_images = []
# class_1_images = []
# class_2_images = []
# class_3_images = []
# class_4_images = []
for batch in train_loader:
    conditioning_images = batch['image']
    conditional_prompts = batch['prompt']
    kl_grade = batch['kl_grade']
    
    class_0_images.extend(conditioning_images[torch.where(kl_grade == 0)[0]])
    # class_1_images.extend(conditioning_images[torch.where(kl_grade == 1)[0]])
    # class_2_images.extend(conditioning_images[torch.where(kl_grade == 2)[0]])
    # class_3_images.extend(conditioning_images[torch.where(kl_grade == 3)[0]])
    # class_4_images.extend(conditioning_images[torch.where(kl_grade == 4)[0]])
class_0_images = torch.stack(class_0_images).to(device)
# class_1_images = torch.stack(class_1_images).to(device)
# class_2_images = torch.stack(class_2_images).to(device)
# class_3_images = torch.stack(class_3_images).to(device)
# class_4_images = torch.stack(class_4_images).to(device)

@torch.no_grad()
def transform_to_kl_grade(images, current_kl_grade, target_kl_grade,
                          controlnet_scale=1.0, guidance_scale=2.0):
    os.makedirs(f"./datasets/zero_to_all/{target_kl_grade}", exist_ok=True)
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

    progressive_images = []
    bs = 16
    caption = [prompt_text]*bs
    prompt_input = tokenizer(caption, padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,return_tensors="pt")['input_ids']
    prompt_input = prompt_input.to(device)
    for i in tqdm(range(0, len(images), bs)):
        condition = images[i:i+bs]
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
        x_hat = x_hat.cpu().numpy()[:,0]
        progressive_images.extend(x_hat)
    
    for idx, img in enumerate(progressive_images):
        plt.imsave(f"./datasets/zero_to_all/{target_kl_grade}/image_{idx}.png", img, cmap='gray')

# random_idx_progress = random.sample(list(range(len(class_0_images))), 1243)
# sample_images = class_0_images[random_idx_progress]
# transform_to_kl_grade(sample_images, '0', '1')

# random_idx_progress = random.sample(list(range(len(class_0_images))), 773)
# sample_images = class_0_images[random_idx_progress]
# transform_to_kl_grade(sample_images, '0', '2')

# random_idx_progress = random.sample(list(range(len(class_0_images))), 1532)
# sample_images = class_0_images[random_idx_progress]
# transform_to_kl_grade(sample_images, '0', '3')

random_idx_progress = random.sample(list(range(len(class_0_images))), 2116)
sample_images = class_0_images[random_idx_progress]
transform_to_kl_grade(sample_images, '0', '4')