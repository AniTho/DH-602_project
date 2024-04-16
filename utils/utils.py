from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import torch
from models.generative_model import build_autoencoder
import os
from tqdm import tqdm
import numpy as np

def get_transforms(cfg):
    return transforms.Compose([
        transforms.Resize(cfg.DATA.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
def log_reconstruction(image, reconstruct, epoch, num_images = 4):
    image = (image.detach().cpu().numpy()*255).astype(int)
    reconstruct = (reconstruct.detach().cpu().numpy()*255).astype(int)
    num_images = min(num_images, image.shape[0])
    fig, ax = plt.subplots(2, num_images, figsize=(20, 5))
    img_choices = random.sample(list(range(image.shape[0])), num_images)
    for idx in range(num_images):
        ax[0, idx].imshow(image[img_choices[idx]].squeeze(), cmap='gray')
        ax[0, idx].set_title("Original")
        ax[0, idx].axis('off')
        ax[1, idx].imshow(reconstruct[img_choices[idx]].squeeze(), cmap='gray')
        ax[1, idx].set_title("Reconstructed")
        ax[1, idx].axis('off')
    plt.savefig(f"images/autoencoder/reconstruction_epoch_{epoch}.png")

def ldm_scaling_factor(cfg, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                              shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=cfg.TRAIN.PIN_MEMORY)


    # Load Autoencoder to produce the latent representations
    print(f"Loading Autoencoder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = build_autoencoder(cfg)
    autoencoder.load_state_dict(torch.load(os.path.join(cfg.TRAIN.MODEL_PATH, 
                                                        cfg.stage1.BEST_MODEL_NAME)))
    autoencoder.eval()
    autoencoder = autoencoder.to(device)

    eda_data = next(iter(train_loader))["image"]

    with torch.no_grad():
        z = autoencoder.encode_stage_2_inputs(eda_data.to(device))

    return 1 / torch.std(z)

@torch.no_grad()
def log_ldm_sample_unconditioned(diffusion_model, auto_sampler, text_encoder,
    scheduler, spatial_shape, device, scale_factor, epoch):
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
    prompt_embeds = prompt_embeds.to(device)
    prompt_embeds = text_encoder(prompt_embeds.squeeze(1))
    prompt_embeds = prompt_embeds[0]

    for t in tqdm(scheduler.timesteps, ncols=70, leave=False):
        noise_pred = diffusion_model(x=latent, 
                                    timesteps=torch.asarray((t,)).to(device), 
                                    context=prompt_embeds)
        latent, _ = scheduler.step(noise_pred, t, latent)

    x_hat = auto_sampler.model.decode(latent / scale_factor)
    img_0 = np.clip(a=x_hat[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    fig = plt.figure(dpi=300)
    plt.imshow(img_0, cmap="gray")
    plt.axis("off")
    plt.savefig(f"images/ldm_sample_gen/unconditioned_{epoch}.png")