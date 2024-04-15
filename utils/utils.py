from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import torch
from models.generative_model import build_autoencoder
import os

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