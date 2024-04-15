from torchvision import transforms
import matplotlib.pyplot as plt
import random

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