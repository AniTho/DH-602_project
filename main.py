import argparse
import omegaconf
from data.kneeOA import KneeOADataset
from utils.utils import *
from train.train_autoencoder import train_autoencoder
from train.train_diffusion import train_diffusion
from train.train_controlnet import train_control
from monai.utils import set_determinism

def main(cfg):
    set_determinism(seed=cfg.TRAIN.SEED)
    img_transform = get_transforms(cfg)
    train_dataset = KneeOADataset(cfg, 'train', img_transform)
    valid_dataset = KneeOADataset(cfg, 'val', img_transform)
    if cfg.TRAIN_VAE:
        train_autoencoder(cfg, train_dataset, valid_dataset)
    elif cfg.TRAIN_DIFF:
        scaling_factor = ldm_scaling_factor(cfg, train_dataset)
        cfg.ldm.scale_factor = scaling_factor.item()
        train_diffusion(cfg, train_dataset, valid_dataset)
    elif cfg.TRAIN_CONTROL:
        scaling_factor = ldm_scaling_factor(cfg, train_dataset)
        cfg.ldm.scale_factor = scaling_factor.item()
        train_control(cfg, train_dataset, valid_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path of configuration yaml file", 
                        default="./configs/cfg.yaml")
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(args.config)
    main(cfg)