import argparse
import omegaconf
from data.kneeOA import KneeOADataset
from utils.utils import *
from train.train_autoencoder import train_autoencoder
from train.train_diffusion import train_diffusion
from train.train_controlnet import train_control
from monai.utils import set_determinism
from train.train_classification_models import train_classification_model
from icecream import ic
import wandb
# ic.disable()

def main(cfg):
    if args.wandb:
        wandb.init(project="DH602", name=f"{cfg.MODEL.BACKBONE}_{cfg.DATA.LONG_TAILED}")
    else:
        wandb.init(mode="dryrun")
    
    wandb.save('./configs/cfg.yaml')
    
    set_determinism(seed=cfg.TRAIN.SEED)
    img_transform = get_transforms(cfg)
    train_dataset = KneeOADataset(cfg, 'train', img_transform)
    valid_dataset = KneeOADataset(cfg, 'val', img_transform)
    test_dataset = KneeOADataset(cfg, 'test', img_transform)
    
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
    elif cfg.TRAIN_CLS_MODEL:
        train_classification_model(cfg, train_dataset, valid_dataset, test_dataset)      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path of configuration yaml file", 
                        default="./configs/cfg.yaml")
    parser.add_argument("--wandb", action="store_true", help="use wandb")
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(args.config)
    main(cfg)
    wandb.finish()