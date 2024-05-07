from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
from torch import nn

def build_autoencoder(cfg):
    return AutoencoderKL(**cfg.stage1.params)

def build_discriminator(cfg):
    return PatchDiscriminator(**cfg.discriminator.params)

class AutoEncoderSampler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z

def build_diffusion_model(cfg):
    return DiffusionModelUNet(**cfg.ldm.params)

def build_scheduler(cfg):
    return DDPMScheduler(**cfg.ldm.scheduler)

def build_controlnet(cfg):
    return ControlNet(**cfg.controlnet.params)