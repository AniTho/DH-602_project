from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator

def build_autoencoder(cfg):
    return AutoencoderKL(**cfg.stage1.params)

def build_discriminator(cfg):
    return PatchDiscriminator(**cfg.discriminator.params)