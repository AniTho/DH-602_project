import torch
from torch import nn
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss

class KL_divergence(nn.Module):
    def __init__(self):
        super(KL_divergence, self).__init__()

    def forward(self, z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = kl_loss.mean()
        return kl_loss
    
class ReconstructionLoss(nn.Module):
    def __init__(self, cfg):
        super(ReconstructionLoss, self).__init__()
        if cfg.stage1.loss == 'MSE':
            self.loss = nn.MSELoss()
        elif cfg.stage1.loss == 'mae':
            self.loss = nn.L1Loss()
            
    def forward(self, reconstruct, image):
        recon_loss = self.loss(reconstruct.float(), image.float())
        return recon_loss
    
def build_adversarial_loss(cfg):
    return PatchAdversarialLoss(criterion="least_squares", 
                                no_activation_leastsq=True)

def build_perceptual_loss(cfg):
    return PerceptualLoss(**cfg.perceptual_network.params)