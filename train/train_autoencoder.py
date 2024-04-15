from torch.utils.data import DataLoader
from models.generative_model import build_autoencoder, build_discriminator
from loss.losses import *
from torch import optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
from utils.utils import log_reconstruction

def train_autoencoder(cfg, train_dataset, valid_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                              shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=cfg.TRAIN.PIN_MEMORY)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKERS,
                            pin_memory=cfg.TRAIN.PIN_MEMORY)
    
    # Loading Model
    model = build_autoencoder(cfg)
    discriminator = build_discriminator(cfg)
    perceptual_loss = build_perceptual_loss(cfg)
    reconstruction_loss = ReconstructionLoss(cfg)
    kl_divergence = KL_divergence()
    adv_loss = build_adversarial_loss(cfg)
    
    print(f"Num of GPU Counts: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)
        reconstruction_loss = torch.nn.DataParallel(reconstruction_loss)
        kl_divergence = torch.nn.DataParallel(kl_divergence)
        adv_loss = torch.nn.DataParallel(adv_loss)
        
    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)
    reconstruction_loss = reconstruction_loss.to(device)
    kl_divergence = kl_divergence.to(device)
    adv_loss = adv_loss.to(device)
    
    # Optimizers
    generator_optim = optim.Adam(model.parameters(), lr=cfg.stage1.base_lr)
    disc_optim = optim.Adam(discriminator.parameters(), lr=cfg.stage1.disc_lr)

    # Get Checkpoint
    best_loss = float("inf")
    scaler_gen = GradScaler()
    scaler_disc = GradScaler()
    
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        # Training
        model.train()
        discriminator.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        train_running_loss = 0.0
        for batch in pbar:
            images = batch["image"].to(device)
            # GENERATOR
            generator_optim.zero_grad()
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = model(x=images)
                r_loss = reconstruction_loss(reconstruction, images)
                p_loss = perceptual_loss(reconstruction.float(), images.float())

                kl_loss = kl_divergence(z_mu, z_sigma)
                if cfg.stage1.adv_weight > 0:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                else:
                    generator_loss = torch.tensor([0.0]).to(device)

                loss = r_loss + cfg.stage1.kl_weight * kl_loss + \
                    cfg.stage1.perceptual_weight * p_loss + \
                    cfg.stage1.adv_weight * generator_loss

                loss = loss.mean()
                r_loss = r_loss.mean()
                p_loss = p_loss.mean()
                kl_loss = kl_loss.mean()
                g_loss = generator_loss.mean()

                losses = dict(loss=loss, r_loss=r_loss, p_loss=p_loss, 
                            kl_loss=kl_loss, g_loss=g_loss)

            scaler_gen.scale(losses["loss"]).backward()
            scaler_gen.unscale_(generator_optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler_gen.step(generator_optim)
            scaler_gen.update()

            # DISCRIMINATOR
            if cfg.stage1.adv_weight > 0:
                disc_optim.zero_grad()

                with autocast(enabled=True):
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, 
                                           for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, 
                                           for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    d_loss = discriminator_loss
                    d_loss = d_loss.mean()

                scaler_disc.scale(d_loss).backward()
                scaler_disc.unscale_(disc_optim)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
                scaler_disc.step(disc_optim)
                scaler_disc.update()
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            losses["d_loss"] = discriminator_loss

            pbar.set_postfix(
                {
                    "epoch": epoch,
                    "loss": f"{losses['loss'].item():.4f}",
                    "r_loss": f"{losses['r_loss'].item():.4f}",
                    "p_loss": f"{losses['p_loss'].item():.4f}",
                    "g_loss": f"{losses['g_loss'].item():.4f}",
                    "d_loss": f"{losses['d_loss'].item():.4f}",
                },
            )
            train_running_loss += losses["loss"].item()
        train_loss = train_running_loss / len(train_loader)
        
        checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": generator_optim.state_dict(),
                "optimizer_d": disc_optim.state_dict(),
                "best_loss": best_loss,
            }
        torch.save(checkpoint, os.path.join(cfg.TRAIN.MODEL_PATH, 
                                            cfg.stage1.CHECKPOINT_NAME))
        
        # Evaluation
        model.eval()
        discriminator.eval()
        val_running_loss = 0.0
        print("===== Validation =====")
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                reconstruction, z_mu, z_sigma = model(x=images)
                r_loss = reconstruction_loss(reconstruction, images)
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                kl_loss = kl_divergence(z_mu, z_sigma)
                if cfg.stage1.adv_weight > 0:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, 
                                              for_discriminator=False)
                else:
                    generator_loss = torch.tensor([0.0]).to(device)

                loss = r_loss + cfg.stage1.kl_weight * kl_loss + \
                    cfg.stage1.perceptual_weight * p_loss + \
                    cfg.stage1.adv_weight * generator_loss

                loss = loss.mean()
                r_loss = r_loss.mean()
                p_loss = p_loss.mean()
                kl_loss = kl_loss.mean()
                g_loss = generator_loss.mean()

                losses = dict(loss=loss, r_loss=r_loss, p_loss=p_loss, 
                            kl_loss=kl_loss, g_loss=g_loss)

                val_running_loss += losses["loss"].item()
        
        images = next(iter(val_loader))["image"].to(device)
        reconstruction, z_mu, z_sigma = model(x=images)
        
        val_loss = val_running_loss / len(val_loader)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
        if val_loss < best_loss:
            print('===== Saving =====')
            log_reconstruction(images, reconstruction, epoch)
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(cfg.TRAIN.MODEL_PATH,
                                                        cfg.stage1.BEST_MODEL_NAME))
        