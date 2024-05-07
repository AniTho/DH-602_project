from torch.utils.data import DataLoader
from models.generative_model import build_autoencoder, build_diffusion_model, \
                                    build_scheduler, AutoEncoderSampler
from loss.losses import *
from torch import optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
from utils.utils import log_ldm_sample_unconditioned
from transformers import CLIPTextModel, CLIPTokenizer

def train_diffusion(cfg, train_dataset, valid_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                              shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=cfg.TRAIN.PIN_MEMORY)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKERS,
                            pin_memory=cfg.TRAIN.PIN_MEMORY)
    
    # Loading Model
    autoencoder = build_autoencoder(cfg)
    autoencoder.load_state_dict(torch.load(os.path.join(cfg.TRAIN.MODEL_PATH, 
                                                        cfg.stage1.BEST_MODEL_NAME)))
    auto_sampler = AutoEncoderSampler(autoencoder)
    auto_sampler.eval()
    
    diffusion_model = build_diffusion_model(cfg)
    scheduler = build_scheduler(cfg)
    text_encoder = CLIPTextModel.from_pretrained(cfg.text_encoder.model_name, 
                                                 subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.text_encoder.model_name, 
                                              subfolder="tokenizer")
    
    reconstruction_loss = ReconstructionLoss(cfg)
    
    print(f"Num of GPU Counts: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        diffusion_model = torch.nn.DataParallel(diffusion_model)
        reconstruction_loss = torch.nn.DataParallel(reconstruction_loss)
        text_encoder = torch.nn.DataParallel(text_encoder)
        auto_sampler = torch.nn.DataParallel(auto_sampler)
        
    diffusion_model = diffusion_model.to(device)
    reconstruction_loss = reconstruction_loss.to(device)
    text_encoder = text_encoder.to(device)
    auto_sampler = auto_sampler.to(device)
    
    # Optimizers
    optimizer = optim.AdamW(diffusion_model.parameters(), lr=cfg.ldm.base_lr)

    # Get Checkpoint
    best_loss = float("inf")
    scaler = GradScaler()
    
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        # Training
        diffusion_model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        train_running_loss = 0.0
        for batch in pbar:
            images = batch["image"].to(device)
            caption = batch["prompt"]
            
            # Tokenize prompt
            prompt_input = tokenizer(caption, padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,return_tensors="pt")['input_ids']
            prompt_input = prompt_input.to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, 
                                      (images.shape[0],), device=device).long()

            optimizer.zero_grad()
            with autocast(enabled=True):
                with torch.no_grad():
                    e = auto_sampler(images) * cfg.ldm.scale_factor

                prompt_embeds = text_encoder(prompt_input.squeeze(1))
                prompt_embeds = prompt_embeds[0]

                noise = torch.randn_like(e).to(device)
                noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
                noise_pred = diffusion_model(x=noisy_e, timesteps=timesteps, context=prompt_embeds)

                if scheduler.prediction_type == "v_prediction":
                    # Use v-prediction parameterization
                    target = scheduler.get_velocity(e, noise, timesteps)
                elif scheduler.prediction_type == "epsilon":
                    target = noise
                loss = reconstruction_loss(noise_pred, target)

            losses = dict(loss=loss)
            train_running_loss += losses["loss"].item()
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"epoch": epoch, 
                              "loss": f"{losses['loss'].item():.5f}"})
        
        train_loss = train_running_loss / len(train_loader)    
        checkpoint = {
                "epoch": epoch + 1,
                "diffusion": diffusion_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
        torch.save(checkpoint, os.path.join(cfg.TRAIN.MODEL_PATH, 
                                            cfg.ldm.CHECKPOINT_NAME))
        
        # Evaluation
        diffusion_model.eval()
        val_running_loss = 0.0
        print("===== Validation =====")
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                caption = batch["prompt"]
                # Tokenize prompt
                prompt_input = tokenizer(caption, padding="max_length",
                                        max_length=tokenizer.model_max_length,
                                        truncation=True,return_tensors="pt")['input_ids']
                prompt_input = prompt_input.to(device)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, 
                                          (images.shape[0],), device=device).long()

                with autocast(enabled=True):
                    e = auto_sampler(images) * cfg.ldm.scale_factor

                    prompt_embeds = text_encoder(prompt_input.squeeze(1))
                    prompt_embeds = prompt_embeds[0]

                    noise = torch.randn_like(e).to(device)
                    noisy_e = scheduler.add_noise(original_samples=e, noise=noise, 
                                                  timesteps=timesteps)
                    noise_pred = diffusion_model(x=noisy_e, timesteps=timesteps, 
                                                 context=prompt_embeds)

                    if scheduler.prediction_type == "v_prediction":
                        # Use v-prediction parameterization
                        target = scheduler.get_velocity(e, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise
                    loss = reconstruction_loss(noise_pred, target)

                loss = loss.mean()
                losses = dict(loss=loss)

                val_running_loss += losses["loss"].item()

        log_ldm_sample_unconditioned(diffusion_model=diffusion_model,
                                    auto_sampler=auto_sampler,
                                    text_encoder=text_encoder,
                                    scheduler=scheduler,
                                    spatial_shape=tuple(e.shape[1:]),
                                    device=device,
                                    scale_factor=cfg.ldm.scale_factor,
                                    epoch=epoch)
        val_loss = val_running_loss / len(val_loader)        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = train_loss
            torch.save(diffusion_model.state_dict(), os.path.join(cfg.TRAIN.MODEL_PATH,
                                                        cfg.ldm.BEST_MODEL_NAME))
        