"""
Script di training per il modello 3D U-Net-GAN.

Uso:
    python train.py --data_dir /path/to/images --output_dir ./output
"""

import os
import time
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric

from config import Config
from model import UNet3DGenerator, CNN3DDiscriminator
from dataset import Mammogram2DTo3DDataset

# Riproducibilità
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net-GAN per mammografie")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Cartella con le immagini mammografiche (jpg/png/pgm)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Cartella di output per i pesi e i plot")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    return parser.parse_args()


def main():
    args = parse_args()
    Config.OUTPUT_DIR = args.output_dir
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"Dispositivo in uso: {DEVICE}")

    # Ricerca immagini
    all_image_files = []
    if args.data_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.pgm"):
            all_image_files.extend(glob.glob(os.path.join(args.data_dir, "**", ext), recursive=True))

    if len(all_image_files) >= 20:
        print(f"Trovate {len(all_image_files)} immagini reali!")
        random.shuffle(all_image_files)
        all_image_files = all_image_files[:Config.MAX_IMAGES]
    else:
        print("Nessun dataset trovato: generazione di 250 mammografie sintetiche per il training.")
        all_image_files = [f"SYNTHETIC_{i}" for i in range(250)]

    n_total = len(all_image_files)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_files = all_image_files[:n_train]
    val_files = all_image_files[n_train:n_train + n_val]

    print(f"Split -> Train: {len(train_files)} | Val: {len(val_files)}")

    train_loader = DataLoader(Mammogram2DTo3DDataset(train_files, augment=True),
                              batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(Mammogram2DTo3DDataset(val_files, augment=False),
                            batch_size=Config.BATCH_SIZE, shuffle=False)

    generator = UNet3DGenerator().to(DEVICE)
    discriminator = CNN3DDiscriminator().to(DEVICE)

    opt_G = torch.optim.Adam(generator.parameters(), lr=Config.LR,
                             betas=(Config.BETA1, Config.BETA2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=Config.LR,
                             betas=(Config.BETA1, Config.BETA2))

    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    history = {"G_loss": [], "D_loss": [], "val_mse": [], "val_ssim": []}
    best_val_mse = float("inf")
    weights_path = os.path.join(Config.OUTPUT_DIR, "unet3d_generator.pth")

    print(f"Inizio training per {Config.EPOCHS} epoche...")
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for seed_vol, target_vol, _ in train_loader:
            seed_vol = seed_vol.to(DEVICE)
            target_vol = target_vol.to(DEVICE)
            bs = seed_vol.size(0)

            real_labels = torch.full((bs, 1), 0.9, device=DEVICE)
            fake_labels = torch.full((bs, 1), 0.0, device=DEVICE)

            # Train Discriminator
            opt_D.zero_grad()
            out_real = discriminator(target_vol)
            loss_d_real = criterion_gan(out_real, real_labels)

            fake_vol = generator(seed_vol)
            out_fake = discriminator(fake_vol.detach())
            loss_d_fake = criterion_gan(out_fake, fake_labels)

            loss_D = 0.5 * (loss_d_real + loss_d_fake)
            loss_D.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            out_fake_for_g = discriminator(fake_vol)
            loss_g_gan = criterion_gan(out_fake_for_g, torch.ones((bs, 1), device=DEVICE))
            loss_g_l1 = criterion_l1(fake_vol, target_vol)

            loss_G = loss_g_gan + Config.LAMBDA_L1 * loss_g_l1
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item() * bs
            epoch_d_loss += loss_D.item() * bs

        epoch_g_loss /= len(train_loader.dataset)
        epoch_d_loss /= len(train_loader.dataset)

        # Validazione
        generator.eval()
        val_mse_list, val_ssim_list = [], []
        with torch.no_grad():
            for seed_vol, target_vol, _ in val_loader:
                seed_vol = seed_vol.to(DEVICE)
                target_vol = target_vol.to(DEVICE)
                pred_vol = generator(seed_vol)

                mse_batch = F.mse_loss(pred_vol, target_vol).item()
                val_mse_list.append(mse_batch)

                p_np = pred_vol[:, 0, 5, :, :].cpu().numpy()
                t_np = target_vol[:, 0, 5, :, :].cpu().numpy()
                for i in range(p_np.shape[0]):
                    val_ssim_list.append(ssim_metric(t_np[i], p_np[i], data_range=1.0))

        mean_val_mse = float(np.mean(val_mse_list))
        mean_val_ssim = float(np.mean(val_ssim_list))

        history["G_loss"].append(epoch_g_loss)
        history["D_loss"].append(epoch_d_loss)
        history["val_mse"].append(mean_val_mse)
        history["val_ssim"].append(mean_val_ssim)

        if mean_val_mse < best_val_mse:
            best_val_mse = mean_val_mse
            torch.save(generator.state_dict(), weights_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == Config.EPOCHS:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{Config.EPOCHS:03d}] | "
                  f"D_Loss: {epoch_d_loss:.4f} | G_Loss: {epoch_g_loss:.4f} | "
                  f"Val MSE: {mean_val_mse:.5f} | Val SSIM: {mean_val_ssim:.4f} | "
                  f"Time: {elapsed:.1f}s")

    print(f"\nTraining completato! Miglior modello salvato in: {weights_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["G_loss"], label="Generator Loss (L_GAN + \u03bbL1)")
    axes[0].plot(history["D_loss"], label="Discriminator Loss")
    axes[0].set_title("Adversarial Training Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_ssim"], color="green", label="Validation SSIM")
    axes[1].set_title("Validation SSIM across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "training_curves.png"), dpi=150)
    print(f"Plot salvato in: {os.path.join(Config.OUTPUT_DIR, 'training_curves.png')}")


if __name__ == "__main__":
    main()
