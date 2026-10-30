"""
Configurazione del progetto 3D Breast Mammogram U-Net-GAN.
"""

class Config:
    IMG_SIZE = 128
    NUM_SLICES = 10
    SLICE_INTERVAL = 5.0    # Delta z = 5 (Sez. IV-A)
    ALPHA_DECAY = 0.1       # alpha = 0.1 (Sez. IV-A)
    BATCH_SIZE = 16         # Batch size = 16 (Sez. IV-D & V)
    EPOCHS = 100            # 100 Epoche (Sez. IV-D & V)
    LR = 1e-4               # Learning rate = 1e-4 (Sez. V)
    BETA1 = 0.5             # Adam beta1 = 0.5 (Sez. V)
    BETA2 = 0.999           # Adam beta2 = 0.999 (Sez. V)
    LAMBDA_L1 = 50.0        # Peso L_recon rispetto a L_GAN (Eq. 4)
    ROI_THRESHOLD = 0.7     # Soglia globale tau = 0.7 (Eq. 5)
    S_XY = 0.5              # Pixel spacing default: 0.5 mm (Sez. IV-E)
    S_Z = 1.0               # Slice spacing default: 1.0 mm (Sez. IV-E)
    MAX_IMAGES = 800        # Limite immagini (nel paper sono 745)
    OUTPUT_DIR = "./output"
