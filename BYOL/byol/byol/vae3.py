import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

#  CNNVae model
class CNNEncoder(nn.Module):
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.fc = nn.Linear(64*9*9, 128)
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        log_var = self.logvar(x)
        return mu, log_var

class CNNDecoder(nn.Module):
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 64*9*9)
        self.deconv_layers = nn.Sequential(
            # nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))        
        x = x.view(x.shape[0], 64, 9, 9)
        x = self.deconv_layers(x)
        return x

class CNNVAE(nn.Module):
    def __init__(self, z_dim, in_dim=150*150*1):
        super(CNNVAE, self).__init__()
        self.encoder = CNNEncoder(z_dim)
        self.decoder = CNNDecoder(z_dim)
        
    def reparam_trick(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(sigma) * torch.sqrt(torch.tensor(0.3, dtype=mu.dtype, device=mu.device))
        z = mu + (epsilon * sigma)
        return z
    
    def forward(self, x, mask_latent_dims):
        mu, log_var = self.encoder(x)
        z = self.reparam_trick(mu, log_var)

        if mask_latent_dims is not None:
            z[:, mask_latent_dims] = 0.0
        
        x_recon = self.decoder(z)
        return x_recon
    
    
    
# #Augmentation
# class VAEAugmentation:
#     def __init__(self, vae_model):
#         self.vae = vae_model
#         self.vae.eval()
#         self.to_tensor = T.ToTensor()

#     @torch.no_grad()
#     def __call__(self, img):
#         # Convert PIL to tensor if necessary
#         if not isinstance(img, torch.Tensor):
#             img = self.to_tensor(img)

#         if img.ndim == 3:
#             img = img.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

#         recon = self.vae(img, mask_latent_dims=None)
#         recon = recon.squeeze(0).cpu()  # (1,C,H,W) -> (C,H,W)
    
#         return to_pil_image(recon)

class VAEAugmentation:
    def __init__(self, vae_model, device="cuda", mask_latent_dims=None):
        self.device = device
        self.vae = vae_model.to(self.device)   # move VAE to GPU
        self.vae.eval()
        self.mask_latent_dims = mask_latent_dims
        self.to_tensor = T.ToTensor()

    @torch.no_grad()
    def __call__(self, img):
        # Convert PIL -> tensor if needed
        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)

        if img.ndim == 3:
            img = img.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

        # Move input to GPU
        img = img.to(self.device)

        # Run through VAE on GPU
        recon = self.vae(img, mask_latent_dims=self.mask_latent_dims)

        # Bring result back to CPU, convert to PIL
        recon = recon.squeeze(0).cpu()
        return to_pil_image(recon)
