# ===============================
# 1. Import Libraries and Dataset
# ===============================
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 2. Custom Dataset with Random Masking
# ===============================
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, img_size=256):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
        if not self.img_files:
            raise ValueError(f"No images found in {img_dir}")
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Create random mask (both rectangular and irregular shapes)
        mask = torch.ones_like(image)
        h, w = image.shape[1], image.shape[2]
        
        # Randomly choose between rectangular or irregular mask
        if np.random.rand() > 0.5:
            # Rectangular mask
            mask_h, mask_w = np.random.randint(int(h*0.1), int(h*0.5)), np.random.randint(int(w*0.1), int(w*0.5))
            mask_x, mask_y = np.random.randint(0, h - mask_h), np.random.randint(0, w - mask_w)
            mask[:, mask_x:mask_x + mask_h, mask_y:mask_y + mask_w] = 0
        else:
            # Irregular mask (random points)
            for _ in range(np.random.randint(1, 5)):
                cx, cy = np.random.randint(0, w), np.random.randint(0, h)
                radius = np.random.randint(int(w*0.05), int(w*0.2))
                y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
                dist = (x - cx)**2 + (y - cy)**2
                mask[:, dist < radius**2] = 0

        corrupted_img = image * mask
        return corrupted_img, image, mask

# Load dataset
img_dir = "images"  # Update with your Google Drive path
dataset = CustomImageDataset(img_dir, img_size=256)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# ===============================
# 3. Define Enhanced Generator and Critic
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 256x256
            nn.Tanh()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64x64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32x32
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 16x16
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0)  # 1x1
        )

    def forward(self, x):
        return self.model(x).view(-1)

# ===============================
# 4. Enhancement Network (as in the paper)
# ===============================
class EnhancementNetwork(nn.Module):
    def __init__(self):
        super(EnhancementNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # Conv+ReLU
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # Conv+BN+ReLU
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # Conv+BN+ReLU
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)   # Conv
        )
        
    def forward(self, x):
        return self.model(x)

# ===============================
# 5. Initialize Models and Optimizers
# ===============================
G = Generator().to(device)
D = Critic().to(device)
E = EnhancementNetwork().to(device)

# Learning rates as per paper
lr_G = 1e-4
lr_D = 4e-4
lr_E = 1e-4

optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.9))
optimizer_E = optim.Adam(E.parameters(), lr=lr_E, betas=(0.5, 0.9))

# ===============================
# 6. Loss Functions
# ===============================
def gradient_penalty(D, real, fake):
    """ Compute gradient penalty for Wasserstein GAN """
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def contextual_loss(generated, real, mask):
    """ L1 loss between non-masked regions """
    return torch.mean(torch.abs(mask * generated - mask * real))

def perceptual_loss(discriminator, generated):
    """ How realistic the generated image looks """
    return -torch.mean(discriminator(generated))

# ===============================
# 7. Training Loop
# ===============================
epochs = 10000  # As in the paper
lambda_gp = 10  # Gradient penalty coefficient
lambda_context = 1.0  # Contextual loss weight
n_critic = 5  # Number of critic iterations per generator iteration

# For tracking best model
best_psnr = 0
best_ssim = 0

for epoch in range(epochs):
    start_time = time.time()
    
    for i, (corrupted, real, mask) in enumerate(dataloader):
        corrupted, real, mask = corrupted.to(device), real.to(device), mask.to(device)
        batch_size = corrupted.size(0)
        
        # ---------------------
        # Train Critic (Discriminator)
        # ---------------------
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            
            # Generate fake images
            fake = G(corrupted)
            
            # Critic scores
            d_real = D(real)
            d_fake = D(fake.detach())
            
            # Gradient penalty
            gp = gradient_penalty(D, real, fake.detach())
            
            # Wasserstein loss with gradient penalty
            loss_D = -torch.mean(d_real) + torch.mean(d_fake) + lambda_gp * gp
            
            loss_D.backward()
            optimizer_D.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        
        fake = G(corrupted)
        
        # Total generator loss (contextual + perceptual)
        loss_context = contextual_loss(fake, real, mask)
        loss_perceptual = perceptual_loss(D, fake)
        loss_G = lambda_context * loss_context + loss_perceptual
        
        loss_G.backward()
        optimizer_G.step()
        
        # ---------------------
        # Train Enhancement Network (every 10 batches)
        # ---------------------
        if i % 10 == 0:
            optimizer_E.zero_grad()
            
            # Generate completed image
            completed = G(corrupted)
            
            # Enhance the completed image
            enhanced = E(completed)
            
            # Residual learning loss
            residual = real - completed
            loss_E = torch.mean((E(completed) - residual) ** 2)
            
            loss_E.backward()
            optimizer_E.step()
    
    # ---------------------
    # Evaluation and Logging
    # ---------------------
    if epoch % 10 == 0:
        with torch.no_grad():
            # Get sample batch for visualization
            sample_corrupted, sample_real, sample_mask = next(iter(dataloader))
            sample_corrupted = sample_corrupted.to(device)
            
            # Generate and enhance sample
            sample_fake = G(sample_corrupted)
            sample_enhanced = E(sample_fake)
            
            # Calculate metrics
            sample_real_np = sample_real.cpu().numpy().transpose(0, 2, 3, 1)
            sample_fake_np = sample_fake.cpu().numpy().transpose(0, 2, 3, 1)
            sample_enhanced_np = sample_enhanced.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Denormalize images for metric calculation
            sample_real_np = (sample_real_np * 0.5 + 0.5) * 255
            sample_fake_np = (sample_fake_np * 0.5 + 0.5) * 255
            sample_enhanced_np = (sample_enhanced_np * 0.5 + 0.5) * 255
            
            psnr_fake = np.mean([peak_signal_noise_ratio(sample_real_np[i], sample_fake_np[i]) 
                                for i in range(len(sample_real_np))])
            psnr_enhanced = np.mean([peak_signal_noise_ratio(sample_real_np[i], sample_enhanced_np[i]) 
                                   for i in range(len(sample_real_np))])
            
            ssim_fake = np.mean([structural_similarity(sample_real_np[i], sample_fake_np[i], multichannel=True, 
                                                     data_range=255) 
                               for i in range(len(sample_real_np))])
            ssim_enhanced = np.mean([structural_similarity(sample_real_np[i], sample_enhanced_np[i], multichannel=True, 
                                                         data_range=255) 
                                   for i in range(len(sample_real_np))])
            
            # Update best models
            if psnr_enhanced > best_psnr:
                best_psnr = psnr_enhanced
                torch.save(G.state_dict(), 'best_generator.pth')
                torch.save(E.state_dict(), 'best_enhancer.pth')
            
            if ssim_enhanced > best_ssim:
                best_ssim = ssim_enhanced
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | Loss E: {loss_E.item():.4f}")
            print(f"PSNR - Fake: {psnr_fake:.2f} dB | Enhanced: {psnr_enhanced:.2f} dB")
            print(f"SSIM - Fake: {ssim_fake:.4f} | Enhanced: {ssim_enhanced:.4f}")
            print(f"Best PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
            print(f"Time: {time.time() - start_time:.2f}s")
            
            # Visualize samples
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            titles = ['Corrupted', 'Generated', 'Enhanced', 'Original']
            images = [sample_corrupted[0], sample_fake[0], sample_enhanced[0], sample_real[0]]
            
            for ax, img, title in zip(axes, images, titles):
                img = img.cpu().detach().permute(1, 2, 0).numpy()
                img = img * 0.5 + 0.5  # Denormalize
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            
            plt.show()