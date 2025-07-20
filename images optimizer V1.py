# ===============================
# 1. استيراد المكتبات والمجموعة البيانات
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
from torch.nn.utils import spectral_norm

# التحقق من وجود GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 2. مجموعة بيانات مخصصة مع إخفاء عشوائي
# ===============================
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))][:100]  # زيادة عدد الصور
        if not self.img_files:
            raise ValueError(f"No images found in {img_dir}")
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  # زيادة حجم الصورة
            transforms.RandomHorizontalFlip(),  # زيادة التنوع
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # تطبيع لكل قناة لونية
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            # إنشاء قناع مستطيل عشوائي
            mask = torch.ones_like(image)
            h, w = image.shape[1], image.shape[2]
            mask_h, mask_w = np.random.randint(30, 100), np.random.randint(30, 100)  # زيادة حجم القناع
            mask_x, mask_y = np.random.randint(0, h - mask_h), np.random.randint(0, w - mask_w)
            mask[:, mask_x:mask_x + mask_h, mask_y:mask_y + mask_w] = 0

            corrupted_img = image * mask
            return corrupted_img, image, mask
        except Exception as e:
            print(f"Error loading image {self.img_files[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # تجربة الصورة التالية في حالة خطأ

# ===============================
# 3. تعريف المولد والناقد المحسن
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 3, 4, 2, 1)),
            nn.Tanh()
        )
        
        # Skip connections
        self.skip1 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 64, 1)),
            nn.InstanceNorm2d(64)
        )
        self.skip2 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 128, 1)),
            nn.InstanceNorm2d(128)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder with skip connections
        d1 = self.dec1(e3) + self.skip2(e2)
        d2 = self.dec2(d1) + self.skip1(e1)
        d3 = self.dec3(d2)
        
        return d3

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0))  # PatchGAN output
        )

    def forward(self, x):
        return self.model(x)

# ===============================
# 4. إعداد التدريب مع التحسينات
# ===============================
def main():
    # تحميل البيانات
    img_dir = "images"
    dataset = CustomImageDataset(img_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, 
                          num_workers=0, pin_memory=False)  # تغيير هنا لتفادي مشاكل multiprocessing
    
    # تهيئة النماذج
    G = Generator().to(device)
    D = Critic().to(device)
    
    # المُحسّنات
    lr = 2e-4
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    epochs = 200  # زيادة عدد الحقب
    lambda_gp = 10
    lambda_l1 = 100  # وزن L1 loss
    
    criterion_l1 = nn.L1Loss()
    
    def gradient_penalty(D, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolated = D(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp
    
    # ===============================
    # 5. حلقة التدريب المحسنة
    # ===============================
    for epoch in range(epochs):
        start_epoch = time.time()
        total_images = 0
        G.train()
        D.train()
        
        for i, (corrupted, real, mask) in enumerate(dataloader):
            corrupted, real, mask = corrupted.to(device), real.to(device), mask.to(device)
            batch_size = corrupted.size(0)
            total_images += batch_size
            
            start_batch = time.time()
            
            # تدريب الناقد
            for _ in range(5):
                fake = G(corrupted).detach()
                d_real = D(real)
                d_fake = D(fake)
                gp = gradient_penalty(D, real, fake)
                loss_D = -torch.mean(d_real) + torch.mean(d_fake) + lambda_gp * gp
    
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
    
            # تدريب المولد
            fake = G(corrupted)
            adv_loss = -torch.mean(D(fake))
            l1_loss = criterion_l1(fake, real)
            loss_G = adv_loss + lambda_l1 * l1_loss
    
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            end_batch = time.time()
    
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(dataloader)}]")
                print(f"Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} (Adv: {adv_loss.item():.4f}, L1: {l1_loss.item():.4f})")
                print(f"Batch time: {end_batch - start_batch:.2f}s\n")
        
        end_epoch = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] completed in {end_epoch - start_epoch:.2f}s")
        print(f"Avg Loss D: {loss_D.item():.4f} | Avg Loss G: {loss_G.item():.4f}\n")
        
        # حفظ النماذج بشكل دوري
        if (epoch + 1) % 20 == 0:
            torch.save(G.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(D.state_dict(), f"critic_epoch_{epoch+1}.pth")
    
    # ===============================
    # 6. دالة عرض النتائج المحسنة
    # ===============================
    def show_samples():
        G.eval()
        with torch.no_grad():
            corrupted, real, mask = next(iter(dataloader))
            corrupted = corrupted.to(device)
            fake = G(corrupted).cpu()
            
            batch_size = corrupted.size(0)
            fig, axes = plt.subplots(3, batch_size, figsize=(batch_size*2, 6))
            
            if batch_size == 1:  # التعامل مع حالة الدفعة المفردة
                axes = axes.reshape(3, 1)
            
            for i in range(batch_size):
                # الصورة التالفة
                axes[0, i].imshow((corrupted[i].cpu().permute(1,2,0) * 0.5 + 0.5).numpy())
                axes[0, i].set_title("Corrupted")
                
                # الصورة المولدة
                axes[1, i].imshow((fake[i].permute(1,2,0) * 0.5 + 0.5).numpy())
                axes[1, i].set_title("Generated")
                
                # الصورة الأصلية
                axes[2, i].imshow((real[i].permute(1,2,0) * 0.5 + 0.5).numpy())
                axes[2, i].set_title("Original")
                
                for ax in axes[:, i]:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    show_samples()

if __name__ == '__main__':
    main()