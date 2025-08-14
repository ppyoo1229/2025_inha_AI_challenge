import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

gray_dir = "/home/guest01/colorize/train/input_image"
color_dir = "/home/guest01/colorize/train/gt_image"
img_list = sorted(os.listdir(gray_dir))   
SUBMISSION_DIR = "/home/guest01/colorize/CLIPUNetGAN"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 경로 미리 정의
gen_best_path    = os.path.join(SUBMISSION_DIR, 'gen_best.pth')
disc_best_path   = os.path.join(SUBMISSION_DIR, 'disc_best.pth')
gen_latest_path  = os.path.join(SUBMISSION_DIR, 'gen_latest.pth')
disc_latest_path = os.path.join(SUBMISSION_DIR, 'disc_latest.pth')
latest_epoch_path= os.path.join(SUBMISSION_DIR, 'latest_epoch.txt')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_EPOCHS  = 20
BATCH_SIZE  = 32
LR          = 2e-4
IMG_SIZE    = 512
NUM_WORKERS = 8


# ---- 데이터셋 ----
class MyColorizationDataset(Dataset):
    def __init__(self, gray_dir, color_dir, img_list, img_size=512):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        fname = self.img_list[idx]
        gray = Image.open(os.path.join(self.gray_dir, fname)).convert('L')
        color = Image.open(os.path.join(self.color_dir, fname)).convert('RGB')
        gray = self.transform(gray).repeat(3,1,1)
        color = self.transform(color)
        return gray, color

# ---- Generator: UNet ----
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64):
        super().__init__()
        # 다운샘플링
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(nf, nf*2, 4, 2, 1), nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(nf*2, nf*4, 4, 2, 1), nn.BatchNorm2d(nf*4), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(nf*4, nf*8, 4, 2, 1), nn.BatchNorm2d(nf*8), nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(nn.Conv2d(nf*8, nf*8, 4, 2, 1), nn.BatchNorm2d(nf*8), nn.LeakyReLU(0.2))
        # 업샘플링
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(nf*8, nf*8, 4, 2, 1), nn.BatchNorm2d(nf*8), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(nf*16, nf*4, 4, 2, 1), nn.BatchNorm2d(nf*4), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(nf*8, nf*2, 4, 2, 1), nn.BatchNorm2d(nf*2), nn.ReLU())
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(nf*4, nf, 4, 2, 1), nn.BatchNorm2d(nf), nn.ReLU())
        self.dec5 = nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], 1))
        d3 = self.dec3(torch.cat([d2, e3], 1))
        d4 = self.dec4(torch.cat([d3, e2], 1))
        d5 = self.dec5(torch.cat([d4, e1], 1))
        return self.tanh(d5)

# ---- Discriminator: PatchGAN ----
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, nf=64):
        super().__init__()
        # 입력: (gray+color) concat → in_ch=6
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf*2, 4, 2, 1), nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2),
            nn.Conv2d(nf*2, nf*4, 4, 2, 1), nn.BatchNorm2d(nf*4), nn.LeakyReLU(0.2),
            nn.Conv2d(nf*4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.main(x)

def save_sample(fake_color, epoch, out_dir=SUBMISSION_DIR):
    # fake_color: [B,3,H,W], 저장은 한 장만 예시로
    img = fake_color[0].detach().cpu().clamp(-1, 1)*0.5 + 0.5  # [-1,1]→[0,1] 변환
    vutils.save_image(img, os.path.join(out_dir, f'epoch_{epoch}_sample.png'))

# ---- train/val split ----
train_list, val_list = train_test_split(img_list, test_size=0.1, random_state=SEED)
train_dataset = MyColorizationDataset(gray_dir, color_dir, train_list, img_size=IMG_SIZE)
val_dataset   = MyColorizationDataset(gray_dir, color_dir, val_list, img_size=IMG_SIZE)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ---- Model, Optimizer, Loss ----
G = UNetGenerator().to('cuda')
D = PatchDiscriminator().to('cuda')
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# ---- Resume ----
start_epoch = 1
# 우선순위: latest > best > 새로 학습
if os.path.exists(gen_latest_path) and os.path.exists(disc_latest_path) and os.path.exists(latest_epoch_path):
    print("latest weight에서 resume")
    G.load_state_dict(torch.load(gen_latest_path))
    D.load_state_dict(torch.load(disc_latest_path))
    with open(latest_epoch_path, 'r') as f:
        start_epoch = int(f.read().strip()) + 1
elif os.path.exists(gen_best_path) and os.path.exists(disc_best_path):
    print("best weight에서 resume")
    G.load_state_dict(torch.load(gen_best_path))
    D.load_state_dict(torch.load(disc_best_path))
    # best.pth가 몇 번째 에폭인지 로그에서 확인해서 start_epoch 정함
    start_epoch = 11   # best.pth가 10에서 저장
else:
    print("새로 학습 시작!")
    start_epoch = 1

best_loss = float('inf')

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    G.train()
    D.train()
    epoch_loss_G, epoch_loss_D = 0, 0

    # --- train_loader ---
    pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{NUM_EPOCHS}")
    for step, (gray, color) in enumerate(pbar):
        gray, color = gray.cuda(), color.cuda()
        fake_color = G(gray)
        real_pair = torch.cat([gray, color], 1)
        fake_pair = torch.cat([gray, fake_color.detach()], 1)
        D_real = D(real_pair)
        D_fake = D(fake_pair)
        real_label = torch.ones_like(D_real)
        fake_label = torch.zeros_like(D_fake)
        loss_D = (criterion_GAN(D_real, real_label) + criterion_GAN(D_fake, fake_label)) * 0.5
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        fake_pair = torch.cat([gray, fake_color], 1)
        D_fake = D(fake_pair)
        loss_G_GAN = criterion_GAN(D_fake, real_label)
        loss_G_L1 = criterion_L1(fake_color, color)
        loss_G = loss_G_GAN + 100 * loss_G_L1
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()

        pbar.set_postfix({
            "loss_G": f"{loss_G.item():.4f}",
            "loss_D": f"{loss_D.item():.4f}"
        })

    avg_loss_G = epoch_loss_G / len(train_loader)
    avg_loss_D = epoch_loss_D / len(train_loader)
    print(f'[Epoch {epoch}] loss_G: {avg_loss_G:.4f}, loss_D: {avg_loss_D:.4f}')

    if (epoch % 10 == 0) or (epoch < 10):
        save_sample(fake_color, epoch)

    # --- Validation Step ---
    G.eval()
    val_loss = 0.0
    val_pbar = tqdm(val_loader, desc=f"[Val] Epoch {epoch}")
    with torch.no_grad():
        for step, (gray, color) in enumerate(val_pbar):
            gray, color = gray.cuda(), color.cuda()
            fake_color = G(gray)
            loss = criterion_L1(fake_color, color)
            val_loss += loss.item()
            val_pbar.set_postfix({"val_L1_loss": f"{loss.item():.4f}"})
    val_loss /= len(val_loader)
    print(f'[Epoch {epoch}] val_L1_loss: {val_loss:.4f}')
    G.train()

    # --- latest weight & epoch 저장 (항상 덮어쓰기) ---
    torch.save(G.state_dict(), gen_latest_path)
    torch.save(D.state_dict(), disc_latest_path)
    with open(latest_epoch_path, 'w') as f:
        f.write(str(epoch))

    # --- best model 저장 (best loss 경신시만) ---
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(G.state_dict(), gen_best_path)
        torch.save(D.state_dict(), disc_best_path)
        print(f"[Epoch {epoch}] Best model updated and previous best deleted.")

print("학습 종료! best.pth / latest.pth / latest_epoch.txt 저장")