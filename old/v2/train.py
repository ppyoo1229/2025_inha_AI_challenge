# CLIP Text 임베딩 + CLIP ViT-L/14 이미지 임베딩 → U-Net → PatchGAN
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import deque
import torch.nn.functional as F 

from transformers import (
    CLIPTokenizer, CLIPTextModel,
    CLIPVisionModel, CLIPProcessor
)


os.chdir('/home/guest01/colorize/')
ROOT_PATH = '/home/guest01/colorize/'
gray_dir = os.path.join(ROOT_PATH, "train/input_image")
color_dir = os.path.join(ROOT_PATH, "train/gt_image")
CSV_PATH = os.path.join(ROOT_PATH, "train.csv")
SUBMISSION_DIR = "./UNetGAN"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
gen_best_path    = os.path.join(SUBMISSION_DIR, 'gen_best.pth')
disc_best_path   = os.path.join(SUBMISSION_DIR, 'disc_best.pth')
gen_latest_path  = os.path.join(SUBMISSION_DIR, 'gen_latest.pth')
disc_latest_path = os.path.join(SUBMISSION_DIR, 'disc_latest.pth')
latest_epoch_path= os.path.join(SUBMISSION_DIR, 'latest_epoch.txt')
best_loss_path   = os.path.join(SUBMISSION_DIR, 'best_loss.txt')

IMG_SIZE = 512
BATCH_SIZE = 6
NUM_EPOCHS = 20
LR = 2e-4
NUM_WORKERS = 8
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- 캡션 딕셔너리 생성 (csv 기반) ----
df = pd.read_csv(CSV_PATH)
# 모든 이미지 파일에 대해: {input_img_path: caption} 매핑
caption_dict = {row['input_img_path']: row['caption'] for _, row in df.iterrows()}
img_list = list(caption_dict.keys())

# ----- 데이터셋 -----
class MyColorizationDataset(Dataset):
    def __init__(self, df, img_size=512):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gray_path  = os.path.join(ROOT_PATH, 'train', row['input_img_path'])
        color_path = os.path.join(ROOT_PATH, 'train', row['gt_img_path'])
        gray  = Image.open(gray_path).convert('L')
        color = Image.open(color_path).convert('RGB')
        gray = self.transform(gray).repeat(3,1,1)
        color = self.transform(color)
        caption = row['caption']
        fname = row['input_img_path']
        return gray, color, caption, fname

# ----- 임베딩 추출 함수 -----
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()

@torch.no_grad()
def get_img_emb(gray_tensor):
    imgs_pil = [transforms.ToPILImage()(img) for img in gray_tensor.cpu()]
    inputs = clip_processor(images=imgs_pil, return_tensors="pt").to(DEVICE)
    out = clip_img_encoder(**inputs).last_hidden_state[:,0]  # [B, 1024] CLS 토큰
    return out

@torch.no_grad()
def get_text_emb(captions):
    toks = clip_tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
    out = clip_text_encoder(**toks).last_hidden_state[:,0]  # [B, 1024] CLS 토큰
    return out

# ----- U-Net/Generator -----
class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            norm_layer(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # 변경: ConvTranspose2d 대신 Upsample + Conv2d 사용
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 정확히 2배 업샘플링
            nn.Conv2d(in_c, out_c, 3, 1, 1), # 일반 컨볼루션 (패딩 1로 크기 유지)
            norm_layer(out_c),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.up_block(x)

class UNetCondition(nn.Module):
    def __init__(self, emb_dim=1792):
        super().__init__()
        # norm_layer로 InstanceNorm2d 전달
        self.down1 = UNetDown(3, 64, norm_layer=nn.InstanceNorm2d)    # 512->256
        self.down2 = UNetDown(64, 128, norm_layer=nn.InstanceNorm2d)  # 256->128
        self.down3 = UNetDown(128, 256, norm_layer=nn.InstanceNorm2d) # 128->64
        self.down4 = UNetDown(256, 512, norm_layer=nn.InstanceNorm2d) # 64->32
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512,512,3,1,1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.emb_fc = None
        # norm_layer로 InstanceNorm2d 전달
        self.up4   = UNetUp(512, 256, norm_layer=nn.InstanceNorm2d)
        self.up3   = UNetUp(256, 128, norm_layer=nn.InstanceNorm2d)
        self.up2   = UNetUp(128, 64, norm_layer=nn.InstanceNorm2d)
        self.up1   = UNetUp(64, 32, norm_layer=nn.InstanceNorm2d)
        self.final = nn.Conv2d(32, 3, 1)
    def forward(self, gray, text_emb, img_emb):
        emb_cat = torch.cat([text_emb, img_emb], dim=1)
        x1 = self.down1(gray)  # 512->256
        x2 = self.down2(x1)    # 256->128
        x3 = self.down3(x2)    # 128->64
        x4 = self.down4(x3)    # 64->32
        b  = self.bottleneck(x4) # 32->32
        B, C, H, W = b.shape
        if self.emb_fc is None or self.emb_fc.out_features != C*H*W:
            self.emb_fc = nn.Linear(emb_cat.shape[1], C*H*W).to(emb_cat.device)
        emb_proj = self.emb_fc(emb_cat)
        emb_proj = emb_proj.view(B, C, H, W)
        b = b + emb_proj
        x = self.up4(b)       # 32→64
        x = self.up3(x)       # 64→128
        x = self.up2(x)       # 128→256
        x = self.up1(x)       # 256→512
        out = torch.sigmoid(self.final(x))
        
        # --- output shape 보정 --- 
        if out.shape[-1] != 512 or out.shape[-2] != 512:
            out = F.interpolate(out, size=(512,512), mode='bilinear', align_corners=False)
        return out

# ----- PatchGAN Discriminator -----
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, 4, 2, 1), nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, 4, 2, 1), nn.BatchNorm2d(nf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.main(x)

# ----- Train/Val Split, Loader -----
train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)
train_dataset = MyColorizationDataset(train_df, img_size=IMG_SIZE)
val_dataset   = MyColorizationDataset(val_df, img_size=IMG_SIZE)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ----- Generator 선언 (이제부터 G 정의됨!) -----
G = UNetCondition().to(DEVICE)

# --- 1. Pretrained weight 불러오기 -----
try:
    from diffusers import UNet2DConditionModel
    sd_unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )
    G_state_dict = G.state_dict()
    sd_unet_state_dict = sd_unet.state_dict()
    
    pretrained_dict = {k: v for k, v in sd_unet_state_dict.items() if k in G_state_dict and G_state_dict[k].shape == v.shape}
    G_state_dict.update(pretrained_dict)
    G.load_state_dict(G_state_dict, strict=False)
    
    print("Stable Diffusion U-Net pretrained weight 적용 완료!")
except Exception as e:
    print("Pretrained weight 로딩 실패 (무시하고 진행):", e)

# --- 2. Freeze Bottleneck+Decoder+emb_fc만 제외하고 모두 freeze -----
for name, param in G.named_parameters():
    if name.startswith("down"):
        param.requires_grad = False
    elif name.startswith("bottleneck") or name.startswith("up") or name.startswith("emb_fc"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# ----- Discriminator -----
D = PatchDiscriminator().to(DEVICE)
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
opt_G = torch.optim.Adam(
    filter(lambda p: p.requires_grad, G.parameters()), lr=LR, betas=(0.5, 0.999)
)
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

def save_sample(fake_color, epoch, out_dir=SUBMISSION_DIR):
    img = fake_color[0].detach().cpu().clamp(0, 1)
    vutils.save_image(img, os.path.join(out_dir, f'epoch_{epoch}_sample.png'))

# ---- Resume ----
start_epoch = 1
best_loss = float('inf')

# 우선순위: latest > best > 새로 학습
if all(os.path.exists(p) for p in [gen_latest_path, disc_latest_path, latest_epoch_path, best_loss_path]):
    print("latest weight에서 resume")
    G.load_state_dict(torch.load(gen_latest_path))
    D.load_state_dict(torch.load(disc_latest_path))
    with open(latest_epoch_path, 'r') as f:
        start_epoch = int(f.read().strip()) + 1
    with open(best_loss_path, 'r') as f:
        best_loss = float(f.read().strip())
elif all(os.path.exists(p) for p in [gen_best_path, disc_best_path, best_loss_path]):
    print("best weight에서 resume")
    G.load_state_dict(torch.load(gen_best_path))
    D.load_state_dict(torch.load(disc_best_path))
    with open(best_loss_path, 'r') as f:
        best_loss = float(f.read().strip())
    start_epoch = 1 
else:
    print("새로 학습 시작!")
    start_epoch = 1
    best_loss = float('inf')

# ----- 학습 루프 -----
scaler_G = GradScaler()
scaler_D = GradScaler()

patience = 5        # early stopping patience (epoch)
no_improve = 0      # best loss 미갱신 카운트

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    G.train(); D.train()
    epoch_loss_G, epoch_loss_D = 0, 0
    recent_loss_G = deque(maxlen=100)
    recent_loss_D = deque(maxlen=100)

    pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{NUM_EPOCHS}", ncols=110)
    for step, (gray, color, captions, fnames) in enumerate(pbar):
        gray, color = gray.to(DEVICE), color.to(DEVICE)
        # ----- D update -----
        with autocast():
            text_emb = get_text_emb(captions)
            img_emb  = get_img_emb(gray)
            fake_color = G(gray, text_emb, img_emb)
            real_pair = torch.cat([gray, color], 1)
            fake_pair = torch.cat([gray, fake_color.detach()], 1)
            D_real = D(real_pair)
            D_fake = D(fake_pair)
            real_label = torch.ones_like(D_real)
            fake_label = torch.zeros_like(D_fake)
            loss_D = (criterion_GAN(D_real, real_label) + criterion_GAN(D_fake, fake_label)) * 0.5
        opt_D.zero_grad()
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # ----- G update -----
        with autocast():
            fake_pair = torch.cat([gray, fake_color], 1)
            D_fake = D(fake_pair)
            loss_G_GAN = criterion_GAN(D_fake, real_label)
            loss_G_L1 = criterion_L1(fake_color, color)
            loss_G = loss_G_GAN + 100 * loss_G_L1
        opt_G.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        recent_loss_G.append(loss_G.item())
        recent_loss_D.append(loss_D.item())

        pbar.set_postfix({
            "step": step,
            "loss_G": f"{np.mean(recent_loss_G):.4f}",
            "loss_D": f"{np.mean(recent_loss_D):.4f}",
            "best": f"{best_loss:.4f}"
        })

    print(f'[Epoch {epoch}] loss_G: {epoch_loss_G/len(train_loader):.4f}, loss_D: {epoch_loss_D/len(train_loader):.4f}')
    if (epoch % 3 == 0):
        save_sample(fake_color, epoch)

    # ----- Validation -----
    G.eval()
    val_loss = 0.0
    with torch.no_grad():
        for gray, color, captions, fnames in val_loader:
            gray, color = gray.to(DEVICE), color.to(DEVICE)
            text_emb = get_text_emb(captions)
            img_emb  = get_img_emb(gray)
            fake_color = G(gray, text_emb, img_emb)
            loss = criterion_L1(fake_color, color)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'[Epoch {epoch}] val_L1_loss: {val_loss:.4f}')
    
    # latest 저장(항상 덮어쓰기)
    torch.save(G.state_dict(), gen_latest_path)
    torch.save(D.state_dict(), disc_latest_path)
    with open(latest_epoch_path, 'w') as f:
        f.write(str(epoch))
    with open(best_loss_path, 'w') as f:
        f.write(str(best_loss))

    # best 저장(최적시만)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(G.state_dict(), gen_best_path)
        torch.save(D.state_dict(), disc_best_path)
        with open(best_loss_path, 'w') as f:
            f.write(str(best_loss))
        print(f"[Epoch {epoch}] Best model updated!")
        no_improve = 0
    else:
        no_improve += 1

    # ----- Early Stopping -----
    if no_improve >= patience:
        print(f"Early stopping: {patience} epochs no improvement. Training finished.")
        break

print("학습 종료! best.pth 저장")