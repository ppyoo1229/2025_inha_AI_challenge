import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))         
BASE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..'))         
GT_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
CKPT_DIR   = os.path.join(PROJECT_DIR, 'checkpoints')
SUBMIT_DIR = os.path.join(PROJECT_DIR, 'submissions')
for d in [OUTPUT_DIR, SUBMIT_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

NUM_EPOCHS  = 5      
BATCH_SIZE  = 32
LR          = 1e-4
IMG_SIZE    = 256
NUM_WORKERS = 4
SEED        = 42
PATIENCE    = 3

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터셋
class ColorizationDataset(Dataset):
    def __init__(self, df, root_dir, img_size=IMG_SIZE, is_test=False):
        self.df       = df
        self.root_dir = root_dir
        self.is_test  = is_test
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row['input_img_path'])).convert('RGB')
        img = self.transform(img)
        gray = transforms.Grayscale(3)(img)
        cap  = row['caption']
        if not self.is_test:
            tgt = Image.open(os.path.join(self.root_dir, row['gt_img_path'])).convert('RGB')
            tgt = self.transform(tgt)
            return gray, cap, tgt
        else:
            return gray, cap, row['ID']

# CrossAttention
class CrossAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, heads=4):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, img_dim)
        self.txt_proj = nn.Linear(txt_dim, img_dim)
        self.attn     = nn.MultiheadAttention(img_dim, heads, batch_first=True)
    def forward(self, img_feat, txt_feat):
        B, C, H, W = img_feat.shape
        flat = img_feat.permute(0,2,3,1).reshape(B, H*W, C)
        q    = self.img_proj(flat)
        kv   = self.txt_proj(txt_feat)
        out, _ = self.attn(q, kv, kv)
        return out.reshape(B, H, W, C).permute(0,3,1,2)

class UNetDown(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

class UNetUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
    def forward(self, x, skip=None):
        x = self.up(x)
        x = self.bn(x)
        return self.relu(x)

class CLIP_CrossAttn_UNet(nn.Module):
    def __init__(self, img_size=IMG_SIZE, txt_dim=512):
        super().__init__()
        self.img_size = img_size
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.down1 = UNetDown(3,   64)
        self.down2 = UNetDown(64,  128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512,512,3,1,1), nn.ReLU(True))
        self.up3   = UNetUp(512, 256); self.attn3 = CrossAttention(256, txt_dim)
        self.up2   = UNetUp(256, 128); self.attn2 = CrossAttention(128, txt_dim)
        self.up1   = UNetUp(128,  64); self.attn1 = CrossAttention(64,  txt_dim)
        self.final = nn.Conv2d(64,3,1)
    def encode_text(self, caps, device):
        toks = self.tokenizer(caps, padding=True, truncation=True,
                              max_length=77, return_tensors="pt")
        toks = {k:v.to(device) for k,v in toks.items()}
        with torch.no_grad():
            return self.text_model(**toks).last_hidden_state
    def forward(self, gray, caps):
        device = gray.device
        txt_emb = self.encode_text(caps, device)
        x1 = self.down1(gray)   
        x2 = self.down2(x1)     
        x3 = self.down3(x2)     
        x4 = self.down4(x3)    
        b  = self.bottleneck(x4)
        x = self.up3(b);   x = self.attn3(x, txt_emb)
        x = self.up2(x);   x = self.attn2(x, txt_emb)
        x = self.up1(x);   x = self.attn1(x, txt_emb)
        out = torch.sigmoid(self.final(x)) 
        return F.interpolate(out,
                             size=(self.img_size, self.img_size),
                             mode='bilinear',
                             align_corners=False)

# 학습/검증
def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 분할
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = np.split(df.sample(frac=1, random_state=SEED), [int(.9*len(df))])
    train_ds = ColorizationDataset(train_df, os.path.join(BASE_DIR, "train"), IMG_SIZE)
    val_ds   = ColorizationDataset(val_df,   os.path.join(BASE_DIR, "train"), IMG_SIZE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = CLIP_CrossAttn_UNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    log_fp = open(os.path.join(OUTPUT_DIR, "train_log.txt"), "w")
    best_val_loss = float('inf')
    counter = 0
    train_loss_list, val_loss_list = [], []

    try:
        for ep in range(1, NUM_EPOCHS+1):
            # Train 
            model.train()
            total = 0
            pbar = tqdm(train_dl, desc=f"Epoch {ep}")
            for i, (gray, caps, tgt) in enumerate(pbar, 1):
                gray, tgt = gray.to(device), tgt.to(device)
                out = model(gray, list(caps))
                loss = loss_fn(out, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
                msg = f"Epoch {ep} Step {i}/{len(train_dl)} Loss: {loss:.4f}"
                pbar.set_postfix(loss=loss.item())
                tqdm.write(msg)
                print(msg, file=log_fp, flush=True)
            avg = total / len(train_dl)
            train_loss_list.append(avg)

            # Validation
            model.eval()
            val_total = 0
            with torch.no_grad():
                for gray, caps, tgt in val_dl:
                    gray, tgt = gray.to(device), tgt.to(device)
                    out = model(gray, list(caps))
                    val_loss = loss_fn(out, tgt)
                    val_total += val_loss.item()
            avg_val_loss = val_total / len(val_dl)
            val_loss_list.append(avg_val_loss)
            tqdm.write(f"[E{ep}] Val loss: {avg_val_loss:.4f}")
            print(f"[E{ep}] Val loss: {avg_val_loss:.4f}", file=log_fp, flush=True)

            # Best model save
            ckpt_path = os.path.join(CKPT_DIR, f"model_epoch{ep}.pth")
            torch.save(model.state_dict(), ckpt_path)
            if avg_val_loss < best_val_loss:
                best_path = os.path.join(CKPT_DIR, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                print(f'No improvement for {counter} epoch(s)')
                if counter >= PATIENCE:
                    print(f'Early stopping at epoch {ep}')
                    break

        # Loss curve plot
        plt.plot(train_loss_list, label='train_loss')
        plt.plot(val_loss_list, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))

        print("학습 끝")
    finally:
        log_fp.close()

if __name__ == "__main__":
    main()