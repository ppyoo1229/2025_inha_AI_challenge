import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch.nn as nn
import numpy as np
import random 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))         
BASE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..'))         
GT_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

# 결과물 
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
CKPT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
SUBMIT_DIR = os.path.join(PROJECT_DIR, 'submissions')
for d in [OUTPUT_DIR, SUBMIT_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# Config
CFG = {
    'SEED': 42,
    'MAX_DATA': None,          
    'BATCH_SIZE': 32,            
    'LR': 3e-4,                  
    'NUM_EPOCHS': 5,           
    'NUM_WORKERS': 4,          
    'IMG_SIZE': 256,
    'INPUT_DIR': INPUT_DIR,
    'GT_DIR': GT_DIR,
    'TRAIN_CSV': TRAIN_CSV,
    'OUTPUT_DIR': OUTPUT_DIR,
    'CKPT_DIR': CKPT_DIR,
    'SUBMIT_DIR': SUBMIT_DIR,
}

# 데이터 로드 및 split
train_csv = CFG['TRAIN_CSV']
input_dir = CFG['INPUT_DIR']
gt_dir = CFG['GT_DIR']
df = pd.read_csv(train_csv)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=CFG['SEED'])
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)


# ColorizationDataset
class ColorizationDataset(Dataset):
    def __init__(self, csv_path, input_dir, gt_dir, img_size=256, max_data=None, is_val=False):
        self.df = pd.read_csv(csv_path)
        if max_data:
            self.df = self.df.sample(n=max_data, random_state=CFG['SEED']).reset_index(drop=True)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.is_val = is_val
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_img = Image.open(os.path.join(self.input_dir, row['input_img_path'])).convert('RGB')
        gt_img    = Image.open(os.path.join(self.gt_dir, row['gt_img_path'])).convert('RGB')
        input_img = self.transform(input_img)
        gt_img    = self.transform(gt_img)
        cap = row['caption']
        return input_img, cap, gt_img

# DataLoader 준비
train_dataset = ColorizationDataset('train_split.csv', input_dir, gt_dir, max_data=CFG['MAX_DATA'])
val_dataset = ColorizationDataset('val_split.csv', input_dir, gt_dir, max_data=None, is_val=True)
train_loader = DataLoader(
    train_dataset, 
    batch_size=CFG['BATCH_SIZE'], 
    shuffle=True,
    num_workers=CFG['NUM_WORKERS'], 
    pin_memory=True 
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CFG['BATCH_SIZE'], 
    shuffle=False, 
    num_workers=CFG['NUM_WORKERS'], 
    pin_memory=True
)

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
    def __init__(self, img_size=256, txt_dim=512):
        super().__init__()
        self.img_size = img_size
        # CLIP text
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # Encoder
        self.down1 = UNetDown(3,   64)
        self.down2 = UNetDown(64,  128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512,512,3,1,1), nn.ReLU(True))
        # Decoder + CrossAttention
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

def main():
    seed_everything(CFG['SEED'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP_CrossAttn_UNet(img_size=CFG['IMG_SIZE']).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
    loss_fn = torch.nn.L1Loss()

    patience = 3
    counter = 0
    best_val_loss = float('inf')
    bs = CFG['BATCH_SIZE']
    lr = CFG['LR']
    num_epochs = CFG['NUM_EPOCHS']

    log_fp = open(os.path.join(OUTPUT_DIR, "train_log.txt"), "w")

    try:
        for ep in range(1, num_epochs + 1):
            model.train()
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {ep}")
            for i, (inp, caps, tgt) in enumerate(pbar, 1):
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp, list(caps))
                loss = loss_fn(out, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()

                msg = f"Epoch {ep} Step {i}/{len(train_loader)} Loss: {loss:.4f}"
                pbar.set_postfix(loss=loss.item())
                tqdm.write(msg)
                print(msg, file=log_fp, flush=True)

            avg = total / len(train_loader)

            # 검증 루프
            model.eval()
            val_total = 0
            with torch.no_grad():
                for val_in, val_cap, val_gt in val_loader:
                    val_in, val_gt = val_in.to(device), val_gt.to(device)
                    val_out = model(val_in, list(val_cap))
                    val_loss = loss_fn(val_out, val_gt)
                    val_total += val_loss.item()
            avg_val_loss = val_total / len(val_loader)
            print(f'[E{ep}] Val loss: {avg_val_loss:.4f}')
            print(f'[E{ep}] Val loss: {avg_val_loss:.4f}', file=log_fp, flush=True)

            # Best 저장 + Early Stopping
            if avg_val_loss < best_val_loss:
                filename = f'best_bs{bs}_lr{lr}_ep{ep}_loss{avg_val_loss:.3f}.pth'
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, filename))
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                print(f'No improvement for {counter} epoch(s)')
                if counter >= patience:
                    print(f'Early stopping triggered at epoch {ep}')
                    break

        print("학습 끝")
    finally:
        log_fp.close()

if __name__ == "__main__":
    main()