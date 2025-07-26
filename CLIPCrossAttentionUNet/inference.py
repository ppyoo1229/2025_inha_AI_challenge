import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import open_clip
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_DIR = os.path.join(BASE_DIR, 'CLIPCrossAttention-UNet')    
TEST_CSV = os.path.join(BASE_DIR, 'test.csv')
TEST_IMG_DIR = os.path.join(BASE_DIR, 'test', 'input_image')
MODEL_PATH = os.path.join(PROJECT_DIR, 'checkpoints', 'best_model.pth')
SUBMISSION_DIR = os.path.join(PROJECT_DIR, 'submission')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
sys.path.append(PROJECT_DIR)
from train import CLIP_CrossAttn_UNet

# 추론용 ColorizationDataset 정의 
class ColorizationDataset(Dataset):
    def __init__(self, df, root_dir, img_size=256):
        self.df = df
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) 
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row['input_img_path'])).convert('RGB')
        img = self.transform(img)
        gray = transforms.Grayscale(3)(img)  # 학습 파이프와 동일하게 그레이스케일 (3채널)
        cap  = row['caption']
        return gray, cap, row['ID']
    
# 데이터 로딩
test_df = pd.read_csv(TEST_CSV)
test_set = ColorizationDataset(test_df, TEST_IMG_DIR, img_size=256)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIP_CrossAttn_UNet(img_size=256).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()   

# 이미지 저장
id_list = []
for gray, caption, img_id in tqdm(test_loader):
    gray = gray.to(device)
    with torch.no_grad():
        output = model(gray, list(caption))
    out_img = (output[0].detach().cpu().clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
    Image.fromarray(out_img).save(os.path.join(SUBMISSION_DIR, f'{img_id[0]}.png'))
    id_list.append(img_id[0])
print('제출 이미지 저장 완료')

# CLIP 임베딩 추출/ CSV 저장 
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
clip_model.to(device)

feat_imgs = []
for img_id in tqdm(id_list):
    img = Image.open(os.path.join(SUBMISSION_DIR, f'{img_id}.png'))
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_img = clip_model.encode_image(img_tensor)
        feat_img /= feat_img.norm(dim=-1, keepdim=True)
    feat_imgs.append(feat_img.cpu().numpy().reshape(-1))
feat_imgs = np.array(feat_imgs)

vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
feat_submission.insert(0, 'ID', id_list)
feat_submission.to_csv(os.path.join(SUBMISSION_DIR, 'embed_submission.csv'), index=False)
print('제출용 임베딩 저장 완료')

#---------------------
import cv2
import numpy as np
import pandas as pd
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

# 경로 세팅
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_DIR = os.path.join(BASE_DIR, 'CLIPCrossAttention-UNet') 
SUBMISSION_DIR = os.path.join(PROJECT_DIR, 'submission')
SCORE_CSV = os.path.join(PROJECT_DIR, 'score.csv')
TEST_CSV = os.path.join(BASE_DIR, 'test.csv')
gt_folder = os.path.join(BASE_DIR, 'test', 'gt_image')
pred_folder = SUBMISSION_DIR

test_df = pd.read_csv(TEST_CSV)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
clip_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model.to(device)

hsv_scores = []
clip_scores = []

def hsv_similarity(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    sim = 0
    for i in range(3):
        h1 = cv2.calcHist([hsv1], [i], None, [256], [0,256])
        h2 = cv2.calcHist([hsv2], [i], None, [256], [0,256])
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()
        sim += cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return sim / 3.0

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_id = row['ID']
    caption = row['caption']

    # HSV
    gt_img = cv2.cvtColor(cv2.imread(f"{gt_folder}/{img_id}.png"), cv2.COLOR_BGR2RGB)
    pred_img = cv2.cvtColor(cv2.imread(f"{pred_folder}/{img_id}.png"), cv2.COLOR_BGR2RGB)
    hsv_score = hsv_similarity(gt_img, pred_img)
    hsv_scores.append(hsv_score)

    # CLIP
    img = Image.open(f"{pred_folder}/{img_id}.png")
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt = open_clip.tokenize([caption]).to(device)
    with torch.no_grad():
        txt_feat = clip_model.encode_text(txt)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    clip_score = (img_feat @ txt_feat.T).item()
    clip_scores.append(clip_score)

# DataFrame 저장
score_df = test_df.copy()
score_df['HSV_Similarity'] = hsv_scores
score_df['CLIP_Score'] = clip_scores
score_df.to_csv(SCORE_CSV, index=False)

print(f'평균 HSV_Similarity: {np.mean(hsv_scores):.4f}')
print(f'평균 CLIP_Score: {np.mean(clip_scores):.4f}')
print(f'score.csv 저장 완료!')