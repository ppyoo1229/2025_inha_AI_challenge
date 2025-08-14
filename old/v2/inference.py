import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import zipfile

# --- open_clip import ---
import open_clip # open_clip 라이브러리 임포트
# --- scipy.spatial.distance.cosine for HSV similarity (if using ensemble) ---
from scipy.spatial.distance import cosine # For the ensemble logic, though not active by default

# --- Configuration (from CFG dict you provided) ---
# It's good practice to centralize configuration
CFG = {
    'SUB_DIR': "./submission", # 추론 결과물 디렉토리
    'SEED': 42,
    'USE_ENSEMBLE': False, # 초기에는 False로 설정하여 프롬프트 강화 없이 진행
    'NUM_ENSEMBLE_SAMPLES': 2, # USE_ENSEMBLE이 True일 때 사용
    'USE_CLIP_SCORING': True, # 앙상블 시 CLIP 점수 사용 여부
    'HSV_CLIP_RATIO': (0.6, 0.4), # 앙상블 시 HSV와 CLIP 점수 비율
}

# --- Path Setup ---
ROOT_PATH = '/home/guest01/colorize/'
TEST_CSV_PATH = os.path.join(ROOT_PATH, "train.csv") # IMPORTANT: Change to actual test CSV path
TEST_IMAGE_DIR = os.path.join(ROOT_PATH, "train/input_image") # IMPORTANT: Change to actual test image directory

GEN_BEST_PATH = os.path.join("./UNetGAN", 'gen_best.pth')
ZIP_PATH = './submission.zip'

# --- Hyperparameters & Device ---
IMG_SIZE = 512
BATCH_SIZE = 6
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Seed Everything ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG['SEED'])

# --- Dataset for Inference ---
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, img_size=512):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # img_id should be like "TEST_001" for output filename
        img_id_base = row['input_img_path'].split('/')[-1].replace('.png', '')
        gray_path = os.path.join(self.img_dir, row['input_img_path'].split('/')[-1])
        
        gray = Image.open(gray_path).convert('L')
        gray = self.transform(gray).repeat(3, 1, 1) # Repeat for 3 channels for UNet input
        caption = row['caption']
        
        return gray, caption, img_id_base # Return base ID, not '.png' suffix here

# --- CLIP Model for Inference (Used by G) ---
# For UNetCondition's CLIP embeddings during generation
# NOTE: This CLIP is used for generating `text_emb` and `img_emb` for G.
# The `ViT-L-14` below will be used for the final submission embedding.
# It's recommended to use the same CLIP model here if your UNet was trained with it.
# Let's assume your UNet was trained with `openai/clip-vit-large-patch14` as before.
# If you trained with `open_clip ViT-L-14`, then use that here too.
# For simplicity and consistency with previous answers, let's keep the `transformers` CLIP for UNet input.
# If your UNet was trained with open_clip ViT-L-14 as the conditioning, please change this section.
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor
clip_processor_for_unet = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_img_encoder_for_unet = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
clip_tokenizer_for_unet = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_text_encoder_for_unet = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()

@torch.no_grad()
def get_img_emb_for_unet(gray_tensor):
    imgs_pil = [transforms.ToPILImage()(img) for img in gray_tensor.cpu()]
    inputs = clip_processor_for_unet(images=imgs_pil, return_tensors="pt").to(DEVICE)
    out = clip_img_encoder_for_unet(**inputs).last_hidden_state[:,0] # [B, 1024]
    return out

@torch.no_grad()
def get_text_emb_for_unet(captions):
    toks = clip_tokenizer_for_unet(captions, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
    out = clip_text_encoder_for_unet(**toks).last_hidden_state[:,0] # [B, 1024]
    return out

# --- UNet/Generator (No changes from last working version) ---
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
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            norm_layer(out_c),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.up_block(x)

class UNetCondition(nn.Module):
    # emb_dim is now 1024 (from transformers CLIP) + 1024 (from transformers CLIP) = 2048
    def __init__(self, emb_dim=2048): 
        super().__init__()
        self.down1 = UNetDown(3, 64, norm_layer=nn.InstanceNorm2d)
        self.down2 = UNetDown(64, 128, norm_layer=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_layer=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_layer=nn.InstanceNorm2d)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512,512,3,1,1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.emb_fc = None
        self.up4   = UNetUp(512, 256, norm_layer=nn.InstanceNorm2d)
        self.up3   = UNetUp(256, 128, norm_layer=nn.InstanceNorm2d)
        self.up2   = UNetUp(128, 64, norm_layer=nn.InstanceNorm2d)
        self.up1   = UNetUp(64, 32, norm_layer=nn.InstanceNorm2d)
        self.final = nn.Conv2d(32, 3, 1)
    def forward(self, gray, text_emb, img_emb):
        emb_cat = torch.cat([text_emb, img_emb], dim=1) 
        x1 = self.down1(gray)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        b  = self.bottleneck(x4)
        B, C, H, W = b.shape
        if self.emb_fc is None or self.emb_fc.in_features != emb_cat.shape[1]:
            self.emb_fc = nn.Linear(emb_cat.shape[1], C*H*W).to(emb_cat.device)
        emb_proj = self.emb_fc(emb_cat)
        emb_proj = emb_proj.view(B, C, H, W)
        b = b + emb_proj
        x = self.up4(b)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        out = torch.sigmoid(self.final(x))
        
        if out.shape[-1] != IMG_SIZE or out.shape[-2] != IMG_SIZE:
            out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return out


# --- CLIP Scorer (for ensemble, using open_clip) ---
class CLIPScorer:
    def __init__(self, clip_model, clip_preprocess):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

    def calculate_clip_score(self, image_pil, caption):
        try:
            img_t = self.clip_preprocess(image_pil).unsqueeze(0).to(DEVICE)
            txt = open_clip.tokenize([caption]).to(DEVICE)
            with torch.no_grad():
                img_f = self.clip_model.encode_image(img_t)
                txt_f = self.clip_model.encode_text(txt)
                img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-8)
                txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-8)
                return torch.cosine_similarity(img_f, txt_f).item()
        except Exception as e:
            # print(f"CLIP Score calculation error: {e}") # Debugging
            return 0.0

# --- HSV Similarity (for ensemble) ---
def hsv_hist_similarity(img1_pil, img2_pil, bins=32):
    try:
        hsv1 = np.array(img1_pil.convert('HSV')).flatten()
        hsv2 = np.array(img2_pil.convert('HSV')).flatten()
        hist1 = np.histogram(hsv1, bins=bins, range=(0,255))[0]
        hist2 = np.histogram(hsv2, bins=bins, range=(0,255))[0]
        hist1 = hist1 / (np.linalg.norm(hist1) + 1e-8)
        hist2 = hist2 / (np.linalg.norm(hist2) + 1e-8)
        sim = 1 - cosine(hist1, hist2)
        return sim
    except Exception as e:
        # print(f"HSV Similarity calculation error: {e}") # Debugging
        return 0.0

# --- Main Inference Logic ---
if __name__ == '__main__':
    print("--- Inference Pipeline Started ---")

    # 1. Load Generator Model
    G = UNetCondition().to(DEVICE)
    try:
        G.load_state_dict(torch.load(GEN_BEST_PATH, map_location=DEVICE))
        print(f"Generator model loaded successfully from: {GEN_BEST_PATH}")
    except FileNotFoundError:
        print(f"Error: Generator model file not found at {GEN_BEST_PATH}. Please ensure your trained model exists.")
        exit()
    G.eval() # Set to evaluation mode

    # 2. Load CLIP Model for Submission Embeddings (ViT-L-14)
    # This is the CLIP model explicitly required for submission.
    clip_model_submission, _, clip_preprocess_submission = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model_submission.to(DEVICE).eval()
    print("CLIP model (ViT-L-14) for submission embeddings loaded.")

    # Initialize CLIPScorer with the submission CLIP model for ensemble use
    clip_scorer = CLIPScorer(clip_model_submission, clip_preprocess_submission)

    # 3. Load Inference Data
    df_test = pd.read_csv(TEST_CSV_PATH) # IMPORTANT: CHANGE TO ACTUAL TEST CSV PATH!
    inference_dataset = InferenceDataset(df_test, TEST_IMAGE_DIR, img_size=IMG_SIZE) # IMPORTANT: CHANGE TO ACTUAL TEST IMAGE DIRECTORY!
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    out_imgs_pil = []  # List to store final selected PIL images
    out_img_ids = []   # List to store corresponding image IDs (e.g., "TEST_001")
    feat_vecs_submission = [] # List to store CLIP embeddings for submission CSV

    print("\nGenerating images and extracting embeddings...")
    os.makedirs(CFG['SUB_DIR'], exist_ok=True) # Ensure submission directory exists

    with torch.no_grad():
        for i, (gray_images, captions, img_id_bases) in enumerate(tqdm(inference_loader, desc="Inference Progress")):
            gray_images = gray_images.to(DEVICE)

            # --- Generate multiple candidates if ensemble is enabled ---
            candidates_per_batch = [] # List of lists: [[img1_cand1, img1_cand2], [img2_cand1, img2_cand2], ...]
            # Original gray images (for HSV comparison) in PIL format
            original_gray_pil_batch = [transforms.ToPILImage()(img.cpu().squeeze().repeat(1,1,1)[:1]) for img in gray_images] # Convert to single channel PIL for HSV

            for _ in range(CFG['NUM_ENSEMBLE_SAMPLES'] if CFG['USE_ENSEMBLE'] else 1):
                # Apply random seed for each ensemble sample to encourage diversity
                # NOTE: If you want true diversity, ensure your G model's forward pass
                # has some randomness, or you vary text/image embeddings slightly.
                # For now, just re-running generation might not be enough without explicit randomness.
                # Here, we rely on potential floating point variations or implicit randomness.
                seed_everything(CFG['SEED'] + i + _) # Vary seed slightly for each sample

                # CLIP embeddings for UNet (from `transformers` CLIP as trained)
                text_emb_for_unet = get_text_emb_for_unet(captions) # [B, 1024]
                img_emb_for_unet = get_img_emb_for_unet(gray_images) # [B, 1024]

                # Generate fake colors
                current_fake_colors = G(gray_images, text_emb_for_unet, img_emb_for_unet)

                # Convert generated tensors to PIL images for scoring/saving
                current_fake_pil_batch = [transforms.ToPILImage()(img.cpu().clamp(0, 1)) for img in current_fake_colors]
                candidates_per_batch.append(current_fake_pil_batch)
            
            # --- Process each image in the batch ---
            for j in range(len(img_id_bases)):
                img_id_base = img_id_bases[j] # e.g., "TEST_001"
                caption = captions[j]
                original_gray_pil = original_gray_pil_batch[j]
                
                # Collect all candidates for the current image
                current_image_candidates = [candidates_per_batch[sample_idx][j] for sample_idx in range(len(candidates_per_batch))]

                if CFG['USE_ENSEMBLE'] and len(current_image_candidates) > 1:
                    hsv_scores = []
                    clip_scores = []

                    for candidate_pil in current_image_candidates:
                        # HSV Similarity: Original gray image vs Generated candidate
                        # Convert original_gray_pil to 3-channel for HSV if it's 1-channel
                        if original_gray_pil.mode == 'L':
                            original_gray_pil_rgb = original_gray_pil.convert('RGB')
                        else:
                            original_gray_pil_rgb = original_gray_pil
                        hsv_scores.append(hsv_hist_similarity(candidate_pil, original_gray_pil_rgb))
                        
                        # CLIP Score: Generated candidate vs Caption
                        clip_scores.append(clip_scorer.calculate_clip_score(candidate_pil, caption))

                    hsv_scores = np.array(hsv_scores)
                    clip_scores = np.array(clip_scores)

                    # --- Ensemble Selection Logic (as provided) ---
                    best_h_idx = int(np.argmax(hsv_scores))
                    best_c_idx = int(np.argmax(clip_scores))

                    hsv_score_at_best_h = hsv_scores[best_h_idx]
                    clip_score_at_best_h = clip_scores[best_h_idx]

                    selected_idx = -1
                    if hsv_score_at_best_h > clip_score_at_best_h:
                        if hsv_score_at_best_h > 0.9 and clip_score_at_best_h < 0.2:
                            # print(f"  [{img_id_base}] HSV high but CLIP low (H:{hsv_score_at_best_h:.2f}, C:{clip_score_at_best_h:.2f}) -> Select by CLIP")
                            selected_idx = best_c_idx
                        else:
                            # print(f"  [{img_id_base}] Select by HSV (H:{hsv_score_at_best_h:.2f}, C:{clip_score_at_best_h:.2f})")
                            selected_idx = best_h_idx
                    else:
                        # print(f"  [{img_id_base}] Select by CLIP (H:{hsv_score_at_best_h:.2f}, C:{clip_score_at_best_h:.2f})")
                        selected_idx = best_c_idx
                    
                    final_selected_image_pil = current_image_candidates[selected_idx]
                else:
                    # If no ensemble or only one sample, just take the first candidate
                    final_selected_image_pil = current_image_candidates[0]

                out_imgs_pil.append(final_selected_image_pil)
                out_img_ids.append(img_id_base)

                # --- Extract CLIP Embedding for Submission (ViT-L-14) ---
                # This must be done on the FINAL selected image
                img_tensor_for_submission_clip = clip_preprocess_submission(final_selected_image_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = clip_model_submission.encode_image(img_tensor_for_submission_clip)
                    feat /= feat.norm(dim=-1, keepdim=True) # L2 Normalization is crucial
                
                feat_vecs_submission.append(feat.detach().cpu().numpy().reshape(-1))

    # 4. Create Submission Files
    print("\nCreating submission files...")

    # Save PNG images
    for img_pil, img_id_base in tqdm(zip(out_imgs_pil, out_img_ids), total=len(out_img_ids), desc="Saving PNGs"):
        path_out_img = os.path.join(CFG['SUB_DIR'], f"{img_id_base}.png")
        img_pil.save(path_out_img)

    # Create embed_submission.csv
    feat_arr_submission = np.stack(feat_vecs_submission, axis=0)
    vec_columns = [f'vec_{i}' for i in range(feat_arr_submission.shape[1])]
    feat_submission_df = pd.DataFrame(feat_arr_submission, columns=vec_columns)
    feat_submission_df.insert(0, 'ID', out_img_ids)
    feat_submission_df.to_csv(os.path.join(CFG['SUB_DIR'], "embed_submission.csv"), index=False)
    print(f"Generated embed_submission.csv with shape: {feat_arr_submission.shape}")

    # 5. Create ZIP file for leaderboard submission
    print("Creating final submission ZIP file...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in os.listdir(CFG['SUB_DIR']):
            file_path = os.path.join(CFG['SUB_DIR'], file_name)
            if os.path.isfile(file_path) and not file_name.startswith('.'): # Ensure it's a file and not hidden
                zipf.write(file_path, arcname=file_name) # arcname ensures no subdirectories in zip

    print(f"✅ Submission ZIP file created: {ZIP_PATH}")
    print("--- Inference Pipeline Finished ---")