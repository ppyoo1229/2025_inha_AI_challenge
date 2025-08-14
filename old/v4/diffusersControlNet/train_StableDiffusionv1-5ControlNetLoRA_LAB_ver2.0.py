"""
[Stable Diffusion v1-5 + ControlNet(Canny) + LoRA] LAB 기반 컬러라이즈 학습 파이프라인
- 텍스트(캡션) 기반 프롬프트 강화 + 동적 파라미터 + ControlNet(Canny) + LoRA + 다중 손실
- 대용량, 다양한 환경에 robust한 실전 학습, 자동화, resume, 얼리스탑, config 통합
- 모든 하이퍼파라미터/경로/옵션 Config에서 관리, robust 전처리, chunk/체크포인트/로깅 자동
-------------------------------------------------------------
1. Stable Diffusion v1-5 (diffusers)
   - 텍스트-이미지 생성 메인 프레임워크
2. ControlNet (lllyasviel/sd-controlnet-canny)
   - 엣지맵(윤곽) 기반 구조 정보 보존
3. LoRA (PEFT, r=8, lora_alpha=32)
   - UNet에 LoRA 어댑터 삽입, 미세조정
4. CLIP (openai/clip-vit-base-patch32)
   - 프롬프트-이미지 의미 손실 및 임베딩 추출
5. PromptEnhancer / DynamicParameterGenerator
   - 품질/질감/조명/장면/색상 등 프롬프트 자동 강화
   - 캡션/키워드 기반 guidance/step/canny threshold 동적 조정
6. 손실함수
   - L1(LAB/이미지) + CLIP 의미 손실 + LPIPS + SSIM 등 가중합
"""
import os
import random
import re
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import color
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig
import colorlab
import math

# CFG
class Config:
    def __init__(self):
        self.IMG_SIZE = 512
        self.IMAGES_PER_CHUNK = 10000
        self.SEED = 42
        self.OUTPUT_DIR = "./output"
        self.TRAIN_CSV = "../train.csv" 
        self.INPUT_DIR = "../train/input_image" 
        self.GT_DIR = "../train/gt_image"     
        self.LR = 1e-5
        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 4
        self.EPOCHS = 10
        self.MAX_DATA = None
        self.LAMBDA_L1 = 1.2 # (1.0 -> 1.2 HSV 평가 비중 고려 결과보고 바꿀 수 있음)
        self.LAMBDA_CLIP = 0.8
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
CFG = Config()

WORK_DIR = os.path.join(CFG.OUTPUT_DIR, 'working_dir')
os.makedirs(WORK_DIR, exist_ok=True)
latest_model_dir = os.path.join(CFG.OUTPUT_DIR, 'latest_checkpoint')
best_model_dir = os.path.join(CFG.OUTPUT_DIR, 'best_model')

# --- ImageDataset 클래스 정의 ---
class ImageDataset(Dataset):
    def __init__(self, data_samples, cfg, tokenizer, transform=None):
        self.data_samples = data_samples
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        input_image_path = os.path.join(self.cfg.INPUT_DIR, sample['input_path'])
        gt_image_path = os.path.join(self.cfg.GT_DIR, sample['gt_path'])
        caption = str(sample['caption'])
        input_image = Image.open(input_image_path).convert("RGB")
        gt_image = Image.open(gt_image_path).convert("RGB")
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        gt_rgb_np = gt_image.permute(1, 2, 0).numpy()
        gt_lab_np = color.rgb2lab(gt_rgb_np)
        L_channel = torch.from_numpy(gt_lab_np[:, :, 0:1]).float().permute(2, 0, 1) / 100.0
        ab_channels = torch.from_numpy(gt_lab_np[:, :, 1:]).float().permute(2, 0, 1)
        ab_channels = (ab_channels + 128) / 255.0 * 2 - 1
        ab_channels = torch.clamp(ab_channels, min=-1.0, max=1.0)
        gt_rgb_tensor = gt_image * 2.0 - 1.0
        return {
            "input_control_image": input_image,
            "gt_rgb_tensor": gt_rgb_tensor,
            "L_channel": L_channel,
            "ab_channels": ab_channels,
            "caption": caption
        }

# --- clean caption ---
def clean_caption(caption):
    patterns = [
        # 확실히 제거할 질문/요청/메타 정보 (공백으로 대체)
        # 단어 경계(\b)
        re.compile(r'\bdo you see\b'), re.compile(r'\bwhat is\b'), re.compile(r'\bis there\b'), 
        re.compile(r'\bcan you\b'), re.compile(r'\bplease\b'), re.compile(r'\btell me\b'), 
        re.compile(r'\bshow me\b'), re.compile(r'\bfind\b'), re.compile(r'\bdescribe\b'), 
        re.compile(r'\bidentify\b'), re.compile(r'\bhow many\b'), re.compile(r'\bwhich\b'), 
        re.compile(r'\bwhere\b'),
        
        # 'in this image'류 
        re.compile(r'in this image(?:,)?(?: there is a)?', re.IGNORECASE),
        re.compile(r'on the (?:left|right) hand side(?:,)?', re.IGNORECASE),
        re.compile(r'in the background(?:,)? there is', re.IGNORECASE),
        re.compile(r'does the shirt look', re.IGNORECASE),
        re.compile(r'on which side of the image is the shopping bag, the right or the left\?', re.IGNORECASE),
        re.compile(r'in this image i can see (?:few|two|a few) (?:persons|people)?', re.IGNORECASE), 
        re.compile(r'this picture (?:is clicked|describe)?(?: outside the city)?', re.IGNORECASE), 

        # 불확실성 표현 (공백으로 대체)
        re.compile(r'\bmaybe\b'), re.compile(r'\bpossibly\b'), re.compile(r'\bprobably\b'),
        re.compile(r'\bcould be\b'), re.compile(r'\bmight be\b'), re.compile(r'\bshould be\b'), 
        re.compile(r'\bappears to be\b'), re.compile(r'\blooks like\b'),
        
        # 단순 연결어구 또는 반복 제거 (공백으로 대체)
        re.compile(r'\band i can also see\b'), re.compile(r'\bwhich is covered with\b'), 
        re.compile(r'\band a\b'), re.compile(r'\bnear the\b'), 
        re.compile(r'\bthere (?:are|is a)(?:,)?\b'), # 'there are' 등 포함
        re.compile(r'\ba(?:,)?(?:n)?(?:,)?(?:other)?(?:,)?(?:an)?\b'), # 관사 제거 
    ]
    
    caption = str(caption).lower()

    # 숫자 접두사 제거
    caption = re.sub(r'^\s*\d+\s*', '', caption)
    
    for pat in patterns:
        caption = pat.sub(' ', caption) # 공백으로 대체

    # 쉼표 앞뒤 공백 제거 및 여러 공백을 하나로 줄이고 앞뒤 공백 제거
    caption = re.sub(r'\s*,\s*', ', ', caption) # 쉼표 뒤에만 공백
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    # 중복 단어 제거 (공백으로 대체 후 중복 제거)
    words = []
    for word in caption.split():
        if not words or words[-1] != word:
            words.append(word)
    caption = ' '.join(words)

    # 'colour' -> 'color' 통일
    caption = caption.replace("colour", "color")

    return caption

# --- PromptEnhancer Class ---
class PromptEnhancer:
    def __init__(self):
        self.illustration_style_keywords = ["digital art", "vector art", "concept art", "anime style", "comic book art"]
        self.quality_keywords = ["masterpiece", "best quality", "high resolution", "4k", "8k"]
        self.texture_keywords = ["detailed texture", "smooth texture", "realistic texture"]
        self.lighting_keywords = ["dramatic lighting", "soft lighting", "cinematic lighting", "studio lighting"]
        self.scene_keywords = ["wide angle", "close up", "full body shot", "dynamic pose", "indoor scene", "outdoor scene"]
        self.skin_color_keywords = ["fair skin", "tanned skin", "dark skin", "light skin", "smooth skin"]
        self.negative_prompts = [
            "bad anatomy", "disfigured", "extra limbs", "mutated hands",
            "poorly drawn hands", "blurry", "low resolution", "ugly", "distorted"
        ]
        self.food_keywords_enhancements = {
            'pizza': ["freshly baked", "gooey cheese", "crispy crust"],
            'burger': ["juicy patty", "fresh toppings", "toasted bun"],
            'sushi': ["fresh fish", "perfectly rolled", "artistic presentation"],
            'food': ["delicious food", "appetizing", "mouth-watering"],
            'fruit': ["fresh fruit", "juicy fruit"], 
            'vegetable': ["fresh vegetable", "organic vegetable"] 
        }
        self.art_keywords = ['cartoon', 'drawing', 'illustration', 'anime', 'comic']
        self.person_keywords = ['person', 'man', 'woman', 'face', 'skin', 'hair', 'eye', 'boy', 'girl', 'child', 'people']
        self.food_detection_keywords = ['food', 'meal', 'dish', 'pizza', 'burger', 'sushi', 'fruit', 'vegetable', 'dessert']

        # --- 분석 결과 반영 ---
        self.detailed_objects = [
            "tank top", "graduation gown", "baseball cap", "doorknob", "clock hands", "shingles",
            "pizza boxes", "donuts", "lilies", "graffiti", "crochet design", "wrought iron fence",
            "silver cargo box", "metal bowls", "metal legs", "drain pipe", "tail light", "baseball team",
            "wooden shutters", "rusted clock face", "elephant tusk", "giraffe horns", "strawberry shapes",
            "wii video game", "silver scissor", "bottles", "creams", "brushes", "red dirt", "green turf",
            "clock", "sign", "building", "car", "train", "truck", "bus", "bike", "van", "fire truck", # 상위 빈도 객체들도 디테일 강조를 위해 추가
            "window", "door", "table", "chair", "bed", "sofa", "couch", "bench", "lamp", "mirror",
            "bowl", "plate", "cup", "glass", "fork", "spoon", "jar", "can", "box", "bag", "toy", "kettle", "basket", "rack",
            "hat", "shirt", "jacket", "dress", "pants", "shoes", "boots", "glove", "backpack", "helmet", "tie", "sunglasses", "bracelet", "necklace", "socks",
            "tree", "flower", "leaves", "grass", "rock", "wall", "fence", "road", "street", "pipe", "pole", "banner", "steeple", "cross", "sign",
            "computer", "phone", "tv", "disc", "speaker", "device",
            "fruit", "vegetable", "dessert", "meat", "broccoli", "tomato" # 음식 관련 상세 키워드
        ]
        
        # 퀄리티/디테일 강조 프롬프트
        self.detail_enhancements = [
            "highly detailed", "intricate details", "sharp focus", "realistic textures",
            "fine detail", "crisp, clear image", "ultra-detailed", "photorealistic", "award-winning photo"
        ]
        
        # 색상 강조 프롬프트 (빈도 높은 색상 위주)
        self.color_enhancements = {
            "white": ["pure white", "bright white", "pristine white"],
            "red": ["vibrant red", "deep red", "scarlet red"],
            "black": ["inky black", "dark black", "jet black"],
            "green": ["lush green", "vivid green", "emerald green"],
            "blue": ["sky blue", "deep ocean blue", "azure blue"],
            "yellow": ["golden yellow", "bright yellow", "lemon yellow"],
            "orange": ["fiery orange", "sunny orange", "vibrant orange"],
            "pink": ["soft pink", "bright pink", "rose pink"],
            "gray": ["muted gray", "cool gray", "steel gray"],
            "brown": ["earthy brown", "rich brown", "chocolate brown"],
            "silver": ["shimmering silver", "polished silver", "chrome silver"],
            "purple": ["royal purple", "deep purple", "lavender purple"],
            "grey": ["muted grey", "cool grey", "steel grey"], # 둘 다 있으니 양쪽 다 처리
            "colorful": ["vibrant colors", "rich color palette", "brightly colored", "rainbow colors", "full color"]
        }

    def enhance_caption(self, caption):
        cleaned_caption = clean_caption(caption)
        enhanced_caption = cleaned_caption
        current_negative_prompts = random.choice(self.negative_prompts)

        # 1. 만화/일러스트 스타일 처리
        if any(word in enhanced_caption for word in self.art_keywords):
            enhanced_caption += f", {random.choice(self.illustration_style_keywords)}"
            current_negative_prompts += ", photorealistic, metallic, hyper-detailed, realistic, detailed skin, " \
                                        "real skin, detailed eyes, 3D render, real photos"
            enhanced_caption += ", soft pastel colors, clean lines, smooth shading"
        else:
            # 2. 일반/포토리얼리스틱 품질, 텍스처, 조명, 장면 관련 키워드 추가 (빈도 분석)
            enhanced_caption = (
                f"{random.choice(self.quality_keywords)}, "
                f"{random.choice(self.texture_keywords)}, "
                f"{random.choice(self.lighting_keywords)}, "
                f"{random.choice(self.scene_keywords)}, "
                f"photorealistic, high resolution, ultra-detailed, " # 기존에 명시적 photorealistic 강화
                f"{enhanced_caption}"
            )
            # 3. 일반 이미지에 대한 부정 프롬프트 강화
            current_negative_prompts += ", poorly rendered, bad quality, low quality, pixelated, noisy, blurry"

        # 4. 인물 관련 강화
        if any(word in enhanced_caption for word in self.person_keywords):
            enhanced_caption += f", {random.choice(self.skin_color_keywords)}, detailed face, expressive eyes"
            current_negative_prompts += ", deformed face, unnatural skin, pale skin, greenish skin, extra fingers, too many limbs"

        # 5. 음식 관련 강화
        food_added = False
        for food_kw, enhancements in self.food_keywords_enhancements.items():
            if food_kw in enhanced_caption:
                enhanced_caption += f", {random.choice(enhancements)}"
                food_added = True
                break 
                
        if not food_added and any(word in enhanced_caption for word in self.food_detection_keywords):
            enhanced_caption += ", delicious food, appetizing, mouth-watering"
        
        # 6. 상세 객체 디테일
        if any(word in enhanced_caption for word in self.detailed_objects):
            enhanced_caption += f", {random.choice(self.detail_enhancements)}"
            
        # 7. 색상 키워드 강화 
        for color_word, enhancements in self.color_enhancements.items():
            if color_word in enhanced_caption:
                enhanced_caption += f", {random.choice(enhancements)}"
        
        # 8. 최종 프롬프트 구성 및 색채화 관련 부정 프롬프트 추가
        final_positive_prompt = f"{enhanced_caption}, colorful, vibrant colors, do not change structure, only colorize"
        final_negative_prompt = current_negative_prompts + ", bad quality, deformed colors, grayscale, monochromatic, desaturated, unrealistic colors, washed out colors, over-saturated colors, color bleeding, color shift, bad lighting, low contrast" # 색채화 특유의 부정 프롬프트 강화 및 추가

        return final_positive_prompt, final_negative_prompt

prompt_enhancer = PromptEnhancer()

# --- DynamicParameterGenerator Class ---
class DynamicParameterGenerator:
    TYPE_CARTOON = 'cartoon'
    TYPE_PERSON = 'person'
    TYPE_LANDSCAPE = 'landscape'
    TYPE_OBJECT = 'object'
    TYPE_DEFAULT = 'default'
    TYPE_SHORT_CAPTION = 'short'
    TYPE_LONG_CAPTION = 'long'
    TYPE_COMPLEX_DETAIL = 'complex_detail'
    TYPE_SIMPLE_OUTLINE = 'simple_outline'

    def __init__(self):
        self.guidance_ranges = {
            self.TYPE_CARTOON: (6.0, 9.0), 
            self.TYPE_PERSON: (7.0, 10.0), 
            self.TYPE_LANDSCAPE: (6.5, 9.5), 
            self.TYPE_OBJECT: (7.0, 10.0), 
            self.TYPE_DEFAULT: (7.0, 9.0) 
        }
        self.step_ranges = {
            self.TYPE_CARTOON: (25, 35), 
            self.TYPE_SHORT_CAPTION: (30, 45), 
            self.TYPE_LONG_CAPTION: (40, 55), 
            self.TYPE_DEFAULT: (35, 50) 
        }
        self.canny_thresholds = {
            self.TYPE_DEFAULT: ((50, 150), (100, 200)), 
            self.TYPE_COMPLEX_DETAIL: ((20, 80), (50, 120)), # 결과보고 상세묘사 못하면 ((10, 60), (30, 100))
            self.TYPE_SIMPLE_OUTLINE: ((100, 200), (150, 250)) 
        }

        self.guidance_keywords_map = {
            self.TYPE_CARTOON: ['cartoon', 'drawing', 'illustration', 'anime'],
            self.TYPE_PERSON: ['person', 'man', 'woman', 'face', 'shirt', 'jacket', 'hat', 'boy', 'girl', 'child', 'people'],
            self.TYPE_LANDSCAPE: ['tree', 'trees', 'sky', 'mountain', 'field', 'grass', 'clouds', 'building', 'buildings', 'city', 'street', 'road', 'river', 'lake', 'ocean'],
            self.TYPE_OBJECT: ['car', 'bus', 'train', 'table', 'chair', 'bowl', 'dog', 'cat', 'book', 'bottle', 'cup', 'food', 'flower', 'clock', 'sign', 'window', 'door'] # 상위 객체 추가
        }
        # canny_complex_keywords와 canny_simple_keywords도 분석 결과와 일치하도록 
        self.canny_complex_keywords = [
            'dirty', 'messy', 'rubbish', 'grimy', 'toilet', 'broken', 
            'detailed', 'intricate', 'complex', 'textured', 'rusty', 'aged', # 디테일 강조 캡션
            'graffiti', 'shingles', 'crochet', 'woven', 'engraved' # 특정 복잡한 질감 
        ]
        self.canny_simple_keywords = [
            'cartoon', 'drawing', 'illustration', 'anime', 'simple', 
            'smooth', 'plain', 'minimal', 'flat' 
        ]

    def _get_category(self, caption, category_map):
        # 캡션 클리닝 후 사용
        caption_clean = clean_caption(caption)
        for category, keywords in category_map.items():
            if any(word in caption_clean for word in keywords):
                return category
        return self.TYPE_DEFAULT

    def get_optimal_guidance(self, caption):
        category = self._get_category(caption, self.guidance_keywords_map)
        return random.uniform(*self.guidance_ranges[category])
            
    def get_optimal_steps(self, caption):
        caption_clean = clean_caption(caption)
        wc = len(caption_clean.split())
        
        if any(word in caption_clean for word in self.guidance_keywords_map[self.TYPE_CARTOON]):
            return random.randint(*self.step_ranges[self.TYPE_CARTOON])
        elif wc < 8: 
            return random.randint(*self.step_ranges[self.TYPE_SHORT_CAPTION])
        elif wc > 16: 
            return random.randint(*self.step_ranges[self.TYPE_LONG_CAPTION])
        else:
            return random.randint(*self.step_ranges[self.TYPE_DEFAULT])
            
    def get_optimal_canny_params(self, caption=""):
        caption_clean = clean_caption(caption)
        
        if any(word in caption_clean for word in self.canny_complex_keywords):
            low_range, high_range = self.canny_thresholds[self.TYPE_COMPLEX_DETAIL]
        elif any(word in caption_clean for word in self.canny_simple_keywords):
            low_range, high_range = self.canny_thresholds[self.TYPE_SIMPLE_OUTLINE]
        else:
            low_range, high_range = self.canny_thresholds[self.TYPE_DEFAULT]
            
        low_threshold = random.randint(low_range[0], low_range[1])
        high_threshold = random.randint(high_range[0], high_range[1])
        return low_threshold, high_threshold

dynamic_param_gen = DynamicParameterGenerator()

# --- DATASET ---
class ColorizationLABDataset(Dataset):
    def __init__(self, df_or_path, input_dir, gt_dir, transform, tokenizer,
                 prompt_enhancer, dynamic_param_gen, max_data=None, img_size=512):
        
        self.df = df_or_path.reset_index(drop=True) if isinstance(df_or_path, pd.DataFrame) else pd.read_csv(df_or_path)
        
        if max_data:
            self.df = self.df.sample(n=min(len(self.df), max_data), random_state=CFG.SEED).reset_index(drop=True)
            
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.enhancer = prompt_enhancer
        self.dynamic = dynamic_param_gen
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        fn_in = os.path.basename(row['input_img_path'])
        fn_gt = os.path.basename(row['gt_img_path'])
        in_path = os.path.join(self.input_dir, fn_in)
        gt_path = os.path.join(self.gt_dir, fn_gt)
        
        if not os.path.exists(in_path): raise FileNotFoundError(f"Input image not found: {in_path}")
        if not os.path.exists(gt_path): raise FileNotFoundError(f"Ground truth image not found: {gt_path}")

        img_gray = Image.open(in_path).convert('L')
        img_rgb = Image.open(gt_path).convert('RGB')

        gray_np = np.array(img_gray.resize((self.img_size, self.img_size)))
        low_threshold, high_threshold = self.dynamic.get_optimal_canny_params(row['caption'])
        canny_edges = cv2.Canny(gray_np, low_threshold, high_threshold)
        
        control_image = self.transform(Image.fromarray(canny_edges)).repeat(3, 1, 1)

        rgb_tensor_0_1 = self.transform(img_rgb)
        gt_rgb_tensor_minus_1_1 = rgb_tensor_0_1 * 2 - 1 # Target for VAE input [-1, 1]

        rgb_np_for_lab = rgb_tensor_0_1.cpu().numpy().transpose(1, 2, 0)
        lab_np = color.rgb2lab(rgb_np_for_lab)

        L_channel = torch.from_numpy(lab_np[:,:,0]).float().unsqueeze(0) / 100.0
        L_channel = torch.clamp(L_channel, 0.0, 1.0) # Crucial: Clamp L channel

        ab_channels = torch.from_numpy(lab_np[:,:,1:]).float().permute(2,0,1)
        ab_channels = (ab_channels + 128) / 255.0 * 2 - 1
        ab_channels = torch.clamp(ab_channels, min=-1.0, max=1.0) # Crucial: Clamp ab channels

        pos_prompt, neg_prompt = self.enhancer.enhance_caption(row['caption'])
        
        input_ids = self.tokenizer(pos_prompt, padding="max_length", truncation=True,
                                   max_length=self.tokenizer.model_max_length,
                                   return_tensors="pt").input_ids[0]
        negative_ids = self.tokenizer(neg_prompt, padding="max_length", truncation=True,
                                      max_length=self.tokenizer.model_max_length,
                                      return_tensors="pt").input_ids[0]

        return {
            "input_ids": input_ids,
            "negative_ids": negative_ids,
            "input_control_image": control_image,
            "gt_rgb_tensor": gt_rgb_tensor_minus_1_1,
            "L_channel": L_channel,
            "ab_channels": ab_channels,
            "caption": row['caption'],
        }

# --- ACCELERATOR & PIPELINE    # mixed_precision="fp16", ---
accelerator = Accelerator(         
    gradient_accumulation_steps=1,  
    log_with="tensorboard",         
    project_dir=CFG.OUTPUT_DIR 
)
accelerator.init_trackers("colorization_training")

# Determine the appropriate dtype for consistency
weight_dtype = torch.float16 if accelerator.mixed_precision == 'fp16' else torch.float32


# --- TRANSFORMS ---
basic_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), 
                      interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(), 
])

# --- Load ControlNet model ---
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=weight_dtype
).to(accelerator.device)

# -- Load Stable Diffusion ControlNet Pipeline ---
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=weight_dtype
).to(accelerator.device)

# --- Set scheduler ---
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# --- Freeze parts ---
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)

# --- Configure and add LoRA to UNet ---
lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05,
    init_lora_weights="gaussian",
    target_modules=["to_q","to_v","to_k","to_out.0"] # Target LoRA to attention layers
)
pipe.unet.add_adapter(lora_cfg)

# --- Define optimizer parameters(LoRA layers and ControlNet) ---
params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters())) # LoRA parameters
controlnet.train(); controlnet.requires_grad_(True) # Ensure ControlNet is trainable
params.extend(controlnet.parameters()) # Add ControlNet parameters

optimizer = torch.optim.AdamW(params, lr=CFG.LR)

# --- CLIP PERCEPTUAL LOSS ---
clip_encoder = CLIPVisionModel.from_pretrained(CFG.CLIP_MODEL, torch_dtype=weight_dtype).to(accelerator.device)
clip_processor = CLIPImageProcessor.from_pretrained(CFG.CLIP_MODEL)
clip_encoder.eval() 

def get_clip_features(imgs):
    pil_list = []
    for img in imgs:
        # Convert tensor to PIL Image: detach, move to CPU, numpy, transpose, scale to 0-255, to uint8
        arr = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_list.append(Image.fromarray(arr))
    
    inputs = clip_processor(images=pil_list, return_tensors="pt")
    # Ensure pixel_values are on the correct device and dtype
    pixel_values = inputs['pixel_values'].to(accelerator.device, dtype=weight_dtype)
    
    with torch.no_grad(): 
        features = clip_encoder(pixel_values=pixel_values).pooler_output
    return features

# --- DATA SPLIT & LOADERS ---
df = pd.read_csv(CFG.TRAIN_CSV)
if CFG.MAX_DATA is not None:
    df = df.sample(n=min(len(df), CFG.MAX_DATA), random_state=CFG.SEED).reset_index(drop=True)

all_data_samples = df.to_dict('records') 

TOTAL_DATASET_SIZE = len(all_data_samples)
IMAGES_PER_CHUNK = CFG.IMAGES_PER_CHUNK
NUM_CHUNKS_PER_FULL_EPOCH = math.ceil(TOTAL_DATASET_SIZE / IMAGES_PER_CHUNK)
accelerator.print(f"Total dataset size: {TOTAL_DATASET_SIZE} images.")
accelerator.print(f"Number of chunks per full epoch: {NUM_CHUNKS_PER_FULL_EPOCH}")

train_split_path = os.path.join(WORK_DIR, 'train_split.csv')
val_split_path = os.path.join(WORK_DIR, 'val_split.csv')

if accelerator.is_main_process:
    tr, va = train_test_split(df, test_size=0.1, random_state=CFG.SEED)
    tr.to_csv(train_split_path, index=False)
    va.to_csv(val_split_path, index=False)
accelerator.wait_for_everyone() 

train_df = pd.read_csv(train_split_path)
val_df = pd.read_csv(val_split_path)

train_ds = ColorizationLABDataset(
    train_df, CFG.INPUT_DIR, CFG.GT_DIR, basic_transform,
    pipe.tokenizer, prompt_enhancer, dynamic_param_gen,
    max_data=CFG.MAX_DATA, img_size=CFG.IMG_SIZE
)
train_loader = DataLoader(
    train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
    num_workers=CFG.NUM_WORKERS, pin_memory=True
)

val_ds = ColorizationLABDataset(
    val_df, CFG.INPUT_DIR, CFG.GT_DIR, basic_transform,
    pipe.tokenizer, prompt_enhancer, dynamic_param_gen,
    max_data=None, img_size=CFG.IMG_SIZE
)
val_loader = DataLoader(
    val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
    num_workers=CFG.NUM_WORKERS, pin_memory=True
)

# --- Accelerator로 준비 (모든 객체들이 초기화되어 있어야 함) ---
pipe.unet, pipe.controlnet, optimizer, train_loader, val_loader = accelerator.prepare(
    pipe.unet, pipe.controlnet, optimizer, train_loader, val_loader
)

# Ensure the best_model_dir exists for saving
if accelerator.is_main_process:
    os.makedirs(best_model_dir, exist_ok=True)

# --- TRAINING LOOP START ---
best_val_loss = float('inf')
start_overall_epoch = 0 
start_chunk_idx = 0     
no_improve = 0
train_losses, val_losses = [], []

# (Optional) 트래커 예외 무시
try:
    accelerator.init_trackers("colorization_training")
except Exception:
    pass

# --- Resume or Start from Scratch ---
if os.path.exists(latest_model_dir) and os.path.isdir(latest_model_dir):
    accelerator.print(f"Resuming training from checkpoint: {latest_model_dir}")
    accelerator.load_state(latest_model_dir)
    tracker_path = os.path.join(latest_model_dir, 'training_tracker.pt')
    if os.path.exists(tracker_path):
        tracker_state = torch.load(tracker_path)
        start_overall_epoch = tracker_state.get('overall_epoch', 0)
        start_chunk_idx = tracker_state.get('chunk_idx', 0)
        best_val_loss = tracker_state['best_val_loss']
        train_losses = tracker_state.get('train_losses', [])
        val_losses = tracker_state.get('val_losses', [])
        accelerator.print(f"Resumed from epoch {start_overall_epoch}, chunk {start_chunk_idx}. Previous best val loss: {best_val_loss:.4f}")
    else:
        accelerator.print("No tracker file, starting from 0.")
else:
    accelerator.print("No checkpoint found, starting from scratch.")

accelerator.print(f"Starting training for {CFG.EPOCHS - start_overall_epoch} overall epochs from overall epoch {start_overall_epoch + 1}.")

for overall_epoch in range(start_overall_epoch, CFG.EPOCHS):
    total_train_loss_for_overall_epoch = 0
    total_train_steps_for_overall_epoch = 0

    # --- Shuffle and Chunk Dataset ---
    random.shuffle(all_data_samples)
    data_chunks = [
        all_data_samples[i:i + IMAGES_PER_CHUNK]
        for i in range(0, len(all_data_samples), IMAGES_PER_CHUNK)
    ]

    for chunk_idx, current_chunk_samples in enumerate(data_chunks):
        if overall_epoch == start_overall_epoch and chunk_idx < start_chunk_idx:
            continue

        accelerator.print(f"\n--- Starting Overall Epoch {overall_epoch+1}, Chunk {chunk_idx+1}/{len(data_chunks)} ---")

        train_dataset_current_chunk = ColorizationLABDataset(
            pd.DataFrame(current_chunk_samples), CFG.INPUT_DIR, CFG.GT_DIR, basic_transform,
            pipe.tokenizer, prompt_enhancer, dynamic_param_gen,
            max_data=None, img_size=CFG.IMG_SIZE
        )
        train_loader_current_chunk = DataLoader(
            train_dataset_current_chunk,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )
        train_loader_current_chunk = accelerator.prepare(train_loader_current_chunk)

        pipe.unet.train()
        pipe.controlnet.train()
        pipe.scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps)

        for step, batch in enumerate(tqdm(train_loader_current_chunk, desc=f"OE {overall_epoch+1}/C {chunk_idx+1} Training")):
            with accelerator.accumulate(pipe.unet, pipe.controlnet):
                # --- 데이터 준비 ---
                control_image = batch['input_control_image'].to(dtype=weight_dtype, device=accelerator.device)
                target_rgb = batch['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
                original_captions = batch['caption']
                prompts, neg_prompts = zip(*(prompt_enhancer.enhance_caption(c) for c in original_captions))
                text_embeddings = pipe.text_encoder(
                    pipe.tokenizer(list(prompts), padding="max_length", truncation=True,
                                   max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
                )[0]

                latents = pipe.vae.encode(target_rgb).latent_dist.sample() * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device=accelerator.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # --- Forward ---
                down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                    controlnet_cond=control_image, return_dict=False
                )
                noise_pred = pipe.unet(
                    noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample, return_dict=False
                )[0]

                # --- Losses ---
                loss_noise_pred = F.mse_loss(noise_pred.to(dtype=weight_dtype), noise.to(dtype=weight_dtype))
                denoised_latents = pipe.scheduler.step(noise_pred, timesteps, noisy_latents, return_dict=True)["prev_sample"]
                rgb_preds = pipe.vae.decode(denoised_latents / pipe.vae.config.scaling_factor).sample.clamp(-1, 1)
                loss_l1_rgb = F.l1_loss(((rgb_preds + 1) / 2).to(dtype=weight_dtype), ((target_rgb + 1) / 2).to(dtype=weight_dtype))

                # CLIP perceptual loss
                clip_features_fake = get_clip_features(((rgb_preds + 1) / 2))
                clip_features_real = get_clip_features(((target_rgb + 1) / 2))
                loss_clip = F.mse_loss(clip_features_fake.to(dtype=weight_dtype), clip_features_real.to(dtype=weight_dtype))

                # LAB L1 loss (학습 포함)
                gt_ab_channels = batch['ab_channels'].to(device=accelerator.device, dtype=weight_dtype)
                pred_rgb_normalized = (rgb_preds + 1) / 2.0
                pred_lab_tensor = colorlab.rgb_to_lab(pred_rgb_normalized)
                pred_ab_channels_tensor = pred_lab_tensor[:, 1:, :, :] / 100.0
                pred_ab_channels_tensor = torch.clamp(pred_ab_channels_tensor, min=-1.0, max=1.0)
                loss_l1_lab = F.l1_loss(pred_ab_channels_tensor, gt_ab_channels)

                # --- 최종 손실 ---
                loss = loss_noise_pred + CFG.LAMBDA_L1 * (loss_l1_rgb + loss_l1_lab) + CFG.LAMBDA_CLIP * loss_clip

                # --- Backward ---
                accelerator.backward(loss)
                if accelerator.mixed_precision == 'fp16':
                    accelerator.clip_grad_norm_(params, max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                # --- 로깅 ---
                if accelerator.is_main_process:
                    current_global_step = (overall_epoch * len(all_data_samples)) + (chunk_idx * IMAGES_PER_CHUNK) + (step * CFG.BATCH_SIZE)
                    accelerator.log({
                        "train_loss_step": loss.item(),
                        "noise_pred_loss_step": loss_noise_pred.item(),
                        "l1_rgb_loss_step": loss_l1_rgb.item(),
                        "l1_lab_loss_step": loss_l1_lab.item(),
                        "clip_loss_step": loss_clip.item(),
                    }, step=current_global_step)
                    if step % 100 == 0:
                        accelerator.print(
                            f"[OE {overall_epoch+1}/C {chunk_idx+1}][Step {step}] "
                            f"Train Loss: {loss.item():.4f} | Noise: {loss_noise_pred.item():.4f} | "
                            f"L1_RGB: {loss_l1_rgb.item():.4f} | L1_LAB: {loss_l1_lab.item():.4f} | CLIP: {loss_clip.item():.4f}"
                        )

        if accelerator.is_main_process:
            accelerator.save_state(latest_model_dir)
            tracker_state = {
                'overall_epoch': overall_epoch,
                'chunk_idx': chunk_idx + 1,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(tracker_state, os.path.join(latest_model_dir, 'training_tracker.pt'))
        
        if overall_epoch == start_overall_epoch and start_chunk_idx > 0:
            start_chunk_idx = 0
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        torch.cuda.empty_cache()

    avg_train_loss = total_train_loss_for_overall_epoch / total_train_steps_for_overall_epoch
    train_losses.append(avg_train_loss)

    # --- Validation ---
    accelerator.print(f"\n--- Starting Validation for Overall Epoch {overall_epoch+1} ---")
    pipe.unet.eval()
    pipe.controlnet.eval()
    total_val_loss = 0
    total_val_batches = 0

    with torch.no_grad():
        for batch_val in tqdm(val_loader, desc=f"Overall Epoch {overall_epoch+1}/{CFG.EPOCHS} Validation"):
            control_image_val = batch_val['input_control_image'].to(dtype=weight_dtype, device=accelerator.device)
            target_rgb_val = batch_val['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
            original_captions_val = batch_val['caption']
            prompts_val, neg_prompts_val = zip(*(prompt_enhancer.enhance_caption(c) for c in original_captions_val))
            guidance_scale_val = dynamic_param_gen.get_optimal_guidance(original_captions_val[0])
            num_inference_steps_val = dynamic_param_gen.get_optimal_steps(original_captions_val[0])

            generated_images = pipe(
                prompt=list(prompts_val),
                image=control_image_val,
                negative_prompt=list(neg_prompts_val),
                guidance_scale=guidance_scale_val,
                num_inference_steps=num_inference_steps_val,
                output_type="pt"
            ).images

            val_l1_rgb = F.l1_loss(generated_images.to(dtype=weight_dtype), ((target_rgb_val + 1) / 2).to(dtype=weight_dtype))
            clip_features_fake_val = get_clip_features(generated_images)
            clip_features_real_val = get_clip_features(((target_rgb_val + 1) / 2))
            val_clip_loss = F.mse_loss(clip_features_fake_val.to(dtype=weight_dtype), clip_features_real_val.to(dtype=weight_dtype))

            val_pred_rgb_normalized = (generated_images + 1) / 2.0
            val_gt_ab_channels = batch_val['ab_channels'].to(device=accelerator.device, dtype=weight_dtype)
            val_pred_lab_tensor = colorlab.rgb_to_lab(val_pred_rgb_normalized)
            val_pred_ab_channels_tensor = val_pred_lab_tensor[:, 1:, :, :] / 100.0
            val_pred_ab_channels_tensor = torch.clamp(val_pred_ab_channels_tensor, min=-1.0, max=1.0)
            val_l1_lab = F.l1_loss(val_pred_ab_channels_tensor, val_gt_ab_channels)

            current_val_loss = CFG.LAMBDA_L1 * (val_l1_rgb + val_l1_lab) + CFG.LAMBDA_CLIP * val_clip_loss
            total_val_loss += current_val_loss.item()
            total_val_batches += 1

    avg_val_loss = total_val_loss / total_val_batches if total_val_batches > 0 else 0
    val_losses.append(avg_val_loss)

    # --- 체크포인트 & 로깅 ---
    if accelerator.is_main_process:
        accelerator.print(f"\n--- Overall Epoch {overall_epoch+1} Summary ---")
        accelerator.print(f"Train Loss (Average): {avg_train_loss:.4f}")
        accelerator.print(f"Val Loss: {avg_val_loss:.4f}")
        accelerator.log({"avg_train_loss_overall_epoch": avg_train_loss, "avg_val_loss_overall_epoch": avg_val_loss}, step=overall_epoch)

        tracker_state_for_next_epoch = {
            'overall_epoch': overall_epoch + 1,
            'chunk_idx': 0,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        accelerator.save_state(latest_model_dir)
        torch.save(tracker_state_for_next_epoch, os.path.join(latest_model_dir, 'training_tracker.pt'))
        accelerator.print(f"End of overall epoch {overall_epoch+1} state saved to: {latest_model_dir}")

        if avg_val_loss < best_val_loss:
            accelerator.print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}!")
            best_val_loss = avg_val_loss
            no_improve = 0
            accelerator.save_state(best_model_dir)
            accelerator.print(f"New best model saved at: {best_model_dir}")
        else:
            no_improve += 1
            accelerator.print(f"Validation loss did not improve. No improvement count: {no_improve}")
            if no_improve >= 5:
                accelerator.print(f"Early stopping at overall epoch {overall_epoch+1}. No improvement for {no_improve} epochs.")
                break

    pipe.unet.train()
    pipe.controlnet.train()

# --- After Training ---
if accelerator.is_main_process:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.title('Loss Curve during Training')
    plt.xlabel('Overall Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, 'loss_curve.png'))
    accelerator.print("Loss curve saved.")

try:
    accelerator.end_training()
except Exception:
    pass

print("Training complete")