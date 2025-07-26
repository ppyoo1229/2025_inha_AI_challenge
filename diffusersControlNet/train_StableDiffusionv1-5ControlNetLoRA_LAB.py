"""
[Stable Diffusion v1-5 + ControlNet(Canny) + LoRA] LAB 기반 컬러라이즈 학습 파이프라인
- 텍스트(캡션) 기반 프롬프트 강화 + 동적 파라미터 + ControlNet(Canny) + LoRA + 다중 손실
-------------------------------------------------------------
1. Stable Diffusion v1-5 [runwayml/stable-diffusion-v1-5, diffusers]
    · 텍스트-이미지 생성 메인 프레임워크
2. ControlNet (Canny) [lllyasviel/sd-controlnet-canny]
    · 엣지맵(윤곽) 기반 구조 정보 보존
3. LoRA (PEFT)
    · UNet에 LoRA 어댑터(r=8, lora_alpha=32) 삽입, 미세조정
4. CLIP (openai/clip-vit-base-patch32)
    · 프롬프트-이미지 의미적 손실 및 임베딩 추출
5. PromptEnhancer / DynamicParameterGenerator
    · 품질/질감/조명/장면/색상 등 프롬프트 자동 강화
    · 캡션/키워드 기반 동적 guidance, step, canny 파라미터 조정
6. 손실함수:  
    · L1(LAB/이미지) + CLIP 의미 손실 + LPIPS + SSIM 등 가중합 손실
-------------------------------------------------------------
- 캡션 전처리/클린(clean_caption_full): 불필요한 n-gram/숫자/중복/길이제한/토큰수 제한 
- 프롬프트 강화: 다양한 품질, 질감, 조명, 색상 키워드 동적 삽입
- 동적 파라미터: 입력 내용/캡션에 따라 guidance/steps/canny threshold 자동 변경
- ControlNet + LoRA 학습만 활성화(UNet/ControlNet만 requires_grad)
- LAB 기반 ground truth(색상채널 ab), GT-LAB/L1 손실 병행
- accelerate + tensorboard 로깅, 체크포인트 자동 저장/복원, 얼리스탑 지원
- 검증 시 L1/CLIP/LPIPS/SSIM/LAB 모든 손실 산출
"""



import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator 
from accelerate.utils import set_seed 
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler 
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model 
from transformers import CLIPVisionModel, CLIPImageProcessor 
import lpips 
from pytorch_msssim import ssim 
from skimage import color 
import pandas as pd
import random
import string
import re
import cv2
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class Config:
    def __init__(self):
        self.IMG_SIZE = 512
        self.SEED = 42
        self.OUTPUT_DIR = "./output"
        self.TRAIN_CSV = "../train.csv"
        self.INPUT_DIR = "../train"
        self.GT_DIR = "../train"
        self.LR = 1e-5
        self.BATCH_SIZE = 1 
        self.NUM_WORKERS = 2
        self.EPOCHS = 15
        self.MAX_DATA = None
        self.LAMBDA_L1 = 1.5
        self.LAMBDA_CLIP = 0.9
        self.LAMBDA_LPIPS = 0.2
        self.LAMBDA_SSIM = 0.8
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.PROJECT_NAME = "colorization_training"
        self.LEARNING_RATE = 1e-5
        self.PATIENCE = 4 
        self.MAX_PROMPT_TOKENS = 35 

CFG = Config()

set_seed(CFG.SEED)

WORK_DIR = os.path.join(CFG.OUTPUT_DIR, 'working_dir')
os.makedirs(WORK_DIR, exist_ok=True)
latest_model_dir = os.path.join(CFG.OUTPUT_DIR, 'latest_checkpoint')
best_model_dir = os.path.join(CFG.OUTPUT_DIR, 'best_model')
os.makedirs(latest_model_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

color_words = set([
    'white', 'black', 'gray', 'grey', 'red', 'blue', 'green', 'yellow', 'orange', 'pink',
    'purple', 'brown', 'tan', 'silver', 'gold', 'beige', 'violet', 'cyan', 'magenta',
    "navy", "olive", "burgundy", "maroon", "teal", "lime", "indigo", "charcoal",
    "peach", "cream", "ivory", "turquoise", "mint", "mustard", "coral"
])
number_words = set([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "a", "an"
])
number_regex = re.compile(r'\b(\d+|[aA]n?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b')

def get_top_ngrams(sentences, n=2, topk=100):
    ngram_counter = Counter()
    for sent in sentences:
        tokens = nltk.word_tokenize(sent.lower())
        tokens = [w for w in tokens if w not in string.punctuation]
        n_grams = list(nltk.ngrams(tokens, n))
        ngram_counter.update(n_grams)
    return [' '.join(k) for k, v in ngram_counter.most_common(topk)]

def build_remove_phrases(captions, ngram_ns=(2,3,4), topk=100):
    remove_phrases = set()
    for n in ngram_ns:
        remove_phrases |= set(get_top_ngrams(captions, n, topk))
    return list(remove_phrases)

def clean_caption_full(caption, remove_phrases, number_words, number_regex, max_tokens=28):
    c = str(caption).lower()
    for phrase in remove_phrases:
        c = re.sub(r'[\s,.!?;:]*' + re.escape(phrase) + r'[\s,.!?;:]*', ' ', c)
    c = c.translate(str.maketrans('', '', string.punctuation))
    c = number_regex.sub(' ', c)
    c = ' '.join([w for w in c.split() if w not in number_words])
    c = re.sub(r'\s+', ' ', c).strip()
    seen = set()
    result = []
    for word in c.split():
        if word not in seen:
            result.append(word)
            seen.add(word)
    return ' '.join(result[:max_tokens])

class PromptEnhancer:
    def __init__(self):
        self.quality_keywords = ["masterpiece", "best quality", "high resolution", "4k", "8k"]
        self.texture_keywords = ["detailed texture", "smooth texture", "realistic texture"]
        self.lighting_keywords = ["dramatic lighting", "soft lighting", "cinematic lighting", "studio lighting"]
        self.scene_keywords = ["wide angle", "close up", "full body shot", "dynamic pose", "indoor scene", "outdoor scene"]
        self.color_enhancements = {
            "white": ["pure white", "bright white", "pristine white"],
            "red": ["vibrant red", "deep red", "scarlet red"],
            "black": ["inky black", "dark black", "jet black"],
            "green": ["lush green", "vivid green", "emerald green"],
            "blue": ["sky blue", "deep ocean blue", "azure blue"],
            "yellow": ["golden yellow", "bright yellow", "lemon yellow"],
            "orange": ["fiery orange", "sunny orange", "vibrant orange"],
            "pink": ["soft pink", "bright pink", "rose pink"],
            "purple": ["royal purple", "deep purple", "lavender purple"],
            "brown": ["earthy brown", "rich brown", "chocolate brown"],
            "tan": ["sandy tan", "warm tan", "desert tan"],
            "silver": ["shimmering silver", "polished silver", "chrome silver"],
            "gold": ["lustrous gold", "bright gold", "metallic gold"],
            "beige": ["creamy beige", "neutral beige", "warm beige"],
            "violet": ["deep violet", "vibrant violet", "amethyst violet"],
            "cyan": ["electric cyan", "bright cyan", "aquamarine cyan"],
            "magenta": ["vibrant magenta", "bright magenta", "fuchsia magenta"],
            "gray": ["muted gray", "cool gray", "steel gray"],
            "grey": ["muted grey", "cool grey", "steel grey"],
            "colorful": ["vibrant colors", "rich color palette", "brightly colored", "rainbow colors", "full color"]
        }
        self.negative_prompts_general = [
            "bad anatomy", "disfigured", "extra limbs", "mutated hands", "blurry", "low resolution", "ugly", "distorted"
        ]
        self.negative_prompts_person = [
            "deformed face", "unnatural skin", "extra fingers", "too many limbs", "pale skin"
        ]
        self.negative_prompts_landscape = [
            "washed out colors", "low contrast", "color shift", "color bleeding", "over-saturated colors"
        ]
        self.negative_prompts_object = [
            "wrong perspective", "misshapen object", "deformed objects"
        ]
    
    def get_base_negative_prompt(self, caption):
        negs = random.sample(self.negative_prompts_general, k=random.randint(2, 4))
        if any(word in caption for word in ['person', 'man', 'woman', 'face', 'people']):
            negs += random.sample(self.negative_prompts_person, k=2)
        if any(word in caption for word in ['tree', 'sky', 'field', 'mountain', 'river', 'cloud']):
            negs += random.sample(self.negative_prompts_landscape, k=2)
        if any(word in caption for word in ['car', 'train', 'bus', 'object', 'table', 'chair']):
            negs += random.sample(self.negative_prompts_object, k=1)
        return ', '.join(set(negs)) + ", bad quality, grayscale, monochromatic, desaturated, unrealistic colors"

    def get_enhancement_keywords(self, caption):
        enhancement_parts = []
        enhancement_parts.append(random.choice(self.quality_keywords))
        enhancement_parts.append(random.choice(self.texture_keywords))
        enhancement_parts.append(random.choice(self.lighting_keywords))
        enhancement_parts.append(random.choice(self.scene_keywords))
        enhancement_parts.append("photorealistic, high resolution, best quality")
        
        caption_lower = caption.lower()
        for color_word, enhancements in self.color_enhancements.items():
            if color_word in caption_lower:
                enhancement_parts.append(random.choice(enhancements))

        enhancement_parts.append("colorful, vibrant colors, maintain original structure, do not change structure, only colorize")
        
        return enhancement_parts

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
            self.TYPE_COMPLEX_DETAIL: ((10, 60), (30, 100)),
            self.TYPE_SIMPLE_OUTLINE: ((100, 200), (150, 250))
        }

        self.guidance_keywords_map = {
            self.TYPE_CARTOON: ['cartoon', 'drawing', 'illustration', 'anime'],
            self.TYPE_PERSON: ['person', 'man', 'woman', 'face', 'shirt', 'jacket', 'hat', 'boy', 'girl', 'child', 'people'],
            self.TYPE_LANDSCAPE: ['tree', 'trees', 'sky', 'mountain', 'field', 'grass', 'clouds', 'building', 'buildings', 'city', 'street', 'road', 'river', 'lake', 'ocean'],
            self.TYPE_OBJECT: ['car', 'bus', 'train', 'table', 'chair', 'bowl', 'dog', 'cat', 'book', 'bottle', 'cup', 'food', 'flower', 'clock', 'sign', 'window', 'door']
        }
        self.canny_complex_keywords = [
            'dirty', 'messy', 'rubbish', 'grimy', 'toilet', 'broken',
            'detailed', 'intricate', 'complex', 'textured', 'rusty', 'aged',
            'graffiti', 'shingles', 'crochet', 'woven', 'engraved'
        ]
        self.canny_simple_keywords = [
            'cartoon', 'drawing', 'illustration', 'anime', 'simple',
            'smooth', 'plain', 'minimal', 'flat'
        ]

    def _clean_caption_for_keywords(self, caption):
        c = str(caption).lower()
        c = c.translate(str.maketrans('', '', string.punctuation))
        c = re.sub(r'\s+', ' ', c).strip()
        return c

    def _get_category(self, caption, category_map):
        caption_clean = self._clean_caption_for_keywords(caption)
        for category, keywords in category_map.items():
            if any(word in caption_clean for word in keywords):
                return category
        return self.TYPE_DEFAULT

    def get_optimal_guidance(self, caption):
        category = self._get_category(caption, self.guidance_keywords_map)
        return random.uniform(*self.guidance_ranges[category])

    def get_optimal_steps(self, caption):
        caption_clean = self._clean_caption_for_keywords(caption)
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
        caption_clean = self._clean_caption_for_keywords(caption)

        if any(word in caption_clean for word in self.canny_complex_keywords):
            low_range, high_range = self.canny_thresholds[self.TYPE_COMPLEX_DETAIL]
        elif any(word in caption_clean for word in self.canny_simple_keywords):
            low_range, high_range = self.canny_thresholds[self.TYPE_SIMPLE_OUTLINE]
        else:
            low_range, high_range = self.canny_thresholds[self.TYPE_DEFAULT]

        low_threshold = random.randint(low_range[0], low_range[1])
        high_threshold = random.randint(high_range[0], high_range[1])
        return low_threshold, high_threshold

class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, dynamic, img_size=512, random_seed=None, max_tokens=30):
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.enhancer = enhancer
        self.dynamic = dynamic
        self.img_size = img_size
        self.max_tokens = max_tokens
        all_captions_for_phrases = self.df['caption'].astype(str).tolist()
        self.remove_phrases = build_remove_phrases(all_captions_for_phrases, ngram_ns=(2,3,4), topk=100)
        self.random_seed = random_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cleaned_input_path_from_csv = os.path.normpath(row['input_img_path'])
        cleaned_gt_path_from_csv = os.path.normpath(row['gt_img_path'])

        input_image_path = os.path.join(self.input_dir, cleaned_input_path_from_csv)
        gt_image_path = os.path.join(self.gt_dir, cleaned_gt_path_from_csv)

        input_image_pil = Image.open(input_image_path).convert("RGB")

        input_image_np = np.array(input_image_pil)

        gray_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)

        raw_caption = str(row['caption'])
        cleaned_caption = clean_caption_full(raw_caption, self.remove_phrases, number_words, number_regex, max_tokens=self.max_tokens)
        
        canny_low, canny_high = self.dynamic.get_optimal_canny_params(cleaned_caption)
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil)

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_image_transformed = self.transform(gt_image_pil)

        gt_rgb_tensor = gt_image_transformed * 2.0 - 1.0 

        gt_rgb_np = gt_image_transformed.permute(1, 2, 0).numpy()
        gt_lab_np = color.rgb2lab(gt_rgb_np)
        ab_channels = torch.from_numpy(gt_lab_np[:, :, 1:]).float().permute(2, 0, 1)
        ab_channels = (ab_channels + 128) / 255.0 * 2 - 1
        ab_channels = torch.clamp(ab_channels, min=-1.0, max=1.0)

        if self.random_seed is not None:
            random.seed(self.random_seed + idx)
        
        # 프롬프트 토큰 길이 조절 로직 재설계
        base_neg_prompt = self.enhancer.get_base_negative_prompt(cleaned_caption)
        
        cleaned_caption_token_ids = self.tokenizer(
            cleaned_caption, 
            padding=False, 
            truncation=False, 
            return_tensors="pt"
        ).input_ids[0]
        
        if len(cleaned_caption_token_ids) > CFG.MAX_PROMPT_TOKENS:
            truncated_ids = self.tokenizer(
                cleaned_caption, 
                padding="max_length", 
                truncation=True, 
                max_length=CFG.MAX_PROMPT_TOKENS, 
                return_tensors="pt"
            ).input_ids[0]
            pos_prompt = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
            final_pos_tokens = self.tokenizer(pos_prompt, return_tensors="pt", max_length=CFG.MAX_PROMPT_TOKENS, truncation=True).input_ids[0]
            pos_prompt = self.tokenizer.decode(final_pos_tokens, skip_special_tokens=True)

        else: 
            enhancement_keywords_list = self.enhancer.get_enhancement_keywords(cleaned_caption)
            
            current_pos_prompt_parts = [cleaned_caption]
            
            random.shuffle(enhancement_keywords_list) 
            
            for keyword_phrase in enhancement_keywords_list:
                temp_prompt = ", ".join(current_pos_prompt_parts + [keyword_phrase])
                temp_token_ids = self.tokenizer(
                    temp_prompt, 
                    padding=False, 
                    truncation=False, 
                    return_tensors="pt"
                ).input_ids[0]
                
                if len(temp_token_ids) <= CFG.MAX_PROMPT_TOKENS:
                    current_pos_prompt_parts.append(keyword_phrase)
                else:
                    break 
            
            pos_prompt = ", ".join(current_pos_prompt_parts)
            final_pos_tokens = self.tokenizer(pos_prompt, return_tensors="pt", max_length=CFG.MAX_PROMPT_TOKENS, truncation=True).input_ids[0]
            pos_prompt = self.tokenizer.decode(final_pos_tokens, skip_special_tokens=True)


        guidance = self.dynamic.get_optimal_guidance(cleaned_caption)
        steps = self.dynamic.get_optimal_steps(cleaned_caption)

        return {
            "input_control_image": input_control_image,
            "gt_rgb_tensor": gt_rgb_tensor,
            "ab_channels": ab_channels,
            "caption": raw_caption,
            "cleaned_caption": cleaned_caption,
            "pos_prompt": pos_prompt, 
            "neg_prompt": base_neg_prompt,
            "guidance": guidance,
            "steps": steps,
            "canny_low": canny_low,
            "canny_high": canny_high,
        }


prompt_enhancer = PromptEnhancer()
dynamic_param_gen = DynamicParameterGenerator()

def tensor_to_pil(t):
    arr = (t.detach().cpu().float().permute(1, 2, 0).numpy() * 255)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


accelerator = Accelerator(
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir=CFG.OUTPUT_DIR
)
accelerator.init_trackers("colorization_training")
weight_dtype = torch.float32

resume_training = True

tracker_path = os.path.join(latest_model_dir, 'training_tracker.pt')

overall_epoch = 0
train_losses = []
val_losses = []
best_val_loss = float('inf')
no_improve = 0 # 얼리 스타핑을 위한 개선되지 않은 에포크 수

if resume_training and os.path.exists(latest_model_dir) and os.path.exists(tracker_path):
    accelerator.print(f"[INFO] Resuming training from {latest_model_dir}")
    try:
        accelerator.load_state(latest_model_dir)
        tracker_state = torch.load(tracker_path)
        overall_epoch = tracker_state.get('overall_epoch', 0)
        train_losses = tracker_state.get('train_losses', [])
        val_losses = tracker_state.get('val_losses', [])
        best_val_loss = tracker_state.get('best_val_loss', float('inf'))
        no_improve = tracker_state.get('no_improve', 0)

        if train_losses:
            avg_train_loss = train_losses[-1]

        accelerator.print(f"[INFO] Resumed at epoch {overall_epoch}. Best Val Loss: {best_val_loss:.4f}")
    except Exception as e:
        accelerator.print(f"[WARNING] Failed to load checkpoint, starting fresh. Error: {e}")
        overall_epoch = 0
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        no_improve = 0
        avg_train_loss = 0.0
else:
    accelerator.print("[INFO] No valid checkpoint found or resume_training is False, starting fresh.")


basic_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

controlnet = ControlNetModel.from_pretrained(
    CFG.CONTROLNET_PATH, torch_dtype=weight_dtype
).to(accelerator.device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    CFG.MODEL_PATH, controlnet=controlnet, torch_dtype=weight_dtype
).to(accelerator.device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["to_q", "to_v", "to_k", "to_out.0"]
)
pipe.unet = get_peft_model(pipe.unet, lora_cfg)

clip_encoder = CLIPVisionModel.from_pretrained(CFG.CLIP_MODEL, torch_dtype=weight_dtype).to(accelerator.device)
clip_processor = CLIPImageProcessor.from_pretrained(CFG.CLIP_MODEL)
clip_encoder.eval()

def get_clip_features(imgs):
    pil_list = []
    for img_tensor in imgs:
        arr = (img_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_list.append(Image.fromarray(arr))
    inputs = clip_processor(images=pil_list, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(accelerator.device, dtype=weight_dtype)
    with torch.no_grad():
        features = clip_encoder(pixel_values=pixel_values).pooler_output
    return features

lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
lpips_loss_fn.eval()

df = pd.read_csv(CFG.TRAIN_CSV)
if CFG.MAX_DATA is not None:
    df = df.sample(n=min(len(df), CFG.MAX_DATA), random_state=CFG.SEED).reset_index(drop=True)
train_split_path = os.path.join(WORK_DIR, 'train_split.csv')
val_split_path = os.path.join(WORK_DIR, 'val_split.csv')
if accelerator.is_main_process:
    tr, va = train_test_split(df, test_size=0.1, random_state=CFG.SEED)
    tr.to_csv(train_split_path, index=False)
    va.to_csv(val_split_path, index=False)
accelerator.wait_for_everyone()
train_df = pd.read_csv(train_split_path)
val_df = pd.read_csv(val_split_path)

train_ds = ColorizationDataset(train_df, CFG.INPUT_DIR, CFG.GT_DIR, basic_transform, pipe.tokenizer, prompt_enhancer, dynamic_param_gen, img_size=CFG.IMG_SIZE, random_seed=None)
val_ds = ColorizationDataset(val_df, CFG.INPUT_DIR, CFG.GT_DIR, basic_transform, pipe.tokenizer, prompt_enhancer, dynamic_param_gen, img_size=CFG.IMG_SIZE, random_seed=42)
train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

controlnet.train()
controlnet.requires_grad_(True)
params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
params.extend(controlnet.parameters())
optimizer = torch.optim.AdamW(params, lr=CFG.LR)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.LR*0.1)

pipe.unet, pipe.controlnet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    pipe.unet, pipe.controlnet, optimizer, train_loader, val_loader, scheduler
)

for epoch in range(overall_epoch, CFG.EPOCHS):
    accelerator.print(f"\n--- Starting Overall Epoch {epoch+1}/{CFG.EPOCHS} ---")
    pipe.unet.train()
    pipe.controlnet.train()
    total_train_loss_for_overall_epoch = 0
    total_train_steps_for_overall_epoch = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        with accelerator.accumulate(pipe.unet, pipe.controlnet):
            control_image = batch['input_control_image'].to(dtype=weight_dtype, device=accelerator.device)
            target_rgb = batch['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
            prompts = [str(batch['pos_prompt'][0])] 
            neg_prompts = [str(batch['neg_prompt'][0])] 

            text_embeddings = pipe.text_encoder(
                pipe.tokenizer(prompts, padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
            )[0]
            latents = pipe.vae.encode(target_rgb).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image, return_dict=False
            )
            noise_pred = pipe.unet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, return_dict=False
            )[0]

            loss = F.mse_loss(noise_pred.to(dtype=weight_dtype), noise.to(dtype=weight_dtype))

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pipe.controlnet.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss_for_overall_epoch += loss.item()
            total_train_steps_for_overall_epoch += 1

            if accelerator.is_main_process and step % 100 == 0:
                accelerator.print(f"[OE {epoch+1}][Step {step}] Train Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss_for_overall_epoch / total_train_steps_for_overall_epoch
    train_losses.append(avg_train_loss)

    accelerator.print(f"\n--- Starting Validation for Overall Epoch {epoch+1} ---")
    pipe.unet.eval()
    pipe.controlnet.eval()
    total_val_loss, total_val_batches = 0, 0

    with torch.no_grad():
        for batch_val in tqdm(val_loader, desc=f"Overall Epoch {epoch+1}/{CFG.EPOCHS} Validation"):
            control_image_tensor = batch_val['input_control_image'][0] 
            
            control_image_for_pipe_list = [tensor_to_pil(control_image_tensor)]

            prompt_list = [str(batch_val['pos_prompt'][0])] 
            neg_prompt_list = [str(batch_val['neg_prompt'][0])] 
            guidance_scale_to_use = float(batch_val['guidance'][0])
            num_inference_steps_to_use = int(batch_val['steps'][0])

            generated_images = pipe(
                prompt=prompt_list,
                image=control_image_for_pipe_list,
                negative_prompt=neg_prompt_list,
                guidance_scale=guidance_scale_to_use,
                num_inference_steps=num_inference_steps_to_use,
                output_type="pt"
            ).images

            generated_images_01 = (generated_images + 1) / 2.0
            target_rgb_val = batch_val['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
            target_rgb_val_01 = (target_rgb_val + 1) / 2.0

            val_l1_rgb = F.l1_loss(generated_images_01.to(dtype=weight_dtype), target_rgb_val_01.to(dtype=weight_dtype))

            clip_features_fake_val = get_clip_features(generated_images_01)
            clip_features_real_val = get_clip_features(target_rgb_val_01)
            val_clip_loss = F.mse_loss(clip_features_fake_val.to(dtype=weight_dtype), clip_features_real_val.to(dtype=weight_dtype))

            val_lpips_loss = lpips_loss_fn(generated_images_01.to(dtype=weight_dtype), target_rgb_val_01.to(dtype=weight_dtype)).mean()

            val_ssim_loss = 1 - ssim(
                generated_images_01.to(dtype=weight_dtype), target_rgb_val_01.to(dtype=weight_dtype),
                data_range=1.0, size_average=True
            )

            val_gt_ab_channels = batch_val['ab_channels'].to(device=accelerator.device, dtype=weight_dtype)

            val_pred_rgb_np = generated_images_01.permute(0, 2, 3, 1).cpu().numpy()
            val_pred_lab_np_list = []
            for img_np in val_pred_rgb_np:
                clipped_img_np = np.clip(img_np, 0.0, 1.0)
                val_pred_lab_np_list.append(color.rgb2lab(clipped_img_np))

            val_pred_lab_tensor = torch.stack([
                torch.from_numpy(lab_img).float().permute(2, 0, 1) for lab_img in val_pred_lab_np_list
            ]).to(accelerator.device, dtype=weight_dtype)

            val_pred_ab_channels_tensor = val_pred_lab_tensor[:, 1:, :, :]
            val_pred_ab_channels_tensor = (val_pred_ab_channels_tensor + 128) / 255.0 * 2 - 1
            val_pred_ab_channels_tensor = torch.clamp(val_pred_ab_channels_tensor, min=-1.0, max=1.0)

            val_l1_lab = F.l1_loss(val_pred_ab_channels_tensor, val_gt_ab_channels)

            val_total_loss_item = CFG.LAMBDA_L1 * (val_l1_rgb.item() + val_l1_lab.item()) \
                                + CFG.LAMBDA_CLIP * val_clip_loss.item() \
                                + CFG.LAMBDA_LPIPS * val_lpips_loss.item() \
                                + CFG.LAMBDA_SSIM * val_ssim_loss.item()
            total_val_loss += val_total_loss_item
            total_val_batches += 1

    avg_val_loss = total_val_loss / total_val_batches
    val_losses.append(avg_val_loss)

    if accelerator.is_main_process:
        accelerator.print(f"\n--- Overall Epoch {epoch+1} Summary ---")
        accelerator.print(f"Train Loss (Average): {avg_train_loss:.4f}")
        accelerator.print(f"Val Loss: {avg_val_loss:.4f}")
        accelerator.log({"avg_train_loss_overall_epoch": avg_train_loss, "avg_val_loss_overall_epoch": avg_val_loss}, step=epoch)

        accelerator.save_state(output_dir=latest_model_dir)
        tracker_state = {
            'overall_epoch': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'no_improve': no_improve,
        }
        torch.save(tracker_state, tracker_path)
        accelerator.print(f"Current model and tracker state saved to {latest_model_dir}")

        if avg_val_loss < best_val_loss:
            accelerator.print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best model.")
            best_val_loss = avg_val_loss
            no_improve = 0 # 개선되었으므로 no_improve 초기화
            unwrapped_unet = accelerator.unwrap_model(pipe.unet)
            unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
            unwrapped_unet.save_pretrained(os.path.join(best_model_dir, "unet_best"))
            unwrapped_controlnet.save_pretrained(os.path.join(best_model_dir, "controlnet_best"))
            pipe.tokenizer.save_pretrained(os.path.join(best_model_dir, "tokenizer_best"))
            pipe.text_encoder.save_pretrained(os.path.join(best_model_dir, "text_encoder_best"))
            accelerator.print(f"Best model saved to {best_model_dir}")
        else:
            no_improve += 1 # 개선되지 않았으므로 no_improve 증가
            accelerator.print(f"Validation loss did not improve. No improvement count: {no_improve}")
            if no_improve >= CFG.PATIENCE: # patience를 초과하면 학습 종료
                accelerator.print(f"Early stopping triggered! Validation loss did not improve for {CFG.PATIENCE} consecutive epochs.")
                break # 에포크 루프 종료

        idx = np.random.randint(len(val_ds))
        sample_data = val_ds[idx]

        control_image_sample = sample_data['input_control_image'] 
        sample_control_image_pil = [tensor_to_pil(control_image_sample)]

        pos_prompt_sample = str(sample_data['pos_prompt']) 
        neg_prompt_sample = str(sample_data['neg_prompt']) 
        guidance_scale_sample = float(sample_data['guidance'])
        num_steps_sample = int(sample_data['steps'])

        accelerator.print(f"\n--- Generating Sample Image for Epoch {epoch+1} ---")
        accelerator.print(f"    Using original caption: {sample_data['caption']}")
        accelerator.print(f"    Using positive prompt: {pos_prompt_sample}")
        accelerator.print(f"    Using negative prompt: {neg_prompt_sample}")
        accelerator.print(f"    Using guidance_scale: {guidance_scale_sample:.2f}")
        accelerator.print(f"    Using num_inference_steps: {num_steps_sample}")

        with torch.no_grad():
            sample_generated_image_tensor = pipe(
                prompt=[pos_prompt_sample], 
                image=sample_control_image_pil,
                negative_prompt=[neg_prompt_sample], 
                guidance_scale=guidance_scale_sample,
                num_inference_steps=num_steps_sample,
                output_type="pt"
            ).images[0].detach().cpu()

        sample_image_np = ((sample_generated_image_tensor + 1) / 2.0).clamp(0, 1).numpy().transpose(1, 2, 0) * 255
        sample_image_pil = Image.fromarray(sample_image_np.astype(np.uint8))
        out_path = os.path.join(CFG.OUTPUT_DIR, f"sample_epoch{epoch+1}.png")
        sample_image_pil.save(out_path)
        accelerator.print(f"Sample image for epoch {epoch+1} saved: {out_path}")

    gc.collect()
    torch.cuda.empty_cache()
    pipe.unet.train()
    pipe.controlnet.train()

try:
    accelerator.end_training()
except Exception:
    pass 

accelerator.print("Training complete")