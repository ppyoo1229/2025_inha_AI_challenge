import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
from skimage import color
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os
import random
import re
import string
from collections import Counter
import nltk
# import lpips # 검증이 없어졌으므로 필요 없음
# from pytorch_msssim import ssim # 검증이 없어졌으므로 필요 없음
import cv2
from PIL import Image
import csv

# nltk 데이터 다운로드 (최초 1회 실행 필요)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Config 클래스 정의
class Config:
    def __init__(self):
        self.IMG_SIZE = 256 # 이미지 사이즈를 256으로 줄였습니다.
        self.SEED = 42
        self.OUTPUT_DIR = "./output2"
        self.TRAIN_CSV = "../train.csv" 
        self.INPUT_DIR = "../train" 
        self.GT_DIR = "../train" 
        self.LR = 1e-5 # 기본 러닝 레이트. test_lrs에서 오버라이드됨.
        self.BATCH_SIZE = 1 
        self.NUM_WORKERS = 4
        self.EPOCHS = 2 # <-- 이 부분을 수정하시면 됩니다.
        self.MAX_DATA = None # 모든 데이터 사용
        self.LAMBDA_L1 = 1.4 # 학습 시 사용하지 않지만 Config 유지
        self.LAMBDA_CLIP = 0.5 # 학습 시 사용하지 않지만 Config 유지
        self.LAMBDA_LPIPS = 0.2 # 검증이 없어졌으므로 의미 없음
        self.LAMBDA_SSIM = 0.2 # 검증이 없어졌으므로 의미 없음
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.PROJECT_NAME = "colorization_lr_test"
        self.PATIENCE = 4 # 검증이 없어졌으므로 의미 없음
        self.MAX_PROMPT_TOKENS = 70

CFG = Config()

# 테스트할 러닝 레이트 리스트
test_lrs = [1e-05, 5e-06, 1e-06, 5e-07]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CLIP Features 추출 함수
# (학습 시 직접적인 loss 계산에 사용되지 않아도, L1/CLIP 로깅을 위해 유지)
def get_clip_features(image_tensor):
    global clip_encoder, clip_processor 
    
    # 이미지가 -1 ~ 1 범위일 수 있으므로 0 ~ 1 범위로 변환
    if image_tensor.min() < 0.0:
        image_tensor = (image_tensor + 1) / 2

    pil_images = []
    for i in range(image_tensor.shape[0]):
        # requires_grad=True인 텐서에 numpy()를 직접 호출할 수 없으므로 detach() 사용
        img_np = image_tensor[i].detach().permute(1, 2, 0).cpu().numpy() 
        img_np = (img_np * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np))

    inputs = clip_processor(images=pil_images, return_tensors="pt").to(image_tensor.device)
    features = clip_encoder(**inputs).pooler_output 
    return features

# 헬퍼 함수 (프롬프트 클리닝)
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

def clean_caption_full(caption, remove_phrases, number_words, number_regex, max_tokens=70):
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

# PromptEnhancer 클래스
class PromptEnhancer:
    def __init__(self):
        self.quality_keywords = ["masterpiece", "best quality", "high resolution", "4k", "8k"]
        self.texture_keywords = ["detailed texture", "smooth texture", "realistic texture"]
        self.lighting_keywords = ["dramatic lighting", "soft lighting", "cinematic lighting", "studio lighting"]
        self.scene_keywords = ["wide angle", "close up", "full body shot", "dynamic pose", "indoor scene", "outdoor scene"]
        self.fixed_tail = "colorful, vibrant colors, maintain original structure, do not change structure, only colorize"
        self.color_enhancements = {
            "white": ["pure white", "bright white", "pristine white"], "red": ["vibrant red", "deep red", "scarlet red"],
            "black": ["inky black", "dark black", "jet black"], "green": ["lush green", "vivid green", "emerald green"],
            "blue": ["sky blue", "deep ocean blue", "azure blue"], "yellow": ["golden yellow", "bright yellow", "lemon yellow"],
            "orange": ["fiery orange", "sunny orange", "vibrant orange"], "pink": ["soft pink", "bright pink", "rose pink"],
            "purple": ["royal purple", "deep purple", "lavender purple"], "brown": ["earthy brown", "rich brown", "chocolate brown"],
            "tan": ["sandy tan", "warm tan", "desert tan"], "silver": ["shimmering silver", "polished silver", "chrome silver"],
            "gold": ["lustrous gold", "bright gold", "metallic gold"], "beige": ["creamy beige", "neutral beige", "warm beige"],
            "violet": ["deep violet", "vibrant violet", "amethyst violet"], "cyan": ["electric cyan", "bright cyan", "aquamarine cyan"],
            "magenta": ["vibrant magenta", "bright magenta", "fuchsia magenta"], "gray": ["muted gray", "cool gray", "steel gray"],
            "grey": ["muted grey", "cool grey", "steel grey"], "colorful": ["vibrant colors", "rich color palette", "brightly colored", "rainbow colors", "full color"]
        }
        self.color_words = set(self.color_enhancements.keys())
        self.person_enhance = ["realistic skin", "detailed face", "expressive eyes", "natural skin tone"]
        self.landscape_enhance = ["lush vegetation", "rich color", "clear sky", "natural light"]
        self.food_enhance = ["delicious", "appetizing", "juicy", "fresh", "mouth-watering"]
        self.object_enhance = ["fine detail", "highly detailed", "realistic texture", "material realism"]
        self.art_enhance = ["anime style", "smooth shading", "clean lines"]
        self.base_negative_prompts = "bad quality, grayscale, monochromatic, desaturated, unrealistic colors"
        self.person_keywords = ['person', 'man', 'woman', 'face', 'boy', 'girl', 'child', 'people']
        self.landscape_keywords = ['tree', 'sky', 'mountain', 'field', 'grass', 'river', 'lake', 'flower', 'sun', 'cloud', 'building', 'city']
        self.food_keywords = ['food', 'pizza', 'burger', 'sushi', 'fruit', 'vegetable', 'meal', 'dish', 'dessert']
        self.object_keywords = ['car', 'table', 'chair', 'bottle', 'cup', 'book', 'bag', 'clock', 'window', 'door', 'sign']
        self.art_keywords = ['cartoon', 'drawing', 'illustration', 'anime', 'comic']

    def get_category(self, caption):
        cat = []
        cap = caption.lower()
        if any(k in cap for k in self.person_keywords): cat.append("person")
        if any(k in cap for k in self.landscape_keywords): cat.append("landscape")
        if any(k in cap for k in self.food_keywords): cat.append("food")
        if any(k in cap for k in self.object_keywords): cat.append("object")
        if any(k in cap for k in self.art_keywords): cat.append("art")
        return cat

    def get_color_enhancements(self, caption):
        colors_found = set()
        cap = caption.lower()
        for c in self.color_words:
            if re.search(rf'\b{c}\b', cap): colors_found.add(c)
        enh = []
        for c in colors_found: enh.append(random.choice(self.color_enhancements[c]))
        return enh

    def get_base_negative_prompt(self):
        return self.base_negative_prompts

    def get_enhancement_keywords(self, cleaned_caption):
        enhancement_list = []
        enhancement_list.append(random.choice(self.quality_keywords))
        enhancement_list.append(random.choice(self.texture_keywords))
        enhancement_list.append(random.choice(self.lighting_keywords))
        enhancement_list.append(random.choice(self.scene_keywords))
        categories = self.get_category(cleaned_caption)
        if "person" in categories: enhancement_list.append(random.choice(self.person_enhance))
        if "landscape" in categories: enhancement_list.append(random.choice(self.landscape_enhance))
        if "food" in categories: enhancement_list.append(random.choice(self.food_enhance))
        if "object" in categories: enhancement_list.append(random.choice(self.object_enhance))
        if "art" in categories: enhancement_list.append(random.choice(self.art_enhance))
        color_enhance_list = self.get_color_enhancements(cleaned_caption)
        enhancement_list.extend(color_enhance_list)
        enhancement_list.append(self.fixed_tail)
        return list(dict.fromkeys([x.strip() for x in enhancement_list if x.strip()]))

# DynamicParameterGenerator 클래스
class DynamicParameterGenerator:
    TYPE_CARTOON = 'cartoon'; TYPE_PERSON = 'person'; TYPE_LANDSCAPE = 'landscape'; TYPE_OBJECT = 'object'; TYPE_DEFAULT = 'default'
    TYPE_SHORT_CAPTION = 'short'; TYPE_LONG_CAPTION = 'long'; TYPE_COMPLEX_DETAIL = 'complex_detail'; TYPE_SIMPLE_OUTLINE = 'simple_outline'
    def __init__(self):
        self.guidance_ranges = { self.TYPE_CARTOON: (6.0, 9.0), self.TYPE_PERSON: (7.0, 10.0), self.TYPE_LANDSCAPE: (6.5, 9.5), self.TYPE_OBJECT: (7.0, 10.0), self.TYPE_DEFAULT: (7.0, 9.0) }
        self.step_ranges = { self.TYPE_CARTOON: (25, 35), self.TYPE_SHORT_CAPTION: (30, 45), self.TYPE_LONG_CAPTION: (40, 55), self.TYPE_DEFAULT: (35, 50) }
        self.canny_thresholds = { self.TYPE_DEFAULT: ((50, 150), (100, 200)), self.TYPE_COMPLEX_DETAIL: ((10, 60), (30, 100)), self.TYPE_SIMPLE_OUTLINE: ((100, 200), (150, 250)) }
        self.guidance_keywords_map = {
            self.TYPE_CARTOON: ['cartoon', 'drawing', 'illustration', 'anime'],
            self.TYPE_PERSON: ['person', 'man', 'woman', 'face', 'shirt', 'jacket', 'hat', 'boy', 'girl', 'child', 'people'],
            self.TYPE_LANDSCAPE: ['tree', 'trees', 'sky', 'mountain', 'field', 'grass', 'clouds', 'building', 'buildings', 'city', 'street', 'road', 'river', 'lake', 'ocean'],
            self.TYPE_OBJECT: ['car', 'bus', 'train', 'table', 'chair', 'bowl', 'dog', 'cat', 'book', 'bottle', 'cup', 'food', 'flower', 'clock', 'sign', 'window', 'door']
        }
        self.canny_complex_keywords = [ 'dirty', 'messy', 'rubbish', 'grimy', 'toilet', 'broken', 'detailed', 'intricate', 'complex', 'textured', 'rusty', 'aged', 'graffiti', 'shingles', 'crochet', 'woven', 'engraved' ]
        self.canny_simple_keywords = [ 'cartoon', 'drawing', 'illustration', 'anime', 'simple', 'smooth', 'plain', 'minimal', 'flat' ]

    def _clean_caption_for_keywords(self, caption):
        c = str(caption).lower(); c = c.translate(str.maketrans('', '', string.punctuation)); c = re.sub(r'\s+', ' ', c).strip(); return c

    def _get_category(self, caption, category_map):
        caption_clean = self._clean_caption_for_keywords(caption)
        for category, keywords in category_map.items():
            if any(word in caption_clean for word in keywords): return category
        return self.TYPE_DEFAULT

    def get_optimal_guidance(self, caption):
        category = self._get_category(caption, self.guidance_keywords_map); return random.uniform(*self.guidance_ranges[category])

    def get_optimal_steps(self, caption):
        category = self._get_category_for_steps(caption); return random.randint(*self.step_ranges[category])
        
    def _get_category_for_steps(self, caption):
        caption_clean = self._clean_caption_for_keywords(caption); wc = len(caption_clean.split())
        if any(word in caption_clean for word in self.guidance_keywords_map[self.TYPE_CARTOON]): return self.TYPE_CARTOON
        elif wc < 8: return self.TYPE_SHORT_CAPTION
        elif wc > 16: return self.TYPE_LONG_CAPTION
        else: return self.TYPE_DEFAULT

    def get_optimal_canny_params(self, caption=""):
        caption_clean = self._clean_caption_for_keywords(caption)
        if any(word in caption_clean for word in self.canny_complex_keywords): low_range, high_range = self.canny_thresholds[self.TYPE_COMPLEX_DETAIL]
        elif any(word in caption_clean for word in self.canny_simple_keywords): low_range, high_range = self.canny_thresholds[self.TYPE_SIMPLE_OUTLINE]
        else: low_range, high_range = self.canny_thresholds[self.TYPE_DEFAULT]
        low_threshold = random.randint(low_range[0], low_range[1]); high_threshold = random.randint(high_range[0], high_range[1]); return low_threshold, high_threshold

# tensor_to_pil 함수 (검증이 없지만 혹시 모를 다른 활용을 위해 유지)
def tensor_to_pil(t):
    arr = (t.detach().cpu().float().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ColorizationDataset 클래스
class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, dynamic, img_size=256, random_seed=None):
        self.df = df.reset_index(drop=True); self.input_dir = input_dir; self.gt_dir = gt_dir; self.transform = transform
        self.tokenizer = tokenizer; self.enhancer = enhancer; self.dynamic = dynamic; self.img_size = img_size
        self.max_tokens = CFG.MAX_PROMPT_TOKENS
        all_captions_for_phrases = self.df['caption'].astype(str).tolist()
        self.remove_phrases = build_remove_phrases(all_captions_for_phrases, ngram_ns=(2,3,4), topk=100)
        self.random_seed = random_seed

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cleaned_input_path_from_csv = os.path.normpath(row['input_img_path'])
        cleaned_gt_path_from_csv = os.path.normpath(row['gt_img_path'])
        
        input_image_full_path = os.path.join(self.input_dir, cleaned_input_path_from_csv)
        gt_image_full_path = os.path.join(self.gt_dir, cleaned_gt_path_from_csv)

        original_input_pil_for_canny = Image.open(input_image_full_path).convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
        input_image_np = np.array(original_input_pil_for_canny)
        gray_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)

        raw_caption = str(row['caption'])
        cleaned_caption = clean_caption_full(raw_caption, self.remove_phrases, number_words, number_regex, max_tokens=self.max_tokens)
        
        canny_low, canny_high = self.dynamic.get_optimal_canny_params(cleaned_caption)
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil_for_transform = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil_for_transform)

        gt_image_pil_for_transform = Image.open(gt_image_full_path).convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
        gt_image_transformed = self.transform(gt_image_pil_for_transform)

        gt_rgb_tensor = gt_image_transformed * 2.0 - 1.0 

        gt_rgb_np = gt_image_transformed.permute(1, 2, 0).numpy() # 이곳은 detach 필요 없음, gt_image_transformed는 학습에 사용되지 않으므로 grad 필요 없음
        gt_lab_np = color.rgb2lab(gt_rgb_np)
        ab_channels = torch.from_numpy(gt_lab_np[:, :, 1:]).float().permute(2, 0, 1)
        ab_channels = (ab_channels + 128) / 255.0 * 2 - 1
        ab_channels = torch.clamp(ab_channels, min=-1.0, max=1.0)

        if self.random_seed is not None: random.seed(self.random_seed + idx)
        
        base_neg_prompt = self.enhancer.get_base_negative_prompt()
        enhancement_keywords_list = self.enhancer.get_enhancement_keywords(cleaned_caption)
        
        current_pos_prompt_parts = [cleaned_caption]
        random.shuffle(enhancement_keywords_list)
        
        for keyword_phrase in enhancement_keywords_list:
            temp_prompt = ", ".join(current_pos_prompt_parts + [keyword_phrase])
            # 토큰 길이 체크를 위해 tokenizer 호출, gradient가 필요하지 않으므로 .input_ids 사용
            temp_token_ids = self.tokenizer( temp_prompt, padding=False, truncation=False, return_tensors="pt" ).input_ids[0]
            if len(temp_token_ids) <= CFG.MAX_PROMPT_TOKENS: current_pos_prompt_parts.append(keyword_phrase)
            else: break
        
        pos_prompt = ", ".join(current_pos_prompt_parts)
        # 최종 프롬프트 토큰화 시 max_length와 truncation 적용
        final_pos_tokens = self.tokenizer( pos_prompt, padding="max_length", truncation=True, max_length=CFG.MAX_PROMPT_TOKENS, return_tensors="pt" ).input_ids[0]
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
            "file_name": os.path.basename(cleaned_input_path_from_csv),
            "input_image_full_path": input_image_full_path, 
            "gt_image_full_path": gt_image_full_path         
        }

# LR 테스트를 위한 메인 함수
def run_lr_test(test_lrs, config):
    all_lr_train_losses = {lr: [] for lr in test_lrs}

    global clip_encoder, clip_processor 

    for lr_idx, current_lr in enumerate(test_lrs):
        print(f"\n--- Testing Learning Rate: {current_lr} ({lr_idx+1}/{len(test_lrs)}) ---")
        set_seed(config.SEED) 

        lr_output_dir = os.path.join(config.OUTPUT_DIR, f"lr_test_{str(current_lr).replace('.', 'p')}")
        os.makedirs(lr_output_dir, exist_ok=True)

        accelerator = Accelerator(
            log_with="tensorboard", 
            project_dir=lr_output_dir, 
            mixed_precision="fp16", # FP16 활성화 (메모리 절약)
            gradient_accumulation_steps=8 # 그래디언트 누적 스텝 (메모리 절약)
        )
        weight_dtype = torch.float32 

        controlnet = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=weight_dtype).to(accelerator.device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config.MODEL_PATH,
            controlnet=controlnet,
            torch_dtype=weight_dtype,
        ).to(accelerator.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        lora_cfg = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, init_lora_weights="gaussian",
            target_modules=["to_q", "to_v", "to_k", "to_out.0"]
        )
        pipe.unet.add_adapter(lora_cfg)

        pipe.text_encoder.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        pipe.unet.requires_grad_(False) 
        controlnet.train(); controlnet.requires_grad_(True) 

        for n, p in pipe.unet.named_parameters():
            if "lora" in n: p.requires_grad_(True) 

        params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
        params.extend(controlnet.parameters())
        optimizer = torch.optim.AdamW(params, lr=current_lr) 

        clip_encoder = CLIPVisionModel.from_pretrained(config.CLIP_MODEL, torch_dtype=weight_dtype).to(accelerator.device)
        clip_processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL)
        clip_encoder.eval() 

        prompt_enhancer = PromptEnhancer()
        dynamic_param_gen = DynamicParameterGenerator()
        basic_transform = transforms.Compose([transforms.ToTensor()])
        df = pd.read_csv(config.TRAIN_CSV)
        
        train_df = df 

        tokenizer = CLIPTokenizer.from_pretrained(config.MODEL_PATH, subfolder="tokenizer")

        train_ds = ColorizationDataset(train_df, config.INPUT_DIR, config.GT_DIR, basic_transform, tokenizer, prompt_enhancer, dynamic_param_gen, config.IMG_SIZE, random_seed=None)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

        pipe.unet, pipe.controlnet, optimizer, train_loader = accelerator.prepare(
            pipe.unet, pipe.controlnet, optimizer, train_loader
        )

        pipe.scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps, device=accelerator.device) 

        train_loss_csv_path = os.path.join(lr_output_dir, f"train_losses_lr_{str(current_lr).replace('.', 'p')}.csv")

        with open(train_loss_csv_path, 'w', newline='') as train_csvfile:
            train_writer = csv.writer(train_csvfile)
            train_writer.writerow(['Epoch', 'Step', 'Train_Noise_Loss', 'Train_L1_Loss', 'Train_CLIP_Loss']) 

            for epoch in range(config.EPOCHS):
                accelerator.print(f"\n--- Starting Epoch {epoch+1}/{config.EPOCHS} for LR {current_lr} ---")
                pipe.unet.train(); pipe.controlnet.train() 
                total_train_loss_for_epoch = 0 
                total_train_steps_for_epoch = 0

                for step, batch in enumerate(tqdm(train_loader, desc=f"LR {current_lr} Epoch {epoch+1} Training")):
                    with accelerator.accumulate(pipe.unet, pipe.controlnet):
                        control_image = batch['input_control_image'].to(dtype=weight_dtype, device=accelerator.device)
                        target_rgb = batch['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
                        prompts = [str(batch['pos_prompt'][0])]; neg_prompts = [str(batch['neg_prompt'][0])]

                        text_embeddings = pipe.text_encoder(
                            pipe.tokenizer(
                                prompts, 
                                padding="max_length", 
                                truncation=True, 
                                max_length=pipe.tokenizer.model_max_length, 
                                return_tensors="pt"
                            ).input_ids.to(accelerator.device)
                        )[0]
                        
                        latents = pipe.vae.encode(target_rgb).latent_dist.sample() * pipe.vae.config.scaling_factor
                        noise = torch.randn_like(latents)
                        batch_size = latents.shape[0]
                        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()
                        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                        down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                            noisy_latents, timesteps, encoder_hidden_states=text_embeddings, controlnet_cond=control_image, return_dict=False
                        )
                        noise_pred = pipe.unet(
                            noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample, return_dict=False
                        )[0]

                        loss_noise = F.mse_loss(noise_pred, noise)
                        
                        # --- L1 및 CLIP 손실은 로깅용으로만 계산 ---
                        with torch.no_grad(): 
                            # 예측된 노이즈를 기반으로 원래 이미지의 잠재 공간을 추정하는 간단한 방법
                            # (이것은 완전한 확산 모델 역변환이 아니며, 오직 로깅 목적임)
                            estimated_denoised_latents = noisy_latents - noise_pred 
                            rgb_preds = pipe.vae.decode(estimated_denoised_latents / pipe.vae.config.scaling_factor).sample.clamp(-1,1)

                            loss_l1 = F.l1_loss(((rgb_preds+1)/2), ((target_rgb+1)/2))
                            clip_features_fake = get_clip_features(rgb_preds) 
                            clip_features_real = get_clip_features(target_rgb) 
                            loss_clip = F.mse_loss(clip_features_fake, clip_features_real)
                        # ----------------------------------------
                        
                        loss = loss_noise 

                        accelerator.backward(loss)
                        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(pipe.controlnet.parameters(), max_norm=1.0)
                        optimizer.step(); optimizer.zero_grad()

                        train_writer.writerow([epoch + 1, step, loss_noise.item(), loss_l1.item(), loss_clip.item()])

                        total_train_loss_for_epoch += loss.item()
                        total_train_steps_for_epoch += 1

                        if accelerator.is_main_process and step % 100 == 0:
                            accelerator.print(f"[LR {current_lr}][E {epoch+1}][S {step}] Train Loss: {loss.item():.4f} (Noise: {loss_noise.item():.4f}, L1: {loss_l1.item():.4f}, CLIP: {loss_clip.item():.4f})")

                avg_train_loss = total_train_loss_for_epoch / total_train_steps_for_epoch
                all_lr_train_losses[current_lr].append(avg_train_loss)
                accelerator.print(f"--- Epoch {epoch+1} Average Train Loss for LR {current_lr}: {avg_train_loss:.4f} ---")

# 메인 실행
if __name__ == "__main__":
    run_lr_test(test_lrs, CFG)