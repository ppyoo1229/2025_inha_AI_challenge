# 추론시 merge_adapter_weights 까먹지 마셈
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import load_file, save_file
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed # Avoid conflict with custom set_seed
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from transformers import CLIPTokenizer, AutoTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel, get_scheduler
import cv2
from skimage import color
import random
import re
import string
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split
import math
import gc
import shutil
import lpips
from pytorch_msssim import ssim as msssim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    nltk.download('punkt')

# --- config ---
class Config:
    def __init__(self):
        self.IMG_SIZE = 512
        self.SEED = 42
        self.OUTPUT_DIR = "./output5"
        self.TRAIN_CSV = "../train.csv" 
        self.INPUT_DIR = ".." 
        self.GT_DIR = ".." 
        self.LR = 1e-6
        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 4
        self.EPOCHS = 5
        self.MAX_DATA = None 
        self.LAMBDA_L1 = 0.2 # 0.1 # 0.7
        self.LAMBDA_CLIP = 0.5 # 0.05 # 1.0
        self.LAMBDA_LPIPS = 0.2 # 0.1 # 0.7
        self.LAMBDA_SSIM =  0.05 # 0.01 # 0.2
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5" 
        self.PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.PROJECT_NAME = "colorization_training"
        self.PATIENCE = 99999
        self.MAX_PROMPT_TOKENS = 77
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "erotic", "nude", "breast", "ass", "penis", "vagina"]
        self.SFW_CAPTION_REPLACEMENT = "a high quality image, realistic, clean, beautiful, bright, colorful" # NSFW 캡션 대체
        self.GRADIENT_ACCUMULATION_STEPS = 4
        self.MAX_GRAD_NORM = 1.0
        self.LR_SCHEDULER_TYPE = "cosine"
        self.LR_WARMUP_STEPS = 500
        self.ADAM_BETA1 = 0.9
        self.ADAM_BETA2 = 0.999
        self.ADAM_WEIGHT_DECAY = 1e-2
        self.ADAM_EPSILON = 1e-08
        self.MIXED_PRECISION = "no" # "no", "fp16", "bf16" 
        self.REPORT_TO = "tensorboard" # "tensorboard", "wandb", "all"
        self.MAX_TRAIN_STEPS = None # 총 학습 스텝 수 (None이면 EPOCHS로 계산)
        self.RESUME_FROM_CHECKPOINT = "" # "./output5/checkpoint-40"
        self.SAMPLE_SAVE_START_STEP = 400 # 샘플 이미지 저장 시작 스텝
        self.SAMPLE_SAVE_END_STEP = 500 # 샘플 이미지 저장 종료 스텝
        self.NUM_SAMPLES_TO_SAVE = 3 # 검증 시 저장할 샘플 이미지 개수
        self.MAX_CHECKPOINTS_TO_KEEP = 2 # 유지할 체크포인트 최대 개수
        self.LOG_INTERVAL = 10 # 10 스텝마다 로깅
        self.VAL_INTERVAL = 1 # 1 에폭마다 검증
        self.SAVE_AND_VAL_INTERVAL = 2000

CFG = Config()

# --- Helper Functions ---
def filter_config_types(config_dict):
    ALLOWED = (int, float, str, bool)
    return {k: v for k, v in config_dict.items() if isinstance(v, ALLOWED)}

def debug_tensor_info(name, tensor):
    try:
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
    except Exception as e:
        print(f"{name}: {type(tensor)}, Error: {e}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

color_words = set([
    'white', 'black', 'gray', 'grey', 'red', 'blue', 'green', 'yellow', 'orange', 'pink',
    'purple', 'brown', 'tan', 'silver', 'gold', 'beige', 'violet', 'cyan', 'magenta',
    "navy", "olive", "burgundy", "maroon", "teal", "lime", "indigo", "charcoal",
    "peach", "cream", "ivory", "turquoise", "mint", "mustard", "coral", "colorful"
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

def simple_caption_clean(
    caption, 
    number_words, 
    number_regex, 
    remove_phrases=None, 
    color_words=None
):
    c = str(caption).lower()
    c = c.translate(str.maketrans('', '', string.punctuation))
    c = number_regex.sub(' ', c)
    c = ' '.join([w for w in c.split() if w not in number_words])
    c = re.sub(r'\s+', ' ', c).strip()

    # ngram 제거(색상 단어 포함된 phrase는 남김)
    if remove_phrases and color_words:
        non_color_phrases = [
            p for p in remove_phrases if not any(color in p for color in color_words)
        ]
        for phrase in non_color_phrases:
            c = re.sub(r'[\s,.!?;:]*' + re.escape(phrase) + r'[\s,.!?;:]*', ' ', c)
        c = re.sub(r'\s+', ' ', c).strip()
    return c

def safe_prompt_str(prompt_str, tokenizer, max_len=77):
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=True, truncation=True,  max_length=max_len, return_tensors="pt")[0]
    prompt_str = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
        )
    return prompt_str

class PromptEnhancer:
    def __init__(self):
        self.fixed_tail = "preserve real tones, muted colors, natural, maintain structure, not oversaturated,"
        self.base_negative_prompts = "bad quality, grayscale, low contrast, unrealistic, color artifact," \
        "deformed, distorted, blurry, posterization, unnatural colors, unrealistic colors, desaturated, " \
        "underexposed, overexposed, " \
        "oversmooth, posterization"

    def get_enhancement_keywords(self, cleaned_caption):
        return [self.fixed_tail]

    def get_base_negative_prompt(self, cleaned_caption=None):
        return self.base_negative_prompts


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
            self.TYPE_OBJECT: ['car', 'bus', 'train', 'table', 'chair', 'cow', 'bowl', 'dog', 'cat', 'book', 'bottle', 'cup', 'food', 'flower', 'clock', 'sign', 'window', 'door']
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

def tensor_to_pil(tensor):
    tensor = tensor.cpu().float()
    # Normalize to [0, 1] if input is [-1, 1]
    tensor = (tensor + 1) / 2.0 if tensor.min() < 0 or tensor.max() > 1 else tensor
    tensor = torch.clamp(tensor, 0, 1) # Ensure values are within [0, 1]

    image_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

lpips_loss_fn = None

def ssim_loss(img1, img2, data_range=1.0, size_average=True):
    # SSIM은 [0, 1] 범위의 입력을 기대하므로 변환
    img1_normalized = (img1 + 1) / 2.0
    img2_normalized = (img2 + 1) / 2.0
    return 1 - msssim(img1_normalized.float(), img2_normalized.float(), data_range=data_range, size_average=size_average)

def get_clip_features(image_tensor, clip_processor, clip_model, accelerator_device, weight_dtype):
    # CLIP은 0-1 범위 또는 PIL 이미지 입력을 기대하므로 변환
    if image_tensor.ndim == 3:
        pil_list = [tensor_to_pil(image_tensor)] # tensor_to_pil이 [-1,1]을 [0,1]로 변환
    elif image_tensor.ndim == 4:
        pil_list = [tensor_to_pil(t) for t in image_tensor]
    else:
        raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

    inputs = clip_processor(images=pil_list, return_tensors="pt")
    inputs = inputs.to(accelerator_device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=weight_dtype)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True) # Normalize features

# --- Dataset Class ---
class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, dynamic, img_size=512):
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.enhancer = enhancer
        self.dynamic = dynamic
        self.img_size = img_size
        self.max_tokens = CFG.MAX_PROMPT_TOKENS
        self.nsfw_keywords = [k.lower() for k in CFG.NSFW_KEYWORDS]
        self.sfw_caption_replacement = CFG.SFW_CAPTION_REPLACEMENT

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cleaned_input_path_from_csv = os.path.normpath(row['input_img_path'])
        cleaned_gt_path_from_csv = os.path.normpath(row['gt_img_path'])
        input_image_path = os.path.join(self.input_dir, cleaned_input_path_from_csv)
        gt_image_path = os.path.join(self.gt_dir, cleaned_gt_path_from_csv)
        
        # 이미지 로드 및 RGB (3채널) 변환 확인
        original_input_pil = Image.open(input_image_path).convert("RGB")
        input_image_np = np.array(original_input_pil)
        gray_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)  # for canny

        raw_caption = str(row['caption'])
        cleaned_caption_raw = simple_caption_clean(raw_caption, number_words, number_regex)

        # NSFW 필터링
        is_nsfw = any(nsfw_kw in cleaned_caption_raw for nsfw_kw in self.nsfw_keywords)
        if is_nsfw:
            cleaned_caption = self.sfw_caption_replacement
        else:
            cleaned_caption = cleaned_caption_raw

        # Positive Prompt 구성
        pos_prompt_parts = [cleaned_caption]
        enhancement_keywords_list = self.enhancer.get_enhancement_keywords(cleaned_caption)
        for keyword_phrase in enhancement_keywords_list:
            temp_prompt = ", ".join(pos_prompt_parts + [keyword_phrase])
            temp_token_ids = self.tokenizer.encode(
                temp_prompt,
                add_special_tokens=True,
                truncation=True,
                return_tensors="pt"
            )[0]
            if len(temp_token_ids) <= self.max_tokens:
                pos_prompt_parts.append(keyword_phrase)
            else:
                break
        pos_prompt_str_raw = ", ".join(pos_prompt_parts)
        final_pos_prompt_str_for_pipe = safe_prompt_str(pos_prompt_str_raw, self.tokenizer, self.max_tokens)
        pos_tokenized_output = self.tokenizer(
            final_pos_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        final_pos_input_ids = pos_tokenized_output.input_ids[0]

        # --- Negative Prompt ---
        base_neg_prompt_str = self.enhancer.get_base_negative_prompt(cleaned_caption)
        final_neg_prompt_str_for_pipe = safe_prompt_str(base_neg_prompt_str, self.tokenizer, self.max_tokens)
        neg_tokenized_output = self.tokenizer(
            final_neg_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        final_neg_input_ids = neg_tokenized_output.input_ids[0]
        # Canny 이미지 생성 및 정규화
        canny_low, canny_high = self.dynamic.get_optimal_canny_params(cleaned_caption)
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil)  # [-1, 1] normalized

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_rgb_tensor = self.transform(gt_image_pil)  # [-1, 1] normalized

        guidance = self.dynamic.get_optimal_guidance(cleaned_caption)
        steps = self.dynamic.get_optimal_steps(cleaned_caption)

        return {
            "conditioning_pixel_values": input_control_image,
            "gt_rgb_tensor": gt_rgb_tensor,
            "caption": raw_caption,
            "cleaned_caption_raw": cleaned_caption,
            "pos_prompt_input_ids": final_pos_input_ids,
            "neg_prompt_input_ids": final_neg_input_ids,
            "pos_prompt_str_for_pipe": final_pos_prompt_str_for_pipe,
            "neg_prompt_str_for_pipe": final_neg_prompt_str_for_pipe,
            "guidance": guidance,
            "steps": steps,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "file_name": os.path.basename(cleaned_input_path_from_csv)
        }

def collate_fn(examples):
    pixel_values = torch.stack([example["gt_rgb_tensor"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    pos_prompt_input_ids = torch.stack([example["pos_prompt_input_ids"] for example in examples])
    neg_prompt_input_ids = torch.stack([example["neg_prompt_input_ids"] for example in examples])

    pos_prompt_str_for_pipe = [str(example["pos_prompt_str_for_pipe"]) for example in examples]
    neg_prompt_str_for_pipe = [str(example["neg_prompt_str_for_pipe"]) for example in examples]

    guidance_scales = torch.tensor([example["guidance"] for example in examples])
    num_inference_steps = torch.tensor([example["steps"] for example in examples])

    return {
        "pixel_values": pixel_values,  # GT image ([-1, 1])
        "conditioning_pixel_values": conditioning_pixel_values, # Canny image ([-1, 1])
        "pos_prompt_input_ids": pos_prompt_input_ids,
        "neg_prompt_input_ids": neg_prompt_input_ids,
        "pos_prompt_str_for_pipe": pos_prompt_str_for_pipe,
        "neg_prompt_str_for_pipe": neg_prompt_str_for_pipe,
        "guidance_scales": guidance_scales,
        "num_inference_steps": num_inference_steps,
        "captions": [example["caption"] for example in examples],
        "file_names": [example["file_name"] for example in examples],
    }

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    set_seed(worker_seed) # Use the custom set_seed function

def save_lora_model_overwriting(model_dict, dir_path, subfolder_unet="unet_lora", is_main_process=True):
    if is_main_process:
        abs_dir_path = os.path.abspath(dir_path)

        # UNet LoRA 저장
        unet_lora_path = os.path.join(abs_dir_path, subfolder_unet)
        if os.path.exists(unet_lora_path):
            shutil.rmtree(unet_lora_path)
        os.makedirs(unet_lora_path, exist_ok=True)

        if isinstance(model_dict['unet'], PeftModel):
            model_dict['unet'].save_pretrained(unet_lora_path)
            print(f"UNet LoRA saved to {unet_lora_path}")
        else:
            print(f"Warning: model_dict['unet'] is not a PeftModel. Skipping UNet LoRA saving.")

def get_peft_leaf_model(m):
    if hasattr(m, "base_model") and isinstance(m.base_model, torch.nn.Module):
        return get_peft_leaf_model(m.base_model)
    return m

# --- Validation Function ---
@torch.no_grad()
def run_validation(
    pipeline,
    accelerator,
    epoch,
    train_global_step,
    val_dataloader,
    clip_processor,
    clip_model,
    lpips_loss_fn,
    weight_dtype,
    output_dir,
    num_samples_to_save,
):
    print("\nRunning validation...")
    pipeline.unet.eval()
    pipeline.controlnet.eval()

    val_output_dir = os.path.join(output_dir, "validation_samples", f"step_{train_global_step}")
    os.makedirs(val_output_dir, exist_ok=True)

    total_l1_loss = 0.0
    total_clip_loss = 0.0
    total_lpips_loss = 0.0
    total_ssim_loss = 0.0

    val_dataset = val_dataloader.dataset
    val_len = len(val_dataset)
    sample_indices = random.sample(range(val_len), min(num_samples_to_save, val_len))

    for count, idx in enumerate(sample_indices):
        example = val_dataset[idx]

        conditioning_image_pil = tensor_to_pil(example["conditioning_pixel_values"])
        gt_rgb_tensor = example["gt_rgb_tensor"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

        pos_prompt_str = example["pos_prompt_str_for_pipe"]
        neg_prompt_str = example["neg_prompt_str_for_pipe"]
        guidance = example["guidance"]
        steps = int(example["steps"])
        file_name = example["file_name"]
        caption = example["caption"]

        print(f"\n--- Validation Sample {count+1} ---")
        print(f" Positive Prompt: {pos_prompt_str}")
        print(f" Negative Prompt: {neg_prompt_str}")
        print(f" Guidance Scale: {guidance:.2f}")
        print(f" Inference Steps: {steps}")
        print(f" Original Caption: {caption}")
        print(f" File Name: {file_name}")

        # 이미지 생성
        generated_images = pipeline(
            prompt=pos_prompt_str,
            negative_prompt=neg_prompt_str,
            image=conditioning_image_pil,
            num_inference_steps=steps,
            guidance_scale=guidance,
            output_type="pil",
        ).images

        gen_pil_image = generated_images[0]
        gen_tensor_image = transforms.ToTensor()(gen_pil_image)
        gen_tensor_image = (gen_tensor_image * 2.0) - 1.0
        gen_tensor_image = gen_tensor_image.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

        # 손실 계산
        l1 = F.l1_loss(gen_tensor_image, gt_rgb_tensor)
        clip_features_gen = get_clip_features(gen_tensor_image, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_features_gt = get_clip_features(gt_rgb_tensor, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_loss = 1 - F.cosine_similarity(clip_features_gen, clip_features_gt).mean()

        lpips_val = lpips_loss_fn(
            ((gen_tensor_image + 1) / 2.0).to(accelerator.device),
            ((gt_rgb_tensor + 1) / 2.0).to(accelerator.device)
        ).mean()

        ssim_val = ssim_loss(gen_tensor_image, gt_rgb_tensor)

        total_l1_loss += l1.item()
        total_clip_loss += clip_loss.item()
        total_lpips_loss += lpips_val.item()
        total_ssim_loss += ssim_val.item()

        # 생성 이미지 저장
        output_filename = os.path.join(val_output_dir, f"sample_{count+1}_{file_name}")
        gen_pil_image.save(output_filename)

        # Canny 및 GT 이미지 비교를 위해 저장
        conditioning_image_pil.save(os.path.join(val_output_dir, f"sample_{count+1}_{file_name}_canny.png"))
        tensor_to_pil(gt_rgb_tensor.squeeze(0)).save(os.path.join(val_output_dir, f"sample_{count+1}_{file_name}_gt.png"))

    count = len(sample_indices)
    avg_l1_loss = total_l1_loss / count
    avg_clip_loss = total_clip_loss / count
    avg_lpips_loss = total_lpips_loss / count
    avg_ssim_loss = total_ssim_loss / count

    log_message = (
        f"Validation Results (Epoch {epoch}, Global Step {train_global_step}):\n"
        f"   Average L1 Loss: {avg_l1_loss:.4f}\n"
        f"   Average CLIP Loss: {avg_clip_loss:.4f}\n"
        f"   Average LPIPS Loss: {avg_lpips_loss:.4f}\n"
        f"   Average SSIM Loss: {avg_ssim_loss:.4f}"
    )
    print(log_message)

    avg_combined_val_loss = (CFG.LAMBDA_L1 * avg_l1_loss +
                             CFG.LAMBDA_CLIP * avg_clip_loss +
                             CFG.LAMBDA_LPIPS * avg_lpips_loss +
                             CFG.LAMBDA_SSIM * avg_ssim_loss)

    accelerator.log({
        "val_avg_l1_loss": avg_l1_loss,
        "val_avg_clip_loss": avg_clip_loss,
        "val_avg_lpips_loss": avg_lpips_loss,
        "val_avg_ssim_loss": avg_ssim_loss,
        "val_avg_combined_loss": avg_combined_val_loss,
    }, step=train_global_step)

    return avg_combined_val_loss

# --- Main Training Loop ---
def train_loop(
    pretrained_model_name_or_path: str,
    controlnet_path: str,
    output_dir: str,
    train_data_df: pd.DataFrame,
    cfg: Config,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=cfg.MIXED_PRECISION,
        log_with=cfg.REPORT_TO,
        project_dir=os.path.join(output_dir, cfg.PROJECT_NAME),
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers(cfg.PROJECT_NAME, config=filter_config_types(vars(cfg)))

    # 1. 모델 및 토크나이저 로드
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    
    # UNet은 LoRA 적용 
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_path)

    # CLIP 모델 로드 
    clip_processor = CLIPProcessor.from_pretrained(cfg.CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(cfg.CLIP_MODEL)
    clip_model.eval() # 평가 모드로 설정
    for param in clip_model.parameters(): # CLIP 모델 파라미터 고정
        param.requires_grad = False

    # LPIPS 손실 함수 초기화 
    global lpips_loss_fn
    lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    lpips_loss_fn.eval() # 평가 모드로 설정
    for param in lpips_loss_fn.parameters(): # LPIPS 모델 파라미터 고정
        param.requires_grad = False

    # 모델의 데이터 타입 설정 (mixed_precision 고려)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # VAE, Text Encoder, ControlNet 고정 (ControlNet은 학습 대상이 아님)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False) # ControlNet 파라미터 고정

    # UNet에만 LoRA 적용
    unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    # 옵티마이저 설정 (UNet LoRA 파라미터만 학습)
    params_to_optimize = list(unet.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=cfg.LR,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        weight_decay=cfg.ADAM_WEIGHT_DECAY,
        eps=cfg.ADAM_EPSILON,
    )

    # 스케줄러 설정
    lr_scheduler = get_scheduler(
        cfg.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=cfg.LR_WARMUP_STEPS * cfg.GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=cfg.MAX_TRAIN_STEPS if cfg.MAX_TRAIN_STEPS is not None else (len(train_data_df) // cfg.BATCH_SIZE) * cfg.EPOCHS,
    )

    # 노이즈 스케줄러 설정
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # 데이터셋 및 데이터로더 설정
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # 이미지를 [-1, 1]로 정규화
    ])
    
    # train_data_df를 학습/검증 세트로 분할
    train_df, val_df = train_test_split(train_data_df, test_size=0.1, random_state=cfg.SEED) # 10% 검증 세트
    if cfg.MAX_DATA is not None:
        train_df = train_df.head(cfg.MAX_DATA)
        val_df = val_df.head(cfg.MAX_DATA // 10 if cfg.MAX_DATA // 10 > 0 else 1) # 검증 데이터도 적절히 제한

    enhancer = PromptEnhancer()
    dynamic_param_gen = DynamicParameterGenerator()

    train_dataset = ColorizationDataset(
        df=train_df,
        input_dir=cfg.INPUT_DIR,
        gt_dir=cfg.GT_DIR,
        transform=transform,
        tokenizer=tokenizer,
        enhancer=enhancer,
        dynamic=dynamic_param_gen,
        img_size=cfg.IMG_SIZE
    )
    val_dataset = ColorizationDataset(
        df=val_df,
        input_dir=cfg.INPUT_DIR,
        gt_dir=cfg.GT_DIR,
        transform=transform,
        tokenizer=tokenizer,
        enhancer=enhancer,
        dynamic=dynamic_param_gen,
        img_size=cfg.IMG_SIZE
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False, 
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Accelerator로 준비
    unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare( 
        unet, optimizer, train_dataloader, lr_scheduler, val_dataloader
    )


    # ControlNet은 학습되지 않으므로 prepare 대신 직접 to device
    controlnet.to(accelerator.device, dtype=weight_dtype)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)
    lpips_loss_fn.to(accelerator.device) # LPIPS는 float32 고정

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=get_peft_leaf_model(unet) if isinstance(unet, PeftModel) else unet, # UNet은 LoRA 적용
        controlnet=controlnet, # ControlNet은 원본 모델 사용
        scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, 
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.to(accelerator.device, dtype=weight_dtype)
    # 이미지 입력 체크를 건너뛰어 Canny 이미지 형식에 유연하게 대응
    pipeline.check_inputs = lambda *args, **kwargs: None 

    # --- RESUME FROM CHECKPOINT ---
    global_step = 0
    first_epoch = 0
    if cfg.RESUME_FROM_CHECKPOINT:
        if cfg.RESUME_FROM_CHECKPOINT != "latest":
            path = cfg.RESUME_FROM_CHECKPOINT
        else:
            # 'latest' 체크포인트를 찾기 위해 가장 최근의 체크포인트 폴더를 찾습니다.
            all_checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if not all_checkpoints:
                raise ValueError("No checkpoints found to resume from 'latest'.")
            all_checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            path = os.path.join(output_dir, all_checkpoints[-1])
            print(f"Resuming from latest checkpoint: {path}")
        
        # LoRA 모델 상태 사전 로드
        unet_lora_path = os.path.join(path, "unet_lora", "adapter_model.safetensors")

        if os.path.exists(unet_lora_path):
            unet_lora_state_dict = load_file(unet_lora_path)
            # PEFT 모델의 경우 set_peft_model_state_dict를 사용
            if isinstance(unet, PeftModel):
                set_peft_model_state_dict(unet, unet_lora_state_dict)
                print(f"Loaded UNet LoRA from {unet_lora_path}")
            else:
                print(f"Warning: UNet is not a PeftModel. Attempting to load state_dict directly.")
                unet.load_state_dict(unet_lora_state_dict)
        else:
            print(f"UNet LoRA adapter_model.safetensors not found in {os.path.join(path, 'unet_lora')}. Skipping UNet LoRA load.")

        # Accelerator의 전체 학습 상태 로드 (옵티마이저, 스케줄러, 글로벌 스텝 등)
        accelerator.load_state(path)

        # --- global_step은 직접 복원 ---
        global_step_path = os.path.join(path, "global_step.pt")
        if os.path.exists(global_step_path):
            global_step = torch.load(global_step_path)
        else:
            global_step = 0
        first_epoch = global_step // len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Resumed training state from {path}, starting at global_step {global_step}, epoch {first_epoch}")

    # 총 학습 스텝 수 계산 (재개 시에도 유효하도록)
    if cfg.MAX_TRAIN_STEPS is None:
        cfg.MAX_TRAIN_STEPS = cfg.EPOCHS * len(train_dataloader)
    
    # 학습 시작
    total_batch_size = cfg.BATCH_SIZE * accelerator.num_processes * cfg.GRADIENT_ACCUMULATION_STEPS
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num epochs = {cfg.EPOCHS}")
    print(f"  Instantaneous batch size per device = {cfg.BATCH_SIZE}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Total optimization steps = {cfg.MAX_TRAIN_STEPS}")

    progress_bar = tqdm(
        range(global_step, cfg.MAX_TRAIN_STEPS),
        disable=not accelerator.is_main_process,
        initial=global_step # Restore progress bar to correct initial step
    )
    progress_bar.set_description("Steps")

    # Early Stopping 및 Best Model Saving을 위한 변수
    best_combined_val_loss = float('inf') 
    epochs_no_improve = 0 # 기능적으로 intervals_no_improve임 바꾸기 귀차낭

    for epoch in range(first_epoch, cfg.EPOCHS):
        unet.train()
        controlnet.eval() 
        train_loss_this_interval = 0.0

        for step, batch in enumerate(train_dataloader):
            # 스텝이 이미 MAX_TRAIN_STEPS에 도달했는지 확인
            if global_step >= cfg.MAX_TRAIN_STEPS:
                break
                
            with accelerator.accumulate(unet): # UNet만 accumulate
                # 텍스트 임베딩 생성
                encoder_hidden_states_pos = text_encoder(batch["pos_prompt_input_ids"])[0]
                
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 노이즈 샘플링 및 노이즈 추가
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
            
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ControlNet 조건 이미지 준비
                # conditioning_pixel_values는 [-1, 1] 범위 (transforms.Normalize에 의해)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # ControlNet forward (ControlNet은 항상 평가 모드에서 실행)
                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states_pos, # Positive prompt for ControlNet
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )
                
                # UNet forward (ControlNet의 출력과 함께)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states_pos, # Positive prompt for UNet
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # 손실 계산 (Diffusion Loss)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Diffusion Loss (MSE)
                loss_diffusion = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                with torch.no_grad(): 
                    decoded_latents = 1 / vae.config.scaling_factor * model_pred
                    generated_image = vae.decode(decoded_latents.float()).sample 
                    generated_image = generated_image.clamp(-1, 1) # VAE 출력 클램핑: 명확히 [-1, 1] 범위 유지

                # 이미지 품질 손실 계산
                gt_pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                l1_loss = F.l1_loss(generated_image, gt_pixel_values)

                # CLIP 손실: 내부적으로 [-1,1] -> [0,1] 변환 수행
                clip_features_gen = get_clip_features(generated_image, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_features_gt = get_clip_features(gt_pixel_values, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_loss = 1 - F.cosine_similarity(clip_features_gen, clip_features_gt).mean()

                # LPIPS는 0~1 범위의 입력을 기대하므로 명시적으로 변환
                lpips_loss = lpips_loss_fn(
                    ((generated_image + 1) / 2.0).to(accelerator.device), # [-1, 1] -> [0, 1] 변환
                    ((gt_pixel_values + 1) / 2.0).to(accelerator.device) # [-1, 1] -> [0, 1] 변환
                ).mean()
                
                # SSIM도 0~1 범위의 입력을 기대 (ssim_loss 함수 내부에서 처리)
                ssim_val_loss = ssim_loss(generated_image, gt_pixel_values)


                # 최종 손실
                total_loss = (cfg.LAMBDA_L1 * l1_loss + 
                              cfg.LAMBDA_CLIP * clip_loss + 
                              cfg.LAMBDA_LPIPS * lpips_loss + 
                              cfg.LAMBDA_SSIM * ssim_val_loss +
                              loss_diffusion)

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, cfg.MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss_this_interval += total_loss.item()
                
                # 로깅
                if global_step % cfg.LOG_INTERVAL == 0:
                    # 현재 스텝까지의 평균 손실 계산
                    avg_train_loss = train_loss_this_interval / (cfg.LOG_INTERVAL * accelerator.gradient_accumulation_steps) 
                    accelerator.log({
                        "train_loss": avg_train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "l1_loss": l1_loss.item(),
                        "clip_loss": clip_loss.item(),
                        "lpips_loss": lpips_loss.item(),
                        "ssim_loss": ssim_val_loss.item(),
                        "diffusion_loss": loss_diffusion.item(),
                    }, step=global_step)
                    train_loss_this_interval = 0.0 

                # 검증 및 샘플 저장
                if (global_step % cfg.SAVE_AND_VAL_INTERVAL == 0 or global_step == cfg.MAX_TRAIN_STEPS) and global_step > 0:
                    if accelerator.is_main_process:
                        # 파이프라인에 현재 학습된 LoRA 모델을 연결
                        pipeline.unet = get_peft_leaf_model(accelerator.unwrap_model(unet)) # UNet은 LoRA 적용
                        pipeline.controlnet = controlnet # ControlNet은 원본 모델 사용
                        pipeline.vae = vae
                        pipeline.text_encoder = text_encoder
                        pipeline.to(accelerator.device, dtype=weight_dtype)

                        current_combined_val_loss = run_validation(
                            pipeline,
                            accelerator,
                            epoch,
                            global_step,
                            val_dataloader,
                            clip_processor,
                            clip_model,
                            lpips_loss_fn, 
                            weight_dtype,
                            output_dir,
                            cfg.NUM_SAMPLES_TO_SAVE,
                        )
                        accelerator.wait_for_everyone()

                        # Early Stopping & Best Model Saving
                        if current_combined_val_loss < best_combined_val_loss:
                            best_combined_val_loss = current_combined_val_loss
                            epochs_no_improve = 0 # 개선되었으니 카운트 리셋
                            print(f"New best validation combined loss: {best_combined_val_loss:.4f}. Saving best model.")
                            accelerator.wait_for_everyone() 
                            save_lora_model_overwriting(
                                {"unet": unet}, # Wrapped UNet only
                                os.path.join(output_dir, "best_model"),
                                is_main_process=accelerator.is_main_process,
                            )
                        else:
                            epochs_no_improve += 1
                            print(f"Validation combined loss did not improve. Intervals without improvement: {epochs_no_improve}")
                            if epochs_no_improve >= cfg.PATIENCE:
                                print(f"Early stopping triggered after {cfg.PATIENCE} intervals without improvement based on combined loss.")
                                break # Break from inner loop, will lead to breaking outer loop

                        # Epoch-based checkpoint saving (using global_step for folder naming)
                        output_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(output_checkpoint_dir)
                        save_lora_model_overwriting(
                            {"unet": unet},
                            output_checkpoint_dir,
                            is_main_process=accelerator.is_main_process,
                        )
                        if accelerator.is_main_process:
                            torch.save(global_step, os.path.join(output_checkpoint_dir, "global_step.pt"))

                        # Remove old checkpoints
                        all_checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")])
                        if len(all_checkpoints) > cfg.MAX_CHECKPOINTS_TO_KEEP:
                            num_to_remove = len(all_checkpoints) - cfg.MAX_CHECKPOINTS_TO_KEEP
                            for old_checkpoint in all_checkpoints[:num_to_remove]:
                                shutil.rmtree(os.path.join(output_dir, old_checkpoint))
                                print(f"Removed old checkpoint: {old_checkpoint}")
                        
                        # Set models back to training mode
                        unet.train()
                        controlnet.eval()
                        vae.to(accelerator.device, dtype=weight_dtype)
                        text_encoder.to(accelerator.device, dtype=weight_dtype)
                        clip_model.to(accelerator.device, dtype=weight_dtype)
                        lpips_loss_fn.to(accelerator.device)

            if global_step >= cfg.MAX_TRAIN_STEPS:
                break
        
        if global_step >= cfg.MAX_TRAIN_STEPS or epochs_no_improve >= cfg.PATIENCE:
            break

    # 학습 종료 후 최종 모델 저장
    if accelerator.is_main_process:
        print("Training finished. Saving final model.")
        save_lora_model_overwriting(
            {"unet": unet}, # Final UNet LoRA only
            os.path.join(output_dir, "final_model"),
            is_main_process=accelerator.is_main_process,
        )
    
    accelerator.end_training()

    del unet, controlnet, vae, text_encoder, tokenizer, optimizer, lr_scheduler
    del train_dataloader, train_dataset, val_dataset
    del clip_processor, clip_model, lpips_loss_fn, pipeline
    gc.collect()
    torch.cuda.empty_cache()

# Main execution block
if __name__ == "__main__":
    set_seed(CFG.SEED) 
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    
    # train.csv 파일 로드
    if not os.path.exists(CFG.TRAIN_CSV):
        raise FileNotFoundError(f"Error: {CFG.TRAIN_CSV} not found. Please ensure your train.csv is in the specified path.")
    
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    print("Starting training loop...")
    train_loop(
        pretrained_model_name_or_path=CFG.PRETRAINED_MODEL_NAME_OR_PATH,
        controlnet_path=CFG.CONTROLNET_PATH,
        output_dir=CFG.OUTPUT_DIR,
        train_data_df=train_df,
        cfg=CFG,
    )