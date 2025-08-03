# 추론시 merge_adapter_weights 까먹지 마셈 동적파라미터 원본캡션
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
import webcolors
from sklearn.cluster import KMeans
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
        self.GUIDANCE_SCALE = 7.5 # 고정된 값
        self.NUM_INFERENCE_STEPS = 35 # 고정된 값
        self.SEED = 42
        self.OUTPUT_DIR = "./output7"
        self.TRAIN_CSV = "../train.csv"
        self.INPUT_DIR = ".."
        self.GT_DIR = ".."
        self.LR = 1e-4
        self.BATCH_SIZE = 10
        self.NUM_WORKERS = 2
        self.EPOCHS = 30
        self.MAX_DATA = None
        self.LAMBDA_L1 = 2.0
        self.LAMBDA_CLIP = 0.0
        self.LAMBDA_LPIPS = 0.5
        self.LAMBDA_SSIM = 0.5
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.PROJECT_NAME = "colorization_training"
        self.PATIENCE = 9999
        self.MAX_PROMPT_TOKENS = 77
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "erotic", "nude", "breast", "ass", "penis", "vagina"]
        self.SFW_CAPTION_REPLACEMENT = "a high quality image, realistic, clean, beautiful, bright, colorful" # NSFW 캡션 대체
        self.GRADIENT_ACCUMULATION_STEPS = 2
        self.MAX_GRAD_NORM = 1.0
        self.LR_SCHEDULER_TYPE = "constant"
        self.LR_WARMUP_STEPS = 0
        self.ADAM_BETA1 = 0.9
        self.ADAM_BETA2 = 0.999
        self.ADAM_WEIGHT_DECAY = 1e-2
        self.ADAM_EPSILON = 1e-08
        self.MIXED_PRECISION = "bf16" # "no", "fp16", "bf16"
        self.REPORT_TO = "tensorboard" # "tensorboard", "wandb", "all"
        self.MAX_TRAIN_STEPS = 100 # 총 학습 스텝 수 (None이면 EPOCHS로 계산)
        self.RESUME_FROM_CHECKPOINT = "" # "./output5/checkpoint-40"
        self.SAMPLE_SAVE_START_STEP = 5 # 샘플 이미지 저장 시작 스텝
        self.SAMPLE_SAVE_END_STEP = 100 # 샘플 이미지 저장 종료 스텝
        self.NUM_SAMPLES_TO_SAVE = None # 손실 평가에 실제로 사용되는 검증 이미지 개수
        self.MAX_CHECKPOINTS_TO_KEEP = 30 # 유지할 체크포인트 최대 개수
        self.LOG_INTERVAL = 10 # 10 스텝마다 로깅
        self.VAL_INTERVAL = 1 # 1 에폭마다 검증 (SAVE_AND_VAL_INTERVAL로 대체됨)
        self.CONTROLNET_STRENGTH = 1.0
        self.SAVE_AND_VAL_INTERVAL = 5
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
    "peach", "cream", 'ivory', 'turquoise', 'mint', 'mustard', 'coral', 'colorful'
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
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=True, truncation=True, max_length=max_len, return_tensors="pt")[0]
    prompt_str = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
        )
    return prompt_str

def extract_dominant_colors(image, topk=3):
    img = image.resize((32,32)).convert('RGB')
    arr = np.array(img).reshape(-1,3)
    kmeans = KMeans(n_clusters=topk, n_init='auto').fit(arr)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

def rgb_to_simple_color_name(rgb_tuple):
    min_dist = float("inf")
    closest_name = None
    for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        dist = (r_c - rgb_tuple[0])**2 + (g_c - rgb_tuple[1])**2 + (b_c - rgb_tuple[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

class PromptEnhancer:
    def __init__(self):
        self.fixed_tail_template = (
            "high detail, neutral tone, photorealistic, real people, sharp, natural color, original color, actual person,"
            "preserve structure, balanced tone, realistic coloration, unexaggerated colors, not oversaturated"
        )
        self.base_negative_prompts = (
            "bad quality, vivid, uncanny, vibrant, sketch, monochrome, grayscale, low detail, deformed, distorted, "
            "missing face, blurry, overexposed, oversaturated, too bright, high saturation, mutated hands, low contrast, artificial, neon, "
            "unrealistic, burnt, posterization, color artifact, noisy"
        )
    def get_enhancement_keywords(self, cleaned_caption):
        return [self.fixed_tail_template]  

    def get_base_negative_prompt(self, cleaned_caption=None):
        return self.base_negative_prompts

def tensor_to_pil(tensor):
    pil_images = []
    if tensor.dim() == 4:
        # 배치 처리
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i].cpu().float()
            img_tensor = (img_tensor + 1) / 2.0 if img_tensor.min() < 0 or img_tensor.max() > 1 else img_tensor
            img_tensor = torch.clamp(img_tensor, 0, 1)
            image_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(image_np))
    elif tensor.dim() == 3:
        # 단일 이미지 처리 (기존 동작 유지)
        img_tensor = tensor.cpu().float()
        img_tensor = (img_tensor + 1) / 2.0 if img_tensor.min() < 0 or img_tensor.max() > 1 else img_tensor
        img_tensor = torch.clamp(img_tensor, 0, 1)
        image_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(image_np))
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Expected 3 or 4.")
    return pil_images if len(pil_images) > 1 else pil_images[0]


def ssim_loss(img1, img2, data_range=2.0, size_average=True):
    return 1 - msssim(img1.float(), img2.float(), data_range=data_range, size_average=size_average)

def get_clip_features(image_tensor, clip_processor, clip_model, accelerator_device, weight_dtype):
    pil_list = tensor_to_pil(image_tensor)
    if not isinstance(pil_list, list):
        pil_list = [pil_list]

    inputs = clip_processor(images=pil_list, return_tensors="pt")
    inputs = {k: v.to(accelerator_device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=weight_dtype)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)

# --- Dataset Class ---
class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, img_size=512):
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.enhancer = enhancer
        self.img_size = img_size
        self.max_tokens = CFG.MAX_PROMPT_TOKENS
        self.nsfw_keywords = [k.lower() for k in CFG.NSFW_KEYWORDS]
        self.sfw_caption_replacement = CFG.SFW_CAPTION_REPLACEMENT
        self.printed_count = 0 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cleaned_input_path_from_csv = os.path.normpath(row['input_img_path'])
        cleaned_gt_path_from_csv = os.path.normpath(row['gt_img_path'])
        input_image_path = os.path.join(self.input_dir, cleaned_input_path_from_csv)
        gt_image_path = os.path.join(self.gt_dir, cleaned_gt_path_from_csv)

        original_input_pil = Image.open(input_image_path).convert("RGB")
        input_image_np = np.array(original_input_pil)
        gray_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)

        raw_caption = str(row['caption'])
        cleaned_caption_raw = simple_caption_clean(raw_caption, number_words, number_regex)

        is_nsfw = any(nsfw_kw in cleaned_caption_raw for nsfw_kw in self.nsfw_keywords)
        if is_nsfw:
            cleaned_caption = self.sfw_caption_replacement
        else:
            cleaned_caption = cleaned_caption_raw

        # dominant color name 추출
        dominant_colors = extract_dominant_colors(original_input_pil, topk=3)
        color_names = [rgb_to_simple_color_name(c) for c in dominant_colors]
        color_names = list(dict.fromkeys(color_names))[:3]
        color_str = ', '.join(color_names)

        # 프롬프트에 dominant color name 삽입
        pos_prompt_parts = [color_str, cleaned_caption]  # 색상명을 맨 앞에 추가
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

        if self.printed_count < 2:
            print(f"[프롬프트 샘플 {self.printed_count+1}] dominant colors: {color_names} | caption: {cleaned_caption}")
            print(f"[프롬프트 전체] {final_pos_prompt_str_for_pipe}\n")
            self.printed_count += 1


        pos_tokenized_output = self.tokenizer(
            final_pos_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        final_pos_input_ids = pos_tokenized_output.input_ids[0]

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
        canny_low, canny_high = 50, 150
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil)

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_rgb_tensor = self.transform(gt_image_pil)

        # Config에서 고정된 guidance와 steps 값 사용
        guidance = CFG.GUIDANCE_SCALE
        steps = CFG.NUM_INFERENCE_STEPS

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
    num_inference_steps = examples[0]["steps"]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
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
    set_seed(worker_seed)

def save_lora_model_overwriting(model_dict, dir_path, subfolder_unet="unet_lora", is_main_process=True):
    if is_main_process:
        abs_dir_path = os.path.abspath(dir_path)

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

# --- save_latest_checkpoint 함수 추가 ---
def save_latest_checkpoint(unet, output_dir, global_step, accelerator, cfg):
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        
        # save_lora_model_overwriting 함수를 사용하여 UNet LoRA 모델 저장
        save_lora_model_overwriting({"unet": unet}, ckpt_dir, is_main_process=True)
        
        # accelerator 상태 (옵티마이저, 스케줄러 등) 저장
        # accelerator.save_state(ckpt_dir)
        
        # global_step 저장
        # torch.save(global_step, os.path.join(ckpt_dir, "global_step.pt"))
        # print(f"Checkpoint saved to {ckpt_dir} at step {global_step}")

        # 오래된 체크포인트 삭제
        # ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))], 
        #                  key=lambda x: int(x.split('-')[-1]))
        # if len(ckpts) > cfg.MAX_CHECKPOINTS_TO_KEEP:
        #     for i in range(len(ckpts) - cfg.MAX_CHECKPOINTS_TO_KEEP):
        #         old_ckpt_path = os.path.join(output_dir, ckpts[i])
        #         print(f"Removing old checkpoint: {old_ckpt_path}")
        #         shutil.rmtree(old_ckpt_path, ignore_errors=True)

lpips_loss_fn = None  # 전역 변수 정의

@torch.no_grad()
def run_validation(pipeline,
                   accelerator,
                   epoch,
                   global_step,
                   val_dataloader,
                   clip_processor,
                   clip_model,
                   weight_dtype,
                   output_dir,
                   num_samples_to_save
                   ):
    global lpips_loss_fn  # 함수 본문 "첫줄"에 선언

    if lpips_loss_fn is None:
        try:
            lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
            lpips_loss_fn.eval() # LPIPS 모델 학습 방지
            for param in lpips_loss_fn.parameters(): # 파라미터 업데이트 방지
                param.requires_grad = False
            print("LPIPS model initialized and moved to device.")
        except ImportError:
            print("Warning: LPIPS library not found. LPIPS loss will be skipped.")
            lpips_loss_fn = "skipped"

    print("\nRunning validation...")
    pipeline.unet.eval()
    pipeline.controlnet.eval()
    # pipeline.vae.eval() # 사용자님의 기존 코드에 없었으므로 제거
    # pipeline.text_encoder.eval() # 사용자님의 기존 코드에 없었으므로 제거

    val_output_dir = os.path.join(output_dir, "validation_samples", f"step_{global_step}")
    os.makedirs(val_output_dir, exist_ok=True)

    total_l1_loss = 0.0
    total_clip_loss = 0.0
    total_lpips_loss = 0.0
    total_ssim_loss = 0.0
    # Diffusion Loss는 run_validation에서 파이프라인으로 생성된 최종 이미지에 대한 손실이 아니므로 여기서는 계산하지 않음
    num_processed_samples = 0

    batches_to_process = 0
    if num_samples_to_save is not None:
        batches_to_process = math.ceil(num_samples_to_save / CFG.BATCH_SIZE)

    for i, batch in enumerate(val_dataloader):
        if num_samples_to_save is not None and i >= batches_to_process:
            break

        conditioning_images_pil_list = tensor_to_pil(batch["conditioning_pixel_values"])
        gt_rgb_tensors = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

        pos_prompt_strs = batch["pos_prompt_str_for_pipe"]
        neg_prompt_strs = batch["neg_prompt_str_for_pipe"]

        num_inference_steps = batch["num_inference_steps"]
        file_names = batch["file_names"]
        captions = batch["captions"]

        current_batch_size = gt_rgb_tensors.shape[0]

        print(f"\n--- Validation Batch {i+1} (Size: {current_batch_size}) ---")
        
        # pipeline을 통해 이미지 생성
        images = pipeline(
            image=conditioning_images_pil_list,
            prompt=pos_prompt_strs,
            negative_prompt=neg_prompt_strs,
            guidance_scale=CFG.GUIDANCE_SCALE,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=CFG.CONTROLNET_STRENGTH,
            output_type="pt", # 텐서 출력을 위해 "pt" 명시
        ).images.to(accelerator.device) # 파이프라인 출력은 이미 device에 있을 수 있지만, 명시적으로 이동

        # Losses 계산 (Diffusion Loss 제외, 왜냐하면 pipeline은 최종 이미지를 바로 생성하기 때문)
        l1_loss = F.l1_loss(images, gt_rgb_tensors, reduction='mean')
        total_l1_loss += l1_loss.item() * current_batch_size

        clip_features_generated = get_clip_features(images, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_features_gt = get_clip_features(gt_rgb_tensors, clip_processor, clip_model, accelerator.device, weight_dtype)

        clip_loss = 1 - F.cosine_similarity(clip_features_generated, clip_features_gt, dim=-1).mean()
        total_clip_loss += clip_loss.item() * current_batch_size

        if lpips_loss_fn != "skipped":
            lpips_loss = lpips_loss_fn(
                ((images + 1) / 2.0).to(accelerator.device), # -1~1 -> 0~1
                ((gt_rgb_tensors + 1) / 2.0).to(accelerator.device) # -1~1 -> 0~1
            ).mean()
            total_lpips_loss += lpips_loss.item() * current_batch_size
        else:
            lpips_loss = torch.tensor(0.0)

        ssim_val = ssim_loss(images, gt_rgb_tensors, data_range=2.0, size_average=True)
        total_ssim_loss += ssim_val.item() * current_batch_size

        # Save sample images
        num_samples_to_save = 3  # 대표 저장 개수(컨피그에서 받아도 됨)
        save_count = 0           # 저장한 샘플 수

        for j in range(current_batch_size):
            if save_count < num_samples_to_save:
                generated_img_pil = tensor_to_pil(images[j:j+1])
                gt_img_pil = tensor_to_pil(gt_rgb_tensors[j:j+1])
                canny_img_pil = conditioning_images_pil_list[j]

                combined_width = canny_img_pil.width + generated_img_pil.width + gt_img_pil.width
                combined_height = canny_img_pil.height
                combined_img = Image.new('RGB', (combined_width, combined_height))

                combined_img.paste(canny_img_pil, (0, 0))
                combined_img.paste(generated_img_pil, (canny_img_pil.width, 0))
                combined_img.paste(gt_img_pil, (canny_img_pil.width + generated_img_pil.width, 0))

                save_path = os.path.join(val_output_dir, f"step_{global_step}_sample_{num_processed_samples + j}_{file_names[j]}.png")
                combined_img.save(save_path)
                save_count += 1
            # else: 저장 안 함

        num_processed_samples += current_batch_size

    if num_processed_samples > 0:
        avg_l1_loss = total_l1_loss / num_processed_samples
        avg_clip_loss = total_clip_loss / num_processed_samples
        avg_lpips_loss = total_lpips_loss / num_processed_samples
        avg_ssim_loss = total_ssim_loss / num_processed_samples
    else:
        avg_l1_loss = avg_clip_loss = avg_lpips_loss = avg_ssim_loss = 0.0

    log_message = (
        f"Validation Results (Epoch {epoch}, Global Step {global_step}):\n"
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
    }, step=global_step)

    pipeline.unet.train()
    pipeline.controlnet.train()

    return avg_combined_val_loss


# --- Main Training Loop ---
def train_loop(
    pretrained_model_name_or_path: str,
    controlnet_path: str,
    output_dir: str,
    train_data_df: pd.DataFrame,
    cfg: Config,
):
    global lpips_loss_fn # 전역 변수 선언
    lpips_loss_fn = None

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=cfg.MIXED_PRECISION,
        log_with=cfg.REPORT_TO,
        project_dir=os.path.join(output_dir, cfg.PROJECT_NAME),
    )
    print(f"Accelerator mixed precision: {accelerator.state.mixed_precision}")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Resume config: {cfg.__dict__}")
        accelerator.init_trackers(cfg.PROJECT_NAME, config=filter_config_types(vars(cfg)))

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_path)
    clip_processor = CLIPProcessor.from_pretrained(cfg.CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(cfg.CLIP_MODEL)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    if lpips_loss_fn is None: 
        try:
            lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
            lpips_loss_fn.eval()
            for param in lpips_loss_fn.parameters():
                param.requires_grad = False
            print("LPIPS model initialized globally for training loop.")
        except ImportError:
            print("Warning: LPIPS library not found. LPIPS loss will be skipped.")
            lpips_loss_fn = "skipped"


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False) # ControlNet은 Fine-tuning에서 고정

    unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    params_to_optimize = list(unet.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=cfg.LR,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        weight_decay=cfg.ADAM_WEIGHT_DECAY,
        eps=cfg.ADAM_EPSILON,
    )

    if cfg.MAX_TRAIN_STEPS is None:
        cfg.MAX_TRAIN_STEPS = cfg.EPOCHS * (len(train_data_df) // cfg.BATCH_SIZE)

    lr_scheduler = get_scheduler(
        cfg.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=cfg.LR_WARMUP_STEPS * cfg.GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=cfg.MAX_TRAIN_STEPS,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_df, val_df = train_test_split(train_data_df, test_size=0.1, random_state=cfg.SEED)
    # 실제 데이터셋에 맞게 MAX_DATA 처리
    if cfg.MAX_DATA is not None:
        train_df = train_df.head(min(cfg.MAX_DATA, len(train_df)))
        val_df = val_df.head(min(cfg.MAX_DATA // 10 if cfg.MAX_DATA // 10 > 0 else 1, len(val_df)))
    else:
    # MAX_DATA 미설정 시, val_df는 300장만 샘플링
        val_df = val_df.sample(n=300, random_state=cfg.SEED).reset_index(drop=True)
    
    enhancer = PromptEnhancer()

    train_dataset = ColorizationDataset(
        df=train_df, input_dir=cfg.INPUT_DIR, gt_dir=cfg.GT_DIR, transform=transform,
        tokenizer=tokenizer, enhancer=enhancer, img_size=cfg.IMG_SIZE
    )
    val_dataset = ColorizationDataset(
        df=val_df, input_dir=cfg.INPUT_DIR, gt_dir=cfg.GT_DIR, transform=transform,
        tokenizer=tokenizer, enhancer=enhancer, img_size=cfg.IMG_SIZE
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn, worker_init_fn=worker_init_fn, pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=True,
    )

    unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, val_dataloader
    )
    controlnet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=get_peft_leaf_model(accelerator.unwrap_model(unet)) if isinstance(unet, PeftModel) else accelerator.unwrap_model(unet),
        controlnet=controlnet, scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.check_inputs = lambda *args, **kwargs: None # 입력 검증 건너뛰기

    global_step = 0
    first_epoch = 0

    if cfg.RESUME_FROM_CHECKPOINT:
        if cfg.RESUME_FROM_CHECKPOINT != "latest":
            path = cfg.RESUME_FROM_CHECKPOINT
        else:
            all_checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if not all_checkpoints:
                raise ValueError("No checkpoints found to resume from 'latest'.")
            all_checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            path = os.path.join(output_dir, all_checkpoints[-1])
            print(f"Resuming from latest checkpoint: {path}")

        accelerator.load_state(path)

        global_step_path = os.path.join(path, "global_step.pt")
        if os.path.exists(global_step_path):
            global_step = torch.load(global_step_path)
        else:
            global_step = 0 # Default to 0 if not found

        first_epoch = global_step // len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Resumed training state from {path}, starting at global_step {global_step}, epoch {first_epoch}")

    total_batch_size = cfg.BATCH_SIZE * accelerator.num_processes * cfg.GRADIENT_ACCUMULATION_STEPS
    print("***** Running training *****")
    print(f"   Num examples = {len(train_dataset)}")
    print(f"   Num epochs = {cfg.EPOCHS}")
    print(f"   Instantaneous batch size per device = {cfg.BATCH_SIZE}")
    print(f"   Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"   Gradient Accumulation steps = {cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Total optimization steps = {cfg.MAX_TRAIN_STEPS}")

    progress_bar = tqdm(
        range(global_step, cfg.MAX_TRAIN_STEPS),
        disable=not accelerator.is_main_process,
        initial=global_step
    )
    progress_bar.set_description("Steps")

    best_combined_val_loss = float('inf')
    intervals_no_improve = 0

    for epoch in range(first_epoch, cfg.EPOCHS):
        unet.train()
        controlnet.eval() # ControlNet은 학습되지 않으므로 eval 모드 유지
        train_loss_this_interval = 0.0
        
        for step_in_epoch, batch in enumerate(train_dataloader):
            if global_step >= cfg.MAX_TRAIN_STEPS:
                break
            with accelerator.accumulate(unet):
                # CLIP text encoder (positive prompt)
                encoder_hidden_states_pos = text_encoder(batch["pos_prompt_input_ids"])[0]

                # Noise & Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # ControlNet Input (Canny image)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                
                # ControlNet forward pass (no_grad)
                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents, timesteps, encoder_hidden_states_pos,
                        controlnet_cond=controlnet_image, return_dict=False
                    )
                
                # UNet forward pass (LoRA 학습)
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states_pos,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample

                # Diffusion Loss (MSE Loss)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError("Unknown prediction type")
                
                loss_diffusion = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # VAE 디코딩 for L1/CLIP Loss (그래디언트 계산)
                # pred_original_sample은 중간 계산이므로 no_grad 유지, 하지만 vae.decode 결과는 그래디언트 연결
                with torch.no_grad():
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    sqrt_alpha_t = alpha_t.sqrt()
                    sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

                    if noise_scheduler.config.prediction_type == "epsilon":
                        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        pred_original_sample = noise_scheduler.get_velocity(noisy_latents, model_pred, timesteps)
                    else:
                        raise ValueError("Unknown prediction type")
                
                # VAE 디코딩 (VAE는 freeze 되어 있어도 Unet과의 연결은 유지)
                decoded_latents = 1 / vae.config.scaling_factor * pred_original_sample
                generated_image = vae.decode(decoded_latents.to(dtype=weight_dtype)).sample
                generated_image = generated_image.clamp(-1, 1) # -1 ~ 1 범위로 클램핑

                gt_pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                # L1 Loss 계산
                l1_loss = F.l1_loss(generated_image, gt_pixel_values, reduction='mean')
                
                # CLIP Loss 계산
                clip_features_generated = get_clip_features(generated_image, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_features_gt = get_clip_features(gt_pixel_values, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_loss = 1 - F.cosine_similarity(clip_features_generated, clip_features_gt, dim=-1).mean()

                lpips_loss_train = torch.tensor(0.0, device=accelerator.device)
                ssim_val_train = torch.tensor(0.0, device=accelerator.device)

                # 모든 Loss를 합산합니다.
                total_loss = cfg.LAMBDA_L1 * l1_loss + \
                             loss_diffusion
                
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, cfg.MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss_this_interval += total_loss.item()
                if global_step % cfg.LOG_INTERVAL == 0:
                    avg_train_loss = train_loss_this_interval / (cfg.LOG_INTERVAL * cfg.GRADIENT_ACCUMULATION_STEPS)
                    accelerator.log({
                        "train_loss": avg_train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "diffusion_loss": loss_diffusion.item(),
                        "l1_loss": l1_loss.item(),
                        "clip_loss": clip_loss.item(),
                        "lpips_loss": lpips_loss_train.item(), 
                        "ssim_loss": ssim_val_train.item(),     
                    }, step=global_step)
                    train_loss_this_interval = 0.0

            if (
                global_step % cfg.SAVE_AND_VAL_INTERVAL == 0
                and global_step >= cfg.SAMPLE_SAVE_START_STEP
                and accelerator.is_main_process
            ):
                # 파이프라인의 구성 요소를 unwrap하여 최신 가중치 적용
                pipeline.unet = get_peft_leaf_model(accelerator.unwrap_model(unet))
                pipeline.controlnet = accelerator.unwrap_model(controlnet)
                pipeline.vae = accelerator.unwrap_model(vae)
                pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
                
                # Ensure pipeline is on the correct device with correct dtype before validation
                pipeline.to(accelerator.device, dtype=weight_dtype)
                
                current_combined_val_loss = run_validation(
                    pipeline, accelerator, epoch, global_step, val_dataloader,
                    clip_processor, clip_model, weight_dtype, output_dir, cfg.NUM_SAMPLES_TO_SAVE
                )
                accelerator.wait_for_everyone()

                if current_combined_val_loss < best_combined_val_loss:
                    best_combined_val_loss = current_combined_val_loss
                    intervals_no_improve = 0
                    print(f"New best validation combined loss: {best_combined_val_loss:.4f}. Saving best model.")
                    best_model_dir = os.path.join(output_dir, f"best_model_step{global_step}")
                    best_models = sorted(
                        [d for d in os.listdir(output_dir) if d.startswith("best_model_step")],
                        key=lambda x: int(x.split("step")[-1])
                    )
                    if len(best_models) > 5:
                        for d in best_models[:-5]:
                            shutil.rmtree(os.path.join(output_dir, d), ignore_errors=True)

                    if accelerator.is_main_process:
                        os.makedirs(best_model_dir, exist_ok=True)
                        save_lora_model_overwriting(
                            {"unet": accelerator.unwrap_model(unet)}, best_model_dir,
                            is_main_process=True
                        )
                        # torch.save(global_step, os.path.join(best_model_dir, "global_step.pt"))
                else:
                    intervals_no_improve += 1
                    print(f"Validation combined loss did not improve. Intervals without improvement: {intervals_no_improve}")
                    if intervals_no_improve >= cfg.PATIENCE:
                        print(f"Early stopping triggered after {cfg.PATIENCE} intervals.")
                        break

                save_latest_checkpoint(accelerator.unwrap_model(unet), output_dir, global_step, accelerator, cfg)

            if global_step >= cfg.MAX_TRAIN_STEPS or intervals_no_improve >= cfg.PATIENCE:
                break

        if global_step >= cfg.MAX_TRAIN_STEPS or intervals_no_improve >= cfg.PATIENCE:
            break

    if accelerator.is_main_process:
        print("Training finished. Saving final model.")
        save_lora_model_overwriting(
            {"unet": accelerator.unwrap_model(unet)}, os.path.join(output_dir, "final_model"),
            is_main_process=accelerator.is_main_process,
        )
    accelerator.end_training()

    # 메모리 해제
    del unet, controlnet, vae, text_encoder, tokenizer, optimizer, lr_scheduler
    del train_dataloader, train_dataset, val_dataset
    del clip_processor, clip_model
    if lpips_loss_fn != "skipped" and lpips_loss_fn is not None: 
        del lpips_loss_fn
    lpips_loss_fn = None 
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

# --- Main ---
if __name__ == "__main__":
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, "validation_samples"), exist_ok=True) # 검증 샘플 저장 디렉토리
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    print("Starting training loop...")
    train_loop(
        pretrained_model_name_or_path=CFG.PRETRAINED_MODEL_NAME_OR_PATH,
        controlnet_path=CFG.CONTROLNET_PATH,
        output_dir=CFG.OUTPUT_DIR,
        train_data_df=train_df,
        cfg=CFG,
    )