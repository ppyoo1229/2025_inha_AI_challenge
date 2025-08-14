import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from accelerate import Accelerator
# Corrected import for ControlNetModelForControlNet if needed for main(), but not for this specific fix directly
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
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
        self.OUTPUT_DIR = "./output2"
        self.TRAIN_CSV = "../train.csv"
        self.INPUT_DIR = "../train"
        self.GT_DIR = "../train"
        self.LR = 1e-6
        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 4
        self.EPOCHS = 20
        self.MAX_DATA = None
        self.LAMBDA_L1 = 1.4
        self.LAMBDA_CLIP = 0.5
        self.LAMBDA_LPIPS = 0.2
        self.LAMBDA_SSIM = 0.2
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.PROJECT_NAME = "colorization_training"
        self.PATIENCE = 4
        self.MAX_PROMPT_TOKENS = 57
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "erotic", "nude", "breast", "ass", "penis", "vagina"]
        self.SFW_CAPTION_REPLACEMENT = "a high quality image, realistic, clean, beautiful, bright, colorful"
        self.VALIDATION_PROMPTS = ["a photo of a cat", "a photo of a dog"]
        self.VALIDATION_NEGATIVE_PROMPTS = ["ugly, bad anatomy", "ugly, bad anatomy"]
        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.MAX_GRAD_NORM = 1.0
        self.LR_SCHEDULER_TYPE = "constant"
        self.LR_WARMUP_STEPS = 500
        self.ADAM_BETA1 = 0.9
        self.ADAM_BETA2 = 0.999
        self.ADAM_WEIGHT_DECAY = 1e-2
        self.ADAM_EPSILON = 1e-08
        self.MIXED_PRECISION = "no"
        self.REPORT_TO = "tensorboard"
        self.CHECKPOINTING_STEPS = 500
        self.VALIDATION_STEPS = 100
        self.MAX_TRAIN_STEPS = None
        self.RESUME_FROM_CHECKPOINT = None
        self.SAMPLE_SAVE_START_STEP = 400
        self.SAMPLE_SAVE_END_STEP = 500
        self.NUM_SAMPLES_TO_SAVE = 1
        self.MAX_CHECKPOINTS_TO_KEEP = 2 

CFG = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

def safe_prompt_str(prompt_str, tokenizer, max_len):
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=True, truncation=False, return_tensors="pt")[0]
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        prompt_str = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    return prompt_str

class PromptEnhancer:
    def __init__(self):
        self.quality_keywords = ["masterpiece", "best quality", "high resolution", "4k", "8k"]
        self.texture_keywords = ["detailed texture", "smooth texture", "realistic texture"]
        self.lighting_keywords = ["dramatic lighting", "soft lighting", "cinematic lighting", "studio lighting"]
        self.scene_keywords = ["wide angle", "close up", "full body shot", "dynamic pose", "indoor scene", "outdoor scene"]
        self.fixed_tail = "colorful, vibrant colors, maintain original structure, do not change structure, only colorize"
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
        if any(k in cap for k in self.person_keywords):
            cat.append("person")
        if any(k in cap for k in self.landscape_keywords):
            cat.append("landscape")
        if any(k in cap for k in self.food_keywords):
            cat.append("food")
        if any(k in cap for k in self.object_keywords):
            cat.append("object")
        if any(k in cap for k in self.art_keywords):
            cat.append("art")
        return cat

    def get_color_enhancements(self, caption):
        colors_found = set()
        cap = caption.lower()
        for c in self.color_words:
            if re.search(rf'\b{c}\b', cap):
                colors_found.add(c)
        enh = []
        for c in colors_found:
            enh.append(random.choice(self.color_enhancements[c]))
        return enh

    def get_base_negative_prompt(self, cleaned_caption=None):
        return self.base_negative_prompts

    def get_enhancement_keywords(self, cleaned_caption):
        enhancement_list = []
        enhancement_list.append(random.choice(self.quality_keywords))
        enhancement_list.append(random.choice(self.texture_keywords))
        enhancement_list.append(random.choice(self.lighting_keywords))
        enhancement_list.append(random.choice(self.scene_keywords))
        categories = self.get_category(cleaned_caption)
        if "person" in categories:
            enhancement_list.append(random.choice(self.person_enhance))
        if "landscape" in categories:
            enhancement_list.append(random.choice(self.landscape_enhance))
        if "food" in categories:
            enhancement_list.append(random.choice(self.food_enhance))
        if "object" in categories:
            enhancement_list.append(random.choice(self.object_enhance))
        if "art" in categories:
            enhancement_list.append(random.choice(self.art_enhance))
        color_enhance_list = self.get_color_enhancements(cleaned_caption)
        enhancement_list.extend(color_enhance_list)
        enhancement_list.append(self.fixed_tail)
        return list(dict.fromkeys([x.strip() for x in enhancement_list if x.strip()]))

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

def tensor_to_pil(tensor):
    tensor = tensor.cpu().float()
    tensor = torch.clamp((tensor + 1) / 2.0 if tensor.min() < 0 or tensor.max() > 1 else tensor, 0, 1)
    image_np = tensor.permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

def get_clip_features(image_tensor, clip_processor, clip_model, accelerator_device, weight_dtype):
    # image_tensor: (B, C, H, W) or (C, H, W)
    if image_tensor.ndim == 3:
        pil_list = [tensor_to_pil(image_tensor)]
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
    return features

lpips_loss_fn = None

def ssim_loss(img1, img2, data_range=1.0, size_average=True):
    # Ensure inputs are float32 for msssim, and within [0, 1] range
    img1_normalized = (img1 + 1) / 2.0 # Assuming img1 is in [-1, 1] range
    img2_normalized = (img2 + 1) / 2.0 # Assuming img2 is in [-1, 1] range
    return 1 - msssim(img1_normalized.float(), img2_normalized.float(), data_range=data_range, size_average=size_average)

class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, dynamic, img_size=512): # Removed random_seed
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
        all_captions_for_phrases = self.df['caption'].astype(str).tolist()
        self.remove_phrases = build_remove_phrases(all_captions_for_phrases, ngram_ns=(2,3,4), topk=100)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # No per-item seed setting here, handled by worker_init_fn
        row = self.df.iloc[idx]
        cleaned_input_path_from_csv = os.path.normpath(row['input_img_path'])
        cleaned_gt_path_from_csv = os.path.normpath(row['gt_img_path'])

        input_image_path = os.path.join(self.input_dir, cleaned_input_path_from_csv)
        gt_image_path = os.path.join(self.gt_dir, cleaned_gt_path_from_csv)

        original_input_pil = Image.open(input_image_path).convert("RGB")
        input_image_np = np.array(original_input_pil)
        gray_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)

        raw_caption = str(row['caption'])
        cleaned_caption_raw = clean_caption_full(raw_caption, self.remove_phrases, number_words, number_regex)

        is_nsfw = False
        cleaned_caption_lower = cleaned_caption_raw.lower()
        for nsfw_kw in self.nsfw_keywords:
            if nsfw_kw in cleaned_caption_lower:
                is_nsfw = True
                break

        if is_nsfw:
            current_cleaned_caption_for_processing = self.sfw_caption_replacement
        else:
            current_cleaned_caption_for_processing = cleaned_caption_raw

        cleaned_caption = current_cleaned_caption_for_processing

        current_pos_prompt_parts = [cleaned_caption]
        enhancement_keywords_list = self.enhancer.get_enhancement_keywords(cleaned_caption)
        random.shuffle(enhancement_keywords_list)

        for keyword_phrase in enhancement_keywords_list:
            temp_prompt = ", ".join(current_pos_prompt_parts + [keyword_phrase])
            temp_token_ids = self.tokenizer.encode(
                temp_prompt,
                add_special_tokens=True, 
                truncation=False,
                return_tensors="pt"
            )[0]
            if len(temp_token_ids) <= self.max_tokens:
                current_pos_prompt_parts.append(keyword_phrase)
            else:
                break
        
        pos_prompt_str_raw = ", ".join(current_pos_prompt_parts)
        final_pos_prompt_str_for_pipe = safe_prompt_str(pos_prompt_str_raw, self.tokenizer, self.max_tokens)
        
        pos_tokenized_output = self.tokenizer(
            final_pos_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        final_pos_input_ids = pos_tokenized_output.input_ids[0]

        base_neg_prompt_str = self.enhancer.get_base_negative_prompt(cleaned_caption)
        final_neg_prompt_str_for_pipe = safe_prompt_str(base_neg_prompt_str, self.tokenizer, self.max_tokens)
        
        neg_tokenized_output = self.tokenizer(
            final_neg_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        final_neg_input_ids = neg_tokenized_output.input_ids[0]

        canny_low, canny_high = self.dynamic.get_optimal_canny_params(cleaned_caption)
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil)

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_image_transformed = self.transform(gt_image_pil)
        gt_rgb_tensor = gt_image_transformed

        gt_rgb_np = gt_image_transformed.permute(1, 2, 0).numpy()
        gt_lab_np = color.rgb2lab(gt_rgb_np)
        ab_channels = torch.from_numpy(gt_lab_np[:, :, 1:]).float().permute(2, 0, 1)
        ab_channels = (ab_channels + 128) / 255.0
        ab_channels = torch.clamp(ab_channels, min=0.0, max=1.0)

        guidance = self.dynamic.get_optimal_guidance(cleaned_caption)
        steps = self.dynamic.get_optimal_steps(cleaned_caption)

        return {
            "input_control_image": input_control_image, 
            "gt_rgb_tensor": gt_rgb_tensor, 
            "ab_channels": ab_channels, 
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
    
    conditioning_pixel_values = torch.stack([example["input_control_image"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    pos_prompt_input_ids = torch.stack([example["pos_prompt_input_ids"] for example in examples])
    neg_prompt_input_ids = torch.stack([example["neg_prompt_input_ids"] for example in examples])
    
    pos_prompt_str_for_pipe = [str(example["pos_prompt_str_for_pipe"]) for example in examples]
    neg_prompt_str_for_pipe = [str(example["neg_prompt_str_for_pipe"]) for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "pos_prompt_input_ids": pos_prompt_input_ids,
        "neg_prompt_input_ids": neg_prompt_input_ids,
        "pos_prompt_str_for_pipe": pos_prompt_str_for_pipe,
        "neg_prompt_str_for_pipe": neg_prompt_str_for_pipe,
        "guidance_scales": torch.tensor([example["guidance"] for example in examples]),
        "num_inference_steps": torch.tensor([example["steps"] for example in examples]),
        "captions": [example["caption"] for example in examples],
        "file_names": [example["file_name"] for example in examples],
    }

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    set_seed(worker_seed)

lpips_loss_fn = None 
def get_peft_leaf_model(m):
    # LoRA 래핑 여러 겹 있을 때 최종 diffusers 원본 model만 뽑아줌
    while "lora" in str(type(m)).lower():
        m = getattr(m, "model", m)
    return m

def run_validation(unet, controlnet, vae, text_encoder, val_dataloader, accelerator, weight_dtype, CFG, global_step, sample_output_dir, clip_processor, clip_model, lpips_loss_fn):
    # unwrap -> base_model (LoRA모델) -> get_peft_leaf_model() (진짜 원본)
    unet_base = accelerator.unwrap_model(unet).base_model
    controlnet_base = accelerator.unwrap_model(controlnet).base_model

    unet_for_pipe = get_peft_leaf_model(unet_base)
    controlnet_for_pipe = get_peft_leaf_model(controlnet_base)

    print("unet_for_pipe type:", type(unet_for_pipe))
    print("controlnet_for_pipe type:", type(controlnet_for_pipe))

    unet_for_pipe.eval()
    controlnet_for_pipe.eval()

    inference_pipe_val = None
    if accelerator.is_main_process:
        inference_pipe_val = StableDiffusionControlNetPipeline.from_pretrained(
            CFG.PRETRAINED_MODEL_NAME_OR_PATH,
            unet=unet_for_pipe,            # 핵심! (원본만 넣어야 에러 없음)
            controlnet=controlnet_for_pipe,
            vae=vae,
            text_encoder=text_encoder,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(accelerator.device)
        inference_pipe_val.scheduler = UniPCMultistepScheduler.from_config(inference_pipe_val.scheduler.config)
        inference_pipe_val.set_progress_bar_config(disable=True)
        inference_pipe_val.check_inputs = lambda *args, **kwargs: None 

    total_val_loss = 0.0
    total_l1_loss = 0.0
    total_clip_loss = 0.0
    total_lpips_loss = 0.0
    total_ssim_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i, batch_val in enumerate(tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_main_process)):
            control_image_tensor = batch_val['conditioning_pixel_values']
            control_image_for_pipe_list = [tensor_to_pil(img) for img in control_image_tensor]
            n_samples = len(control_image_for_pipe_list)

            prompt_list = batch_val['pos_prompt_str_for_pipe']
            neg_prompt_list = batch_val['neg_prompt_str_for_pipe']

            if not isinstance(prompt_list, list):
                prompt_list = [str(prompt_list)]
            if not isinstance(neg_prompt_list, list):
                neg_prompt_list = [str(neg_prompt_list)]
            
            if len(prompt_list) != n_samples:
                prompt_list = [prompt_list[0]] * n_samples 
            if len(neg_prompt_list) != n_samples:
                neg_prompt_list = [neg_prompt_list[0]] * n_samples 

            guidance_scale_to_use = float(batch_val['guidance_scales'][0])
            num_inference_steps_to_use = int(batch_val['num_inference_steps'][0])
            original_filename = batch_val['file_names'][0]

            save_sample = accelerator.is_main_process and \
                          (CFG.SAMPLE_SAVE_START_STEP <= global_step <= CFG.SAMPLE_SAVE_END_STEP) and \
                          (i < CFG.NUM_SAMPLES_TO_SAVE)

            generated_images = None
            if accelerator.is_main_process:
                if n_samples == 1:
                    prompt_for_pipe = prompt_list[0]
                    image_for_pipe = control_image_for_pipe_list[0]
                    negative_prompt_for_pipe = neg_prompt_list[0]
                else:
                    prompt_for_pipe = prompt_list
                    image_for_pipe = control_image_for_pipe_list
                    negative_prompt_for_pipe = neg_prompt_list

                generated_images = inference_pipe_val(
                    prompt=prompt_for_pipe,
                    image=image_for_pipe,
                    negative_prompt=negative_prompt_for_pipe,
                    guidance_scale=guidance_scale_to_use,
                    num_inference_steps=num_inference_steps_to_use,
                    output_type="pt"
                ).images
            
            if generated_images is not None:
                target_rgb_val_01 = batch_val['pixel_values'].to(dtype=weight_dtype, device=accelerator.device)
                generated_images_01 = (generated_images + 1) / 2.0
                target_rgb_val_01_for_metrics = (target_rgb_val_01 + 1) / 2.0 

                val_l1_rgb = F.l1_loss(generated_images_01, target_rgb_val_01_for_metrics)
                
                clip_features_fake_val = get_clip_features(generated_images, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_features_real_val = get_clip_features(target_rgb_val_01, clip_processor, clip_model, accelerator.device, weight_dtype)
                val_clip_loss = F.mse_loss(clip_features_fake_val, clip_features_real_val)

                val_lpips_loss = lpips_loss_fn(generated_images, target_rgb_val_01).mean()
                val_ssim_loss_val = ssim_loss(generated_images, target_rgb_val_01, data_range=2.0, size_average=True)

                val_total_loss_item = CFG.LAMBDA_L1 * val_l1_rgb.item() \
                                    + CFG.LAMBDA_CLIP * val_clip_loss.item() \
                                    + CFG.LAMBDA_LPIPS * val_lpips_loss.item() \
                                    + CFG.LAMBDA_SSIM * val_ssim_loss_val.item()

                total_l1_loss += val_l1_rgb.item()
                total_clip_loss += val_clip_loss.item()
                total_lpips_loss += val_lpips_loss.item()
                total_ssim_loss += val_ssim_loss_val.item()
            else:
                val_total_loss_item = 0.0

            total_val_loss += val_total_loss_item
            total_batches += 1

            if save_sample and generated_images is not None:
                os.makedirs(sample_output_dir, exist_ok=True)
                generated_pil_list = [tensor_to_pil(img.cpu()) for img in generated_images] if isinstance(generated_images, torch.Tensor) and generated_images.dim() == 4 else [tensor_to_pil(generated_images.cpu())]

                gt_image_pil = tensor_to_pil(batch_val['pixel_values'][0]) 
                input_control_pil = tensor_to_pil(batch_val['conditioning_pixel_values'][0])
                generated_pil = generated_pil_list[0]

                combined_image = Image.new('RGB', (CFG.IMG_SIZE * 3, CFG.IMG_SIZE))
                combined_image.paste(input_control_pil, (0, 0))
                combined_image.paste(generated_pil, (CFG.IMG_SIZE, 0))
                combined_image.paste(gt_image_pil, (CFG.IMG_SIZE * 2, 0))
                sample_filename = f"sample_step{global_step}_{original_filename}"
                combined_image.save(os.path.join(sample_output_dir, sample_filename))

    if accelerator.is_main_process:
        if inference_pipe_val is not None:
            del inference_pipe_val
        gc.collect()
        torch.cuda.empty_cache()

    avg_val_loss = total_val_loss / total_batches if total_batches > 0 else 0.0
    avg_l1_loss = total_l1_loss / total_batches if total_batches > 0 else 0.0
    avg_clip_loss = total_clip_loss / total_batches if total_batches > 0 else 0.0
    avg_lpips_loss = total_lpips_loss / total_batches if total_batches > 0 else 0.0
    avg_ssim_loss = total_ssim_loss / total_batches if total_batches > 0 else 0.0

    validation_metrics = {
        "val_total_loss": avg_val_loss,
        "val_l1_loss": avg_l1_loss,
        "val_clip_loss": avg_clip_loss,
        "val_lpips_loss": avg_lpips_loss,
        "val_ssim_loss": avg_ssim_loss,
    }
    return avg_val_loss, validation_metrics


# --- Helper for LoRA Checkpoint Management ---
def save_lora_model_overwriting(model, dir_path, subfolder_unet="unet_lora", subfolder_controlnet="controlnet_lora", is_main_process=True):

    if is_main_process:
        unet_lora_path = os.path.join(dir_path, subfolder_unet)
        if os.path.exists(unet_lora_path):
            shutil.rmtree(unet_lora_path)
        model['unet'].save_pretrained(unet_lora_path)

        controlnet_lora_path = os.path.join(dir_path, subfolder_controlnet)
        if os.path.exists(controlnet_lora_path):
            shutil.rmtree(controlnet_lora_path)
        model['controlnet'].save_pretrained(controlnet_lora_path)

def save_lora_checkpoint(unet_model, controlnet_model, checkpoint_dir, global_step, max_checkpoints_to_keep, is_main_process=True):
    if is_main_process:
        current_checkpoint_path = os.path.join(checkpoint_dir, f"lora_checkpoint-{global_step}")
        
        os.makedirs(current_checkpoint_path, exist_ok=True)
        unet_model.save_pretrained(os.path.join(current_checkpoint_path, "unet_lora"))
        controlnet_model.save_pretrained(os.path.join(current_checkpoint_path, "controlnet_lora"))
        
        with open(os.path.join(current_checkpoint_path, "global_step.txt"), "w") as f:
            f.write(str(global_step))
        print(f"Saved LoRA checkpoint to {current_checkpoint_path}")

        all_checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("lora_checkpoint-")]
        all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split('-')[1]))

        if len(all_checkpoints) > max_checkpoints_to_keep:
            for i in range(len(all_checkpoints) - max_checkpoints_to_keep):
                old_checkpoint_path = os.path.join(checkpoint_dir, all_checkpoints[i])
                if os.path.exists(old_checkpoint_path):
                    shutil.rmtree(old_checkpoint_path)
                    print(f"Removed old checkpoint: {old_checkpoint_path}")


# --- Main Training Loop ---
def main():
    set_seed(CFG.SEED) 

    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=CFG.MIXED_PRECISION,
        log_with=CFG.REPORT_TO,
        project_dir=CFG.OUTPUT_DIR,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(CFG.PROJECT_NAME)
        accelerator.wait_for_everyone()

    latest_lora_dir = os.path.join(CFG.OUTPUT_DIR, 'lora_latest_checkpoint')
    best_lora_dir = os.path.join(CFG.OUTPUT_DIR, 'lora_best_model')
    checkpoints_base_dir = os.path.join(CFG.OUTPUT_DIR, 'lora_checkpoints')
    sample_output_dir = os.path.join(CFG.OUTPUT_DIR, 'samples')
    log_dir = os.path.join(CFG.OUTPUT_DIR, 'logs')

    os.makedirs(checkpoints_base_dir, exist_ok=True)
    os.makedirs(sample_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(CFG.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(CFG.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(CFG.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(CFG.CONTROLNET_PATH) 
                                                                    
    clip_processor = CLIPProcessor.from_pretrained(CFG.CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(CFG.CLIP_MODEL)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    clip_model.requires_grad_(False) 

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], 
        lora_dropout=0.1,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    controlnet = get_peft_model(controlnet, lora_config)

    print("학습 직후")
    print("unet:", type(unet))
    print("unet base_model:", type(accelerator.unwrap_model(unet).base_model))
    print("unet model:", type(accelerator.unwrap_model(unet).model) if hasattr(accelerator.unwrap_model(unet), 'model') else "No model attr")
    print("controlnet:", type(controlnet))
    print("controlnet base_model:", type(accelerator.unwrap_model(controlnet).base_model))
    print("controlnet model:", type(accelerator.unwrap_model(controlnet).model) if hasattr(accelerator.unwrap_model(controlnet), 'model') else "No model attr")

    if accelerator.is_main_process:
        unet.print_trainable_parameters()
        controlnet.print_trainable_parameters()

    unet.train()
    controlnet.train()

    noise_scheduler = DDPMScheduler.from_pretrained(CFG.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler")

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, unet.parameters())) + \
        list(filter(lambda p: p.requires_grad, controlnet.parameters())), 
        lr=CFG.LR,
        betas=(CFG.ADAM_BETA1, CFG.ADAM_BETA2),
        weight_decay=CFG.ADAM_WEIGHT_DECAY,
        eps=CFG.ADAM_EPSILON,
    )

    df = pd.read_csv(CFG.TRAIN_CSV)
    if CFG.MAX_DATA is not None:
        df = df.head(CFG.MAX_DATA)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=CFG.SEED)

    train_transforms = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    enhancer = PromptEnhancer()
    dynamic = DynamicParameterGenerator()
    train_dataset = ColorizationDataset(train_df, CFG.INPUT_DIR, CFG.GT_DIR, train_transforms, tokenizer, enhancer, dynamic, CFG.IMG_SIZE)
    val_dataset = ColorizationDataset(val_df, CFG.INPUT_DIR, CFG.GT_DIR, train_transforms, tokenizer, enhancer, dynamic, CFG.IMG_SIZE)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False, 
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    lr_scheduler = get_scheduler(
        CFG.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=CFG.LR_WARMUP_STEPS * CFG.GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=(CFG.MAX_TRAIN_STEPS if CFG.MAX_TRAIN_STEPS else CFG.EPOCHS * len(train_dataloader)) * CFG.GRADIENT_ACCUMULATION_STEPS,
    )

    unet, controlnet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    clip_model.to(accelerator.device)

    global lpips_loss_fn
    lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    lpips_loss_fn.eval()

    global_step = 0
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    break_outer_loop = False

    if CFG.RESUME_FROM_CHECKPOINT:
        resume_lora_path = None
        if os.path.exists(os.path.join(latest_lora_dir, "unet_lora", "adapter_model.safetensors")):
            resume_lora_path = latest_lora_dir
        else:
            lora_checkpoints = [d for d in os.listdir(checkpoints_base_dir) if d.startswith("lora_checkpoint-")]
            lora_checkpoints = sorted(lora_checkpoints, key=lambda x: int(x.split('-')[1]))
            if lora_checkpoints:
                resume_lora_path = os.path.join(checkpoints_base_dir, lora_checkpoints[-1])
            elif CFG.RESUME_FROM_CHECKPOINT != "latest" and \
                 os.path.exists(os.path.join(CFG.RESUME_FROM_CHECKPOINT, "unet_lora", "adapter_model.safetensors")):
                resume_lora_path = CFG.RESUME_FROM_CHECKPOINT

        if resume_lora_path:
            if accelerator.is_main_process:
                print(f"Loading LoRA weights from: {resume_lora_path}")
    
            accelerator.unwrap_model(unet).load_adapter(os.path.join(resume_lora_path, "unet_lora"), adapter_name="default")
            accelerator.unwrap_model(controlnet).load_adapter(os.path.join(resume_lora_path, "controlnet_lora"), adapter_name="default")
            
            step_file = os.path.join(resume_lora_path, "global_step.txt")
            if os.path.exists(step_file):
                with open(step_file, "r") as f:
                    global_step = int(f.read().strip())
            else:
                global_step = 0

            start_epoch = global_step // len(train_dataloader)
            if accelerator.is_main_process:
                print(f"Resuming training from LoRA checkpoint at global step {global_step}")
        else:
            if accelerator.is_main_process:
                print("No valid LoRA checkpoint found for resuming. Starting training from scratch.")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = len(train_dataloader)
    if CFG.MAX_TRAIN_STEPS is None:
        CFG.MAX_TRAIN_STEPS = CFG.EPOCHS * num_update_steps_per_epoch
    else:
        CFG.EPOCHS = math.ceil(CFG.MAX_TRAIN_STEPS / num_update_steps_per_epoch)

    total_batch_size = CFG.BATCH_SIZE * accelerator.num_processes * CFG.GRADIENT_ACCUMULATION_STEPS

    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num epochs = {CFG.EPOCHS}")
        print(f"  Instantaneous batch size per device = {CFG.BATCH_SIZE}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {CFG.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Total optimization steps = {CFG.MAX_TRAIN_STEPS}")

    progress_bar = tqdm(
        range(global_step, CFG.MAX_TRAIN_STEPS),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )

    for epoch in range(start_epoch, CFG.EPOCHS):
        unet.train()
        controlnet.train()
        train_loss = 0.0
        break_outer_loop = False

        for step, batch in enumerate(train_dataloader):
            current_step_in_epoch_overall = epoch * num_update_steps_per_epoch + step
            if current_step_in_epoch_overall < global_step:
                if accelerator.is_main_process:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["pos_prompt_input_ids"])[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    batch["conditioning_pixel_values"].to(weight_dtype),
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                avg_loss = accelerator.gather(loss.repeat(CFG.BATCH_SIZE)).mean()
                train_loss += avg_loss.item() 
                
                accelerator.backward(loss) 

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(filter(lambda p: p.requires_grad, unet.parameters())) + 
                        list(filter(lambda p: p.requires_grad, controlnet.parameters())), 
                        CFG.MAX_GRAD_NORM
                    )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    global_step += 1
                    
                    accelerator.log({"train_loss": train_loss / CFG.GRADIENT_ACCUMULATION_STEPS}, step=global_step)
                    train_loss = 0.0

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                if global_step >= CFG.MAX_TRAIN_STEPS:
                    break_outer_loop = True
                    break

        # ------ 여기서만 검증 ------
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.cuda.empty_cache()
            gc.collect()
            val_loss, val_metrics = run_validation(
                unet, controlnet, vae, text_encoder,
                val_dataloader, accelerator, weight_dtype, CFG, global_step, sample_output_dir,
                clip_processor, clip_model, lpips_loss_fn
            )
            accelerator.log(val_metrics, step=global_step)
            print(f"Epoch {epoch+1} / Step {global_step}: Validation Total Loss: {val_loss:.4f} "
                  f"(L1: {val_metrics['val_l1_loss']:.4f}, CLIP: {val_metrics['val_clip_loss']:.4f}, "
                  f"LPIPS: {val_metrics['val_lpips_loss']:.4f}, SSIM_Loss: {val_metrics['val_ssim_loss']:.4f})")
            torch.cuda.empty_cache()
            gc.collect()

            # 베스트 저장, 얼리스탑 
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_lora_model_overwriting(
                    {'unet': accelerator.unwrap_model(unet), 'controlnet': accelerator.unwrap_model(controlnet)}, 
                    best_lora_dir
                )
                with open(os.path.join(best_lora_dir, "global_step.txt"), "w") as f:
                    f.write(str(global_step))
                print(f"Saved best LoRA weights to {best_lora_dir} with loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= CFG.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} / step {global_step} due to no improvement in validation loss for {patience_counter} consecutive validation checks.")
                    break_outer_loop = True

            save_lora_model_overwriting(
                {'unet': accelerator.unwrap_model(unet), 'controlnet': accelerator.unwrap_model(controlnet)}, 
                latest_lora_dir
            )
            with open(os.path.join(latest_lora_dir, "global_step.txt"), "w") as f:
                f.write(str(global_step))
            print(f"Saved latest LoRA weights to {latest_lora_dir}")

        if break_outer_loop:
            break

    if accelerator.is_main_process:
        print("\nTraining complete. Saving final LoRA weights.")
        save_lora_model_overwriting(
            {'unet': accelerator.unwrap_model(unet), 'controlnet': accelerator.unwrap_model(controlnet)}, 
            os.path.join(CFG.OUTPUT_DIR, "final_lora_model")
        )
        print(f"Final LoRA weights saved to {os.path.join(CFG.OUTPUT_DIR, 'final_lora_model')}")
        
    accelerator.end_training()

if __name__ == "__main__":
    main()