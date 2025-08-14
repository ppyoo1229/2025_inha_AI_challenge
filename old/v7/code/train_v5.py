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
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
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
# import lpips
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
        self.IMG_SIZE = 384 # 이미지 크기 384로 변경
        self.SEED = 42
        self.OUTPUT_DIR = "./output5" 
        self.TRAIN_CSV = "../train.csv" 
        self.INPUT_DIR = ".." 
        self.GT_DIR = ".."  
        self.LR = 1e-5
        self.BATCH_SIZE = 1 
        self.NUM_WORKERS = 1
        self.EPOCHS = 4
        self.MAX_DATA = None 
        self.LAMBDA_L1 = 0.7 
        self.LAMBDA_CLIP = 1.0 
        # self.LAMBDA_LPIPS = 0.7 
        self.LAMBDA_SSIM = 0.2
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.PROJECT_NAME = "colorization_training_full_finetune" 
        self.PATIENCE = 2
        self.MAX_PROMPT_TOKENS = 55
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
        self.MIXED_PRECISION = "fp16" 
        self.REPORT_TO = "tensorboard"
        self.MAX_TRAIN_STEPS = None 
        self.RESUME_FROM_CHECKPOINT = None 
        self.SAMPLE_SAVE_START_STEP = 400 
        self.SAMPLE_SAVE_END_STEP = 500 
        self.NUM_SAMPLES_TO_SAVE = 1
        self.LOG_INTERVAL = 10 
        self.VAL_INTERVAL = 1 
        self.SAVE_INTERVAL = 14500 
        self.MAX_CHECKPOINTS_TO_KEEP = 2 

CFG = Config()

def filter_config_types(config_dict):
    ALLOWED = (int, float, str, bool)
    return {k: v for k, v in config_dict.items() if isinstance(v, ALLOWED)}

def debug_tensor_info(name, tensor):
    try:
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    except Exception as e:
        print(f"{name}: {type(tensor)}, Error: {e}")
    pass 

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

def safe_prompt_str(prompt_str, tokenizer, max_len=77):
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
        self.food_keywords = ['food', 'sushi', 'fruit', 'vegetable', 'meal', 'dish', 'dessert']
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
    # Normalize to [0, 1] if input is [-1, 1]
    tensor = (tensor + 1) / 2.0 if tensor.min() < 0 or tensor.max() > 1 else tensor
    tensor = torch.clamp(tensor, 0, 1) # Ensure values are within [0, 1]

    image_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

# lpips_loss_fn = None # 전역 변수로 선언

def ssim_loss(img1, img2, data_range=1.0, size_average=True):
    # SSIM 계산을 위해 입력을 [0, 1] 범위로 정규화하고 float32로 캐스팅
    img1_normalized = (img1.float() + 1) / 2.0 
    img2_normalized = (img2.float() + 1) / 2.0 
    return 1 - msssim(img1_normalized, img2_normalized, data_range=data_range, size_average=size_average)

def get_clip_features(image_tensor, clip_processor, clip_model, accelerator_device, weight_dtype):
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
        all_captions_for_phrases = self.df['caption'].astype(str).tolist()
        self.remove_phrases = build_remove_phrases(all_captions_for_phrases, ngram_ns=(2,3,4), topk=100)

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
        cleaned_caption_raw = clean_caption_full(raw_caption, self.remove_phrases, number_words, number_regex)

        # NSFW 필터링
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

        # Positive Prompt 구성
        current_pos_prompt_parts = [cleaned_caption]
        enhancement_keywords_list = self.enhancer.get_enhancement_keywords(cleaned_caption)
        random.shuffle(enhancement_keywords_list)

        for keyword_phrase in enhancement_keywords_list:
            temp_prompt = ", ".join(current_pos_prompt_parts + [keyword_phrase])
            temp_token_ids = self.tokenizer.encode(
                temp_prompt,
                add_special_tokens=True, 
                truncation=False, # 토큰 길이 체크를 위해 일단 자르지 않음
                return_tensors="pt"
            )[0]
            if len(temp_token_ids) <= self.max_tokens:
                current_pos_prompt_parts.append(keyword_phrase)
            else:
                break # max_tokens를 초과하면 더 이상 추가하지 않음
        
        pos_prompt_str_raw = ", ".join(current_pos_prompt_parts)
        final_pos_prompt_str_for_pipe = safe_prompt_str(pos_prompt_str_raw, self.tokenizer, self.max_tokens)
        
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
        canny_low, canny_high = self.dynamic.get_optimal_canny_params(cleaned_caption)
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil) # Normalizes to [-1, 1]

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_rgb_tensor = self.transform(gt_image_pil) # Normalizes to [-1, 1]

        # 동적 파라미터 (guidance, steps)
        guidance = self.dynamic.get_optimal_guidance(cleaned_caption)
        steps = self.dynamic.get_optimal_steps(cleaned_caption)

        return {
            "conditioning_pixel_values": input_control_image, # Canny image ([-1, 1])
            "gt_rgb_tensor": gt_rgb_tensor, # GT image ([-1, 1])
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
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float() # float32로 유지
    
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float() # float32로 유지
    
    pos_prompt_input_ids = torch.stack([example["pos_prompt_input_ids"] for example in examples])
    neg_prompt_input_ids = torch.stack([example["neg_prompt_input_ids"] for example in examples])
    
    pos_prompt_str_for_pipe = [str(example["pos_prompt_str_for_pipe"]) for example in examples]
    neg_prompt_str_for_pipe = [str(example["neg_prompt_str_for_pipe"]) for example in examples]

    guidance_scales = torch.tensor([example["guidance"] for example in examples])
    num_inference_steps = torch.tensor([example["steps"] for example in examples])

    return {
        "pixel_values": pixel_values,  # GT image ([-1, 1], float32)
        "conditioning_pixel_values": conditioning_pixel_values, # Canny image ([-1, 1], float32)
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


# Stable Diffusion 파이프라인만 저장 (추론용)
def save_model_overwriting(pipeline, dir_path, is_main_process=True):
    if is_main_process:
        abs_dir_path = os.path.abspath(dir_path)
        os.makedirs(abs_dir_path, exist_ok=True)
        # save_pretrained는 기본적으로 존재하는 내용을 덮어씁니다.
        pipeline.save_pretrained(abs_dir_path)
        print(f"Model (pipeline) saved to {abs_dir_path}")

@torch.no_grad()
def run_validation(
    pipeline,
    accelerator,
    epoch,
    train_global_step,
    val_dataset,
    clip_processor,
    clip_model,
    lpips_loss_fn,
    weight_dtype, # weight_dtype을 받아서 사용
    output_dir,
    num_samples_to_save=CFG.NUM_SAMPLES_TO_SAVE,
):
    print(f"Running validation for Epoch {epoch}, Global Step {train_global_step}...")
    
    val_output_dir = os.path.join(output_dir, f"validation_epoch_{epoch}_step_{train_global_step}")
    os.makedirs(val_output_dir, exist_ok=True)

    pipeline.scheduler.set_timesteps(50) 
    
    total_l1_loss = 0.0
    total_clip_loss = 0.0
    total_lpips_loss = 0.0
    total_ssim_loss = 0.0
    count = 0

    sample_indices = random.sample(range(len(val_dataset)), min(num_samples_to_save, len(val_dataset)))
    
    for i, idx in enumerate(sample_indices):
        data = val_dataset[idx]
        print(f"[VAL] data index {idx}")
        # debug_tensor_info는 주석 처리됨
        
        prompt = data["pos_prompt_str_for_pipe"]
        negative_prompt = data["neg_prompt_str_for_pipe"]
        conditioning_image = tensor_to_pil(data["conditioning_pixel_values"]) # Convert Canny tensor to PIL
        # GT 텐서도 float32로 유지 (손실 계산을 위해)
        gt_rgb_tensor = data["gt_rgb_tensor"].unsqueeze(0).to(accelerator.device).float() 
        
        guidance_scale = data["guidance"]
        num_inference_steps = data["steps"]

        # AMP context manager with correct dtype
        with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
            generated_images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=conditioning_image, # pipeline의 image 입력은 PIL Image이므로 dtype 변환 불필요
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil", 
            ).images

        gen_pil_image = generated_images[0]
        gen_tensor_image = transforms.ToTensor()(gen_pil_image) 
        
        # 생성된 이미지 텐서도 [-1, 1] 범위로 정규화 및 float32로 변환 (손실 계산을 위해)
        gen_tensor_image = (gen_tensor_image * 2.0) - 1.0 
        gen_tensor_image = gen_tensor_image.unsqueeze(0).to(accelerator.device).float()
        
        # Calculate losses
        # L1 loss
        l1 = F.l1_loss(gen_tensor_image, gt_rgb_tensor) 
        
        # CLIP loss (입력 텐서를 weight_dtype으로 전달)
        clip_features_gen = get_clip_features(gen_tensor_image, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_features_gt = get_clip_features(gt_rgb_tensor, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_loss = 1 - F.cosine_similarity(clip_features_gen, clip_features_gt).mean()
        
        # # LPIPS loss (LPIPS 모델 자체는 float32로 유지하지만, 입력은 [0,1] float32로)
        # lpips_val = lpips_loss_fn(
        #     ((gen_tensor_image + 1) / 2.0).to(torch.float32), # LPIPS는 float32 선호
        #     ((gt_rgb_tensor + 1) / 2.0).to(torch.float32)    # LPIPS는 float32 선호
        # ).mean()

        # SSIM loss (입력 텐서를 float32로 전달)
        ssim_val = ssim_loss(gen_tensor_image, gt_rgb_tensor)

        total_l1_loss += l1.item()
        total_clip_loss += clip_loss.item()
        # total_lpips_loss += lpips_val.item()
        total_ssim_loss += ssim_val.item()
        count += 1
        
        # Save generated images
        output_filename = os.path.join(val_output_dir, f"sample_{i+1}_{os.path.basename(data['file_name'])}")
        gen_pil_image.save(output_filename)

        # Save Canny and GT for comparison
        conditioning_image.save(os.path.join(val_output_dir, f"sample_{i+1}_{os.path.basename(data['file_name'])}_canny.png"))
        tensor_to_pil(gt_rgb_tensor.squeeze(0)).save(os.path.join(val_output_dir, f"sample_{i+1}_{os.path.basename(data['file_name'])}_gt.png"))

    avg_l1_loss = total_l1_loss / count
    avg_clip_loss = total_clip_loss / count
    # avg_lpips_loss = total_lpips_loss / count
    avg_ssim_loss = total_ssim_loss / count

    log_message = (
        f"Validation Results (Epoch {epoch}, Global Step {train_global_step}):\n"
        f" Average L1 Loss: {avg_l1_loss:.4f}\n"
        f" Average CLIP Loss: {avg_clip_loss:.4f}\n"
        # f" Average LPIPS Loss: {avg_lpips_loss:.4f}\n"
        f" Average SSIM Loss: {avg_ssim_loss:.4f}"
    )
    print(log_message)
    
    accelerator.log({
        "val_avg_l1_loss": avg_l1_loss,
        "val_avg_clip_loss": avg_clip_loss,
        # "val_avg_lpips_loss": avg_lpips_loss,
        "val_avg_ssim_loss": avg_ssim_loss,
    }, step=train_global_step)

    return avg_l1_loss
    # return avg_lpips_loss

def train_loop(
    pretrained_model_name_or_path: str,
    controlnet_path: str,
    output_dir: str,
    train_data_df: pd.DataFrame,
    cfg: Config,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=cfg.MIXED_PRECISION, # Config에서 설정된 mixed_precision 사용
        log_with=cfg.REPORT_TO,
        project_dir=os.path.join(output_dir, cfg.PROJECT_NAME),
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers(cfg.PROJECT_NAME, config=filter_config_types(vars(cfg)))

    # Get the proper dtype for training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 1. Load models
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_path)

    # 2. Set requires_grad for fine-tuning
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    unet.train()
    controlnet.train()
    unet.requires_grad_(True)
    controlnet.requires_grad_(True)

    # 3. Create pipeline for validation (not for training steps)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet, 
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None, 
        feature_extractor=None,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention() 

    # 4. Dataset and DataLoader
    train_transforms = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Lambda(lambda img: transforms.Pad(
            (max(0, (cfg.IMG_SIZE - img.width) // 2), max(0, (cfg.IMG_SIZE - img.height) // 2)),
            fill=(0,0,0) 
        )(img) if img.width < cfg.IMG_SIZE or img.height < cfg.IMG_SIZE else img),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), 
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
, 
    ])

    enhancer = PromptEnhancer()
    dynamic_param_gen = DynamicParameterGenerator()

    if cfg.MAX_DATA:
        train_data_df = train_data_df.head(cfg.MAX_DATA)

    train_df, val_df = train_test_split(train_data_df, test_size=0.1, random_state=cfg.SEED) 
    train_dataset = ColorizationDataset(
        df=train_df,
        input_dir=cfg.INPUT_DIR,
        gt_dir=cfg.GT_DIR,
        transform=train_transforms,
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
        pin_memory=False
    )

    val_dataset = ColorizationDataset(
        df=val_df,
        input_dir=cfg.INPUT_DIR,
        gt_dir=cfg.GT_DIR,
        transform=train_transforms,
        tokenizer=tokenizer,
        enhancer=enhancer,
        dynamic=dynamic_param_gen,
        img_size=cfg.IMG_SIZE
    )

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(controlnet.parameters()),
        lr=cfg.LR,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        weight_decay=cfg.ADAM_WEIGHT_DECAY,
        eps=cfg.ADAM_EPSILON,
    )

    # 6. CLIP Model for CLIP Loss
    clip_model = CLIPModel.from_pretrained(cfg.CLIP_MODEL).eval()
    clip_processor = CLIPProcessor.from_pretrained(cfg.CLIP_MODEL)
    
    # # 7. LPIPS Loss
    # global lpips_loss_fn 
    # lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device) 

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    unet, controlnet, optimizer, train_dataloader = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader
    )
    clip_model.to(accelerator.device, dtype=weight_dtype)
    
    # Scheduler and Total Steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.GRADIENT_ACCUMULATION_STEPS)
    if cfg.MAX_TRAIN_STEPS is None:
        cfg.MAX_TRAIN_STEPS = cfg.EPOCHS * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        cfg.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=cfg.LR_WARMUP_STEPS * cfg.GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=cfg.MAX_TRAIN_STEPS,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # --- Training loop ---
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoints_history = [] 

    # --- Resume from checkpoint using Accelerator's load_state ---
    if cfg.RESUME_FROM_CHECKPOINT:
        if os.path.exists(cfg.RESUME_FROM_CHECKPOINT):
            print(f"Resuming training from checkpoint: {cfg.RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(cfg.RESUME_FROM_CHECKPOINT)
       
            global_step = accelerator.state.completed_steps
            initial_epoch = global_step // num_update_steps_per_epoch
            print(f"Resumed from global_step {global_step}, starting epoch {initial_epoch + 1}")
            
            existing_checkpoints = sorted([
                os.path.join(output_dir, d) 
                for d in os.listdir(output_dir) 
                if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))
            ])
            checkpoints_history = existing_checkpoints[-cfg.MAX_CHECKPOINTS_TO_KEEP:] # 마지막 N개만 유지
        else:
            print(f"Resume checkpoint path not found: {cfg.RESUME_FROM_CHECKPOINT}")
            print("Starting training from scratch.")
            initial_epoch = 0
    else:
        print("Starting training from scratch (no resume checkpoint specified).")
        initial_epoch = 0

    for epoch in range(initial_epoch, cfg.EPOCHS):
        unet.train()
        controlnet.train()
        
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(train_bar):
            if global_step >= cfg.MAX_TRAIN_STEPS:
                break 

            with accelerator.accumulate(unet, controlnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor 
                
                noise = torch.randn_like(latents).to(dtype=weight_dtype) # noise도 weight_dtype
                
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype) # noisy_latents도 weight_dtype
                
                encoder_hidden_states = text_encoder(batch["pos_prompt_input_ids"].to(accelerator.device))[0].to(dtype=weight_dtype)
                
                controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device).to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    controlnet_cond=controlnet_image, conditioning_scale=1.0, return_dict=False
                )
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                predicted_original_sample_latent = noise_scheduler.step(model_pred, timesteps, noisy_latents).pred_original_sample
                
                # 예측된 latent를 픽셀 공간으로 디코딩
                with torch.no_grad():
                    predicted_pixels = vae.decode(
                    (predicted_original_sample_latent / vae.config.scaling_factor).to(dtype=weight_dtype)
                    ).sample.float()
                    # GT 픽셀도 float32로 유지
                    gt_pixels = batch["pixel_values"].to(accelerator.device).float()
                    
                    l1_loss = F.l1_loss(predicted_pixels, gt_pixels)
                    
                    clip_features_gen = get_clip_features(predicted_pixels, clip_processor, clip_model, accelerator.device, weight_dtype)
                    clip_features_gt = get_clip_features(gt_pixels, clip_processor, clip_model, accelerator.device, weight_dtype)
                    clip_loss = 1 - F.cosine_similarity(clip_features_gen, clip_features_gt).mean()
                    
                    # lpips_value = lpips_loss_fn(
                    #     ((predicted_pixels + 1) / 2.0).to(torch.float32), 
                    #     ((gt_pixels + 1) / 2.0).to(torch.float32)
                    # ).mean()

                    ssim_value = ssim_loss(predicted_pixels, gt_pixels)

                total_current_loss = (
                    loss +
                    cfg.LAMBDA_L1 * l1_loss +
                    cfg.LAMBDA_CLIP * clip_loss +
                    cfg.LAMBDA_SSIM * ssim_value
                )

                accelerator.backward(total_current_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad] +
                        [p for p in controlnet.parameters() if p.requires_grad],
                        cfg.MAX_GRAD_NORM
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                train_bar.set_postfix(loss=total_current_loss.item(), global_step=global_step) # global_step도 표시
                accelerator.log({
                    "train_loss": total_current_loss.item(),
                    "diffusion_loss": loss.item(),
                    "l1_loss": l1_loss.item(),
                    "clip_loss": clip_loss.item(),
                    # "lpips_loss": lpips_value.item(),
                    "ssim_loss": ssim_value.item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }, step=global_step)

                # 스텝마다 검증 & 전체 상태 저장
                if global_step % cfg.SAVE_INTERVAL == 0:

                    val_pipeline = StableDiffusionControlNetPipeline(
                        vae=pipeline.vae,
                        text_encoder=pipeline.text_encoder,
                        tokenizer=pipeline.tokenizer,
                        unet=accelerator.unwrap_model(unet),
                        controlnet=accelerator.unwrap_model(controlnet),
                        scheduler=pipeline.scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                    )
                    val_pipeline.vae.to(accelerator.device, dtype=weight_dtype)
                    val_pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
                    val_pipeline.unet.to(accelerator.device, dtype=weight_dtype)
                    val_pipeline.controlnet.to(accelerator.device, dtype=weight_dtype)

                    val_pipeline.enable_xformers_memory_efficient_attention()

                    current_val_loss = run_validation(
                        pipeline=val_pipeline,
                        accelerator=accelerator,
                        epoch=epoch + 1,
                        train_global_step=global_step,
                        val_dataset=val_dataset,
                        clip_processor=clip_processor,
                        clip_model=clip_model,
                        # lpips_loss_fn=lpips_loss_fn,
                        weight_dtype=weight_dtype,
                        output_dir=output_dir,
                        num_samples_to_save=cfg.NUM_SAMPLES_TO_SAVE,
                    )
                    
                    output_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(output_checkpoint_dir)
                    print(f"Full Accelerator state saved to {output_checkpoint_dir}")
                    checkpoints_history.append(output_checkpoint_dir)

                    if len(checkpoints_history) > cfg.MAX_CHECKPOINTS_TO_KEEP:
                        oldest_checkpoint = checkpoints_history.pop(0) # 가장 오래된 것 제거
                        if os.path.exists(oldest_checkpoint):
                            print(f"Removing old checkpoint: {oldest_checkpoint}")
                            shutil.rmtree(oldest_checkpoint)

                    # Stable Diffusion pipeline 저장
                    save_model_overwriting(
                        pipeline=val_pipeline,
                        dir_path=os.path.join(output_dir, "latest_model"),
                        is_main_process=accelerator.is_main_process
                    )
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        print(f"New best validation loss: {best_val_loss:.4f}. Saving best model.")
                        save_model_overwriting(
                            pipeline=val_pipeline,
                            dir_path=os.path.join(output_dir, "best_model"),
                            is_main_process=accelerator.is_main_process
                        )
                    del val_pipeline 
                    gc.collect()
                    torch.cuda.empty_cache()

            if global_step >= cfg.MAX_TRAIN_STEPS:
                break # 전체 MAX_TRAIN_STEPS를 초과하면 루프 종료

    accelerator.end_training()
    print("Training finished.")

    if accelerator.is_main_process:
        print("Saving final models...")
        final_pipeline = StableDiffusionControlNetPipeline(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=accelerator.unwrap_model(unet),
            controlnet=accelerator.unwrap_model(controlnet),
            scheduler=pipeline.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        final_pipeline.vae.to(accelerator.device, dtype=weight_dtype)
        final_pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
        final_pipeline.unet.to(accelerator.device, dtype=weight_dtype)
        final_pipeline.controlnet.to(accelerator.device, dtype=weight_dtype)

        save_model_overwriting(
            pipeline=final_pipeline,
            dir_path=os.path.join(output_dir, "final_model"),
            is_main_process=accelerator.is_main_process
        )
        print("Final models saved.")

# Main execution block
if __name__ == "__main__":
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    train_df = pd.read_csv(CFG.TRAIN_CSV)

    print("Starting training loop (Full Finetune)...")
    print("\n---------------------------------------------------------------------------------------------------")
    print(f"WARNING: To enable mixed precision (fp16), please run the script using the following command:")
    print(f"accelerate launch --mixed_precision=\"{CFG.MIXED_PRECISION}\" train4re.py")
    print("---------------------------------------------------------------------------------------------------\n")

    train_loop(
        pretrained_model_name_or_path=CFG.PRETRAINED_MODEL_NAME_OR_PATH,
        controlnet_path=CFG.CONTROLNET_PATH,
        output_dir=CFG.OUTPUT_DIR,
        train_data_df=train_df,
        cfg=CFG,
    )