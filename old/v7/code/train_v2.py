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
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPTokenizer
import lpips
import cv2
from skimage import color
from pytorch_msssim import ssim
import random
import re
import string
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
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
        self.BATCH_SIZE = 2 
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
        self.MAX_PROMPT_TOKENS = 55
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "erotic", "nude", "breast", "ass", "penis", "vagina"]
        self.SFW_CAPTION_REPLACEMENT = "a high quality image, realistic, clean, beautiful, bright, colorful"

CFG = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.SEED)

WORK_DIR = os.path.join(CFG.OUTPUT_DIR, 'working_dir')
os.makedirs(WORK_DIR, exist_ok=True)
latest_model_dir = os.path.join(CFG.OUTPUT_DIR, 'latest_checkpoint')
best_model_dir = os.path.join(CFG.OUTPUT_DIR, 'best_model')
sample_output_dir = os.path.join(CFG.OUTPUT_DIR, 'samples')
os.makedirs(latest_model_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(sample_output_dir, exist_ok=True)

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
        # If too long, truncate and then decode
        input_ids = input_ids[:max_len]
        prompt_str = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    return prompt_str

# --- PromptEnhancer ---
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

# --- tensor to PIL 변환 ---
def tensor_to_pil(tensor):
    tensor = torch.clamp((tensor + 1) / 2.0 if tensor.min() < 0 or tensor.max() > 1 else tensor, 0, 1)
    # C, H, W -> H, W, C
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    # 0-1 -> 0-255
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

# --- 데이터셋 클래스 ---
class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, dynamic, img_size=512, random_seed=None):
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
        self.random_seed = random_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.random_seed is not None:
            random.seed(self.random_seed + idx)
            np.random.seed(self.random_seed + idx)
            # torch.manual_seed(self.random_seed + idx) 

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
        random.shuffle(enhancement_keywords_list) # 순서 랜덤화

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
                break # 더 이상 추가할 수 없으면 중단

        pos_prompt_str_raw = ", ".join(current_pos_prompt_parts)
        # 최종 프롬프트 문자열을 safe_prompt_str로 처리하여 길이 제한 및 클린업
        final_pos_prompt_str_for_pipe = safe_prompt_str(pos_prompt_str_raw, self.tokenizer, self.max_tokens)
        
        # 실제 모델에 들어갈 input_ids는 padding 및 truncation을 적용
        pos_tokenized_output = self.tokenizer(
            final_pos_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        final_pos_input_ids = pos_tokenized_output.input_ids[0]
        
        # Negative prompt
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
        input_control_image = self.transform(canny_image_pil) # transforms.ToTensor()로 인해 0-1 범위

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_image_transformed = self.transform(gt_image_pil) # transforms.ToTensor()로 인해 0-1 범위
        gt_rgb_tensor = gt_image_transformed # GT RGB는 이미 0-1 범위이므로 그대로 사용

        gt_rgb_np = gt_image_transformed.permute(1, 2, 0).numpy() # 0-1 범위 numpy 배열
        gt_lab_np = color.rgb2lab(gt_rgb_np) # LAB 변환
        ab_channels = torch.from_numpy(gt_lab_np[:, :, 1:]).float().permute(2, 0, 1) # a*b* 채널 추출
        ab_channels = (ab_channels + 128) / 255.0 # LAB a*b* 채널을 대략 -128~128에서 0~1 범위로 정규화
        ab_channels = torch.clamp(ab_channels, min=0.0, max=1.0) # 0-1 범위

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

# --- 인스턴스 생성 ---
prompt_enhancer = PromptEnhancer()
dynamic_param_gen = DynamicParameterGenerator()

df = pd.read_csv(CFG.TRAIN_CSV)
if CFG.MAX_DATA:
    df = df.sample(CFG.MAX_DATA, random_state=CFG.SEED).reset_index(drop=True)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=CFG.SEED) 

transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
])

# --- Accelerator ---
accelerator = Accelerator(
    project_dir=CFG.OUTPUT_DIR,
    log_with="tensorboard", 
    gradient_accumulation_steps=1
)

controlnet = ControlNetModel.from_pretrained(
    CFG.CONTROLNET_PATH, torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    CFG.MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
tokenizer = CLIPTokenizer.from_pretrained(CFG.MODEL_PATH, subfolder="tokenizer")

train_dataset = ColorizationDataset(
    train_df, CFG.INPUT_DIR, CFG.GT_DIR, transform, tokenizer, prompt_enhancer, dynamic_param_gen,
    img_size=CFG.IMG_SIZE, random_seed=CFG.SEED
)
val_dataset = ColorizationDataset(
    val_df, CFG.INPUT_DIR, CFG.GT_DIR, transform, tokenizer, prompt_enhancer, dynamic_param_gen,
    img_size=CFG.IMG_SIZE, random_seed=CFG.SEED + 1 
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True
)

# --- LoRA ---
lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, init_lora_weights="gaussian",
    target_modules=["to_q", "to_v", "to_k", "to_out.0"]
)
pipe.unet.add_adapter(lora_cfg)

pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)
for n, p in pipe.unet.named_parameters():
    if "lora" in n:
        p.requires_grad_(True)
controlnet.train(); controlnet.requires_grad_(True) 

params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
params.extend(controlnet.parameters()) 
optimizer = torch.optim.AdamW(params, lr=CFG.LR)

clip_encoder = CLIPVisionModel.from_pretrained(CFG.CLIP_MODEL, torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32)
clip_processor = CLIPImageProcessor.from_pretrained(CFG.CLIP_MODEL)
clip_encoder.eval()

def get_clip_features(imgs):
    pil_list = [(img.detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5).clip(0,1) * 255 for img in imgs]
    pil_list = [Image.fromarray(x.astype(np.uint8)) for x in pil_list]
    inputs = clip_processor(images=pil_list, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(accelerator.device)
    with torch.no_grad():
        features = clip_encoder(pixel_values=pixel_values).pooler_output
    return features

lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
lpips_loss_fn.eval()

pipe.unet, controlnet, optimizer, train_loader, val_loader, \
pipe.vae, pipe.text_encoder, clip_encoder, lpips_loss_fn = accelerator.prepare(
    pipe.unet, controlnet, optimizer, train_loader, val_loader,
    pipe.vae, pipe.text_encoder, clip_encoder, lpips_loss_fn
)
pipe.controlnet = controlnet 
weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

overall_epoch = 0
train_losses = []
val_losses = []
best_val_loss = float('inf')
no_improve = 0
resume_training = True
tracker_path = os.path.join(latest_model_dir, 'training_tracker.pt')

if resume_training and os.path.exists(latest_model_dir) and os.path.exists(tracker_path):
    accelerator.print(f"[INFO] 이전 체크포인트에서 복구: {latest_model_dir}")
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
        else:
            avg_train_loss = 0.0
        accelerator.print(f"[INFO] 복구된 epoch: {overall_epoch}, Best Val Loss: {best_val_loss:.4f}")
    except Exception as e:
        accelerator.print(f"[WARNING] 체크포인트 복구 실패, 새로 시작: {e}")
        overall_epoch = 0
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        no_improve = 0
        avg_train_loss = 0.0
else:
    accelerator.print("[INFO] 체크포인트 없음, 새로 학습 시작.")
    avg_train_loss = 0.0

for epoch in range(overall_epoch, CFG.EPOCHS):
    accelerator.print(f"\n--- Overall Epoch {epoch+1}/{CFG.EPOCHS} 시작 ---")
    pipe.unet.train()
    pipe.controlnet.train()

    total_train_loss_for_overall_epoch = 0
    total_train_steps_for_overall_epoch = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        with accelerator.accumulate(pipe.unet, pipe.controlnet):
            control_image = batch['input_control_image'].to(dtype=weight_dtype, device=accelerator.device)
            target_rgb = batch['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
            
            latents = pipe.vae.encode(target_rgb * 2.0 - 1.0).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            pos_prompt_input_ids = batch['pos_prompt_input_ids'].to(accelerator.device)
            text_embeddings = pipe.text_encoder(pos_prompt_input_ids)[0]
            
            down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image, return_dict=False
            )
            noise_pred = pipe.unet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, return_dict=False
            )[0]
            loss_noise = F.mse_loss(noise_pred.to(dtype=weight_dtype), noise.to(dtype=weight_dtype))
            accelerator.backward(loss_noise)
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pipe.controlnet.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss_for_overall_epoch += loss_noise.item()
            total_train_steps_for_overall_epoch += 1

    avg_train_loss = total_train_loss_for_overall_epoch / total_train_steps_for_overall_epoch
    train_losses.append(avg_train_loss)

    # 텐서보드 학습 손실 기록
    accelerator.log({"train_loss": avg_train_loss}, step=epoch)

    accelerator.print(f"\n--- Validation for Overall Epoch {epoch+1} ---")

    pipe.unet.eval()
    pipe.controlnet.eval()

    # 1. validation용 inference pipe 한 번만 생성 (메인 프로세스)
    if accelerator.is_main_process:
        inference_pipe_val = pipe.__class__.from_pretrained(
            CFG.PRETRAINED_MODEL_NAME_OR_PATH,
            unet=accelerator.unwrap_model(pipe.unet),
            controlnet=accelerator.unwrap_model(pipe.controlnet),
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            safety_checker=None,
            torch_dtype=weight_dtype
        ).to(accelerator.device)
    else:
        inference_pipe_val = None

    total_val_loss = 0
    total_val_batches = 0
    val_detail_loss_log = []

    with torch.no_grad():
        for i, batch_val in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
            control_image_tensor = batch_val['input_control_image'][0]
            control_image_for_pipe_list = [tensor_to_pil(control_image_tensor)]
            prompt_list = [batch_val['pos_prompt_str_for_pipe'][0]]
            neg_prompt_list = [batch_val['neg_prompt_str_for_pipe'][0]]
            guidance_scale_to_use = float(batch_val['guidance'][0])
            num_inference_steps_to_use = int(batch_val['steps'][0])
            original_filename = batch_val['file_name'][0]

            if accelerator.is_main_process:
                generated_images = inference_pipe_val(
                    prompt=prompt_list,
                    image=control_image_for_pipe_list,
                    negative_prompt=neg_prompt_list,
                    guidance_scale=guidance_scale_to_use,
                    num_inference_steps=num_inference_steps_to_use,
                    output_type="pt"
                ).images
            else:
                generated_images = torch.zeros((1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE), device=accelerator.device, dtype=weight_dtype)

            target_rgb_val_01 = batch_val['gt_rgb_tensor'].to(dtype=weight_dtype, device=accelerator.device)
            generated_images_01 = (generated_images + 1) / 2.0

            val_l1_rgb = F.l1_loss(generated_images_01, target_rgb_val_01)

            clip_features_fake_val = get_clip_features(generated_images)
            clip_features_real_val = get_clip_features(target_rgb_val_01 * 2.0 - 1.0)
            val_clip_loss = F.mse_loss(clip_features_fake_val, clip_features_real_val)

            val_lpips_loss = lpips_loss_fn(generated_images_01, target_rgb_val_01).mean()
            val_ssim_loss = 1 - ssim(generated_images_01, target_rgb_val_01, data_range=1.0, size_average=True)

            val_gt_ab_channels_01 = batch_val['ab_channels'].to(device=accelerator.device, dtype=weight_dtype)

            val_pred_rgb_np_01 = generated_images_01.permute(0, 2, 3, 1).cpu().numpy()
            val_pred_lab_np_list = [color.rgb2lab(np.clip(img_np, 0.0, 1.0)) for img_np in val_pred_rgb_np_01]
            val_pred_lab_tensor = torch.stack([
                torch.from_numpy(lab_img).float().permute(2, 0, 1) for lab_img in val_pred_lab_np_list
            ]).to(accelerator.device, dtype=weight_dtype)

            val_pred_ab_channels_tensor = val_pred_lab_tensor[:, 1:, :, :]
            val_pred_ab_channels_tensor = (val_pred_ab_channels_tensor + 128) / 255.0
            val_pred_ab_channels_tensor = torch.clamp(val_pred_ab_channels_tensor, min=0.0, max=1.0)

            val_l1_lab = F.l1_loss(val_pred_ab_channels_tensor, val_gt_ab_channels_01)

            val_total_loss_item = CFG.LAMBDA_L1 * (val_l1_rgb.item() + val_l1_lab.item()) \
                                + CFG.LAMBDA_CLIP * val_clip_loss.item() \
                                + CFG.LAMBDA_LPIPS * val_lpips_loss.item() \
                                + CFG.LAMBDA_SSIM * val_ssim_loss.item()

            if accelerator.is_main_process:
                total_val_loss += val_total_loss_item
                total_val_batches += 1

                val_detail_loss_log.append({
                    "L1_RGB": val_l1_rgb.item(),
                    "L1_LAB": val_l1_lab.item(),
                    "CLIP": val_clip_loss.item(),
                    "LPIPS": val_lpips_loss.item(),
                    "SSIM": val_ssim_loss.item()
                })

                if i < 3:  # 상위 3개 샘플 이미지 저장
                    gt_image_pil = tensor_to_pil(batch_val['gt_rgb_tensor'][0])
                    input_control_pil = tensor_to_pil(batch_val['input_control_image'][0])
                    generated_pil = tensor_to_pil(generated_images[0].cpu())
                    combined_image = Image.new('RGB', (CFG.IMG_SIZE * 3, CFG.IMG_SIZE))
                    combined_image.paste(input_control_pil, (0, 0))
                    combined_image.paste(generated_pil, (CFG.IMG_SIZE, 0))
                    combined_image.paste(gt_image_pil, (CFG.IMG_SIZE * 2, 0))
                    sample_filename = f"sample_epoch{epoch+1:02d}_{original_filename}"
                    combined_image.save(os.path.join(sample_output_dir, sample_filename))
                    accelerator.print(f"    Sample image saved: {sample_filename}")

    # 2. validation 끝난 뒤 캐시 해제
    if inference_pipe_val is not None:
        del inference_pipe_val
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    avg_val_loss = total_val_loss / total_val_batches if total_val_batches > 0 else 0
    val_losses.append(avg_val_loss)

    if accelerator.is_main_process:
        accelerator.print(f"Overall Epoch {epoch+1} / {CFG.EPOCHS} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        last_val = val_detail_loss_log[-1] if val_detail_loss_log else {}
        accelerator.print(f"Val Losses - L1(RGB): {last_val.get('L1_RGB', 0):.4f}, L1(LAB): {last_val.get('L1_LAB', 0):.4f}, CLIP: {last_val.get('CLIP', 0):.4f}, LPIPS: {last_val.get('LPIPS', 0):.4f}, SSIM: {last_val.get('SSIM', 0):.4f}")

        # 텐서보드 로그 기록
        accelerator.log({
            "val_loss": avg_val_loss,
            "val_l1_rgb": last_val.get("L1_RGB", 0),
            "val_l1_lab": last_val.get("L1_LAB", 0),
            "val_clip": last_val.get("CLIP", 0),
            "val_lpips": last_val.get("LPIPS", 0),
            "val_ssim": last_val.get("SSIM", 0)
        }, step=epoch)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve = 0
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.save_state(best_model_dir)
            unwrapped_unet = accelerator.unwrap_model(pipe.unet)
            lora_weights = get_peft_model(unwrapped_unet, lora_cfg).state_dict()
            torch.save(lora_weights, os.path.join(best_model_dir, "lora_best.pt"))
            unwrapped_controlnet = accelerator.unwrap_model(pipe.controlnet)
            unwrapped_controlnet.save_pretrained(os.path.join(best_model_dir, "controlnet_best"))
            accelerator.print(f"Best model saved to {best_model_dir} with Val Loss: {best_val_loss:.4f}")
    else:
        no_improve += 1
        accelerator.print(f"Validation loss did not improve for {no_improve} epochs.")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.save_state(latest_model_dir)
        tracker_state = {
            'overall_epoch': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'no_improve': no_improve,
        }
        torch.save(tracker_state, tracker_path)
        accelerator.print(f"Latest checkpoint saved to {latest_model_dir}")

    if no_improve >= CFG.PATIENCE:
        accelerator.print(f"Early stopping triggered after {epoch+1} epochs due to no improvement for {CFG.PATIENCE} epochs.")
        break

accelerator.end_training()