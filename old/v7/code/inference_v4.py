import os
import gc
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
from peft import PeftModel
import open_clip
import zipfile
import re
import cv2
import string
import torch.nn.functional as F
import nltk
from collections import Counter
from transformers import CLIPTokenizer
from sklearn.cluster import KMeans
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

# NLTK punkt 토크나이저 다운로드 (필요시)
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    nltk.download('punkt')

# --- Config 클래스 정의 ---
class Config:
    def __init__(self):
        self.IMG_SIZE = 512
        self.SEED = 42
        self.OUTPUT_ROOT_DIR = "./output4"
        self.SUB_DIR = os.path.join(self.OUTPUT_ROOT_DIR, "submission")
        self.SUBMISSION_ZIP = os.path.join(self.OUTPUT_ROOT_DIR, "submission.zip")
        self.BEST_MODEL_DIR = os.path.join(self.OUTPUT_ROOT_DIR, 'lora_best_model')
        self.TEST_CSV = "../test.csv" 
        self.TEST_INPUT_DIR = "../" 
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.WEIGHT_DTYPE = torch.float16 if self.DEVICE == "cuda" else torch.float32
        self.num_inference_steps_for_submission = 50
        self.N_ATTEMPTS_PER_IMAGE = 3
        self.EMBED_MODEL = "ViT-L-14"
        self.EMBED_PRETRAINED = "openai"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.MAX_PROMPT_TOKENS = 55
        # NSFW 관련 키워드 및 대체 문구
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "xxx", "erotic", "nude", "breast", "ass", "vagina", "penis", "groping", "rape", "molest"] 
        self.SFW_CAPTION_REPLACEMENT = "a person" 

CFG = Config()
os.makedirs(CFG.SUB_DIR, exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 유틸리티 함수 정의 ---
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
        input_ids = input_ids[:max_len-1] 
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
    min_colors = {}
    for key, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = hex_to_rgb(key)
        rd = (r_c - rgb_tuple[0]) ** 2
        gd = (g_c - rgb_tuple[1]) ** 2
        bd = (b_c - rgb_tuple[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

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

basic_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

def preprocess_for_controlnet(image_pil, detector_type="canny", low=100, high=200):
    image_np = np.array(image_pil)
    if detector_type == "canny":
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edged_image = cv2.Canny(gray_image, low, high)
        control_image = Image.fromarray(edged_image).convert("RGB")
    else:
        control_image = image_pil 
    return control_image

def calc_hsv(image_pil):
    image_np = np.array(image_pil.convert("RGB"))
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    _, s_channel, _ = cv2.split(hsv_image)
    return np.mean(s_channel) / 255.0

def calc_clip_embedding(image_pil, clip_model, clip_preprocess, device):
    with torch.no_grad():
        processed_img = clip_preprocess(image_pil).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(processed_img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

def pick_best(candidates, text_features_for_clip_score, device):
    if not candidates:
        return None, None
    best_score = -float('inf')
    best_image = None
    best_embedding = None
    for img_pil, img_embedding_np, hsv_saturation_mean in candidates:
        # HSV 유사도와 CLIP Score를 결합하여 최종 점수 계산 (가중치는 조절 가능)
        hsv_similarity = hsv_saturation_mean 
        img_embedding_tensor = torch.tensor(img_embedding_np, device=device, dtype=torch.float32).unsqueeze(0) 
        clip_score = F.cosine_similarity(img_embedding_tensor, text_features_for_clip_score).item()
        current_total_score = 0.6 * hsv_similarity + 0.4 * clip_score
        
        if current_total_score > best_score:
            best_score = current_total_score
            best_image = img_pil
            best_embedding = img_embedding_np
    return best_image, best_embedding

# --- PEFT 래퍼 언랩핑 함수  ---
def get_peft_leaf_model(m):
    while True:
        m_type_str = str(type(m)).lower()
        if "peftmodel" in m_type_str or "loramodel" in m_type_str:
            if hasattr(m, "base_model") and m.base_model is not None:
                m = m.base_model
            elif hasattr(m, "model") and m.model is not None:
                m = m.model
            else:
                break
        else:
            break 
    return m

# --- 추론 실행 함수 ---
def run_inference(config, prompt_enhancer, dynamic_param_gen, basic_transform):
    print("\n--- 추론 파이프라인 시작 ---")
    
    # Stable Diffusion 및 ControlNet 기본 모델 로드
    base_controlnet = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=config.WEIGHT_DTYPE)
    base_unet = UNet2DConditionModel.from_pretrained(config.MODEL_PATH, subfolder="unet", torch_dtype=config.WEIGHT_DTYPE)
    
    # LoRA 가중치 로드 (PEFT 사용)
    lora_unet_dir = os.path.join(config.BEST_MODEL_DIR, "unet_lora")
    lora_controlnet_dir = os.path.join(config.BEST_MODEL_DIR, "controlnet_lora")
    
    # PeftModel로 기본 모델에 LoRA 어댑터 적용
    unet_lora = PeftModel.from_pretrained(base_unet, lora_unet_dir).to(config.DEVICE)
    controlnet_lora = PeftModel.from_pretrained(base_controlnet, lora_controlnet_dir).to(config.DEVICE)

    # 파이프라인에 전달하기 전에 PEFT 래퍼를 언랩핑 
    unet_for_pipe = get_peft_leaf_model(unet_lora)
    controlnet_for_pipe = get_peft_leaf_model(controlnet_lora)

    print(f"UNet type for pipeline: {type(unet_for_pipe)}")
    print(f"ControlNet type for pipeline: {type(controlnet_for_pipe)}")

    # Stable Diffusion ControlNet 파이프라인 설정
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.MODEL_PATH,
        unet=unet_for_pipe,      # 언랩핑된 UNet 모델 전달
        controlnet=controlnet_for_pipe, # 언랩핑된 ControlNet 모델 전달
        torch_dtype=config.WEIGHT_DTYPE,
        safety_checker=None    
    ).to(config.DEVICE)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.check_inputs = lambda *args, **kwargs: None # 입력 유효성 검사 비활성화

    # 모델을 평가 모드로 설정
    pipe.unet.eval()
    pipe.controlnet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    tokenizer = CLIPTokenizer.from_pretrained(config.MODEL_PATH, subfolder="tokenizer")

    # --- CLIP Score 계산을 위한 CLIP 임베딩 모델 로드 (대회 규정: ViT-L-14) ---
    clip_model_for_score, _, clip_preprocess_for_score = open_clip.create_model_and_transforms(
        config.EMBED_MODEL, pretrained=config.EMBED_PRETRAINED)
    clip_model_for_score = clip_model_for_score.to(config.DEVICE)
    clip_model_for_score.eval()
    clip_tokenizer_for_score = open_clip.get_tokenizer(config.EMBED_MODEL)
    
    # CLIP Score 계산에 사용될 텍스트 임베딩 
    with torch.no_grad():
        text_prompt_for_score = "A vibrant and colorful image, photorealistic, high quality"
        tokenized_text = clip_tokenizer_for_score(text_prompt_for_score).to(config.DEVICE)
        text_features_for_clip_score = clip_model_for_score.encode_text(tokenized_text)
        text_features_for_clip_score = text_features_for_clip_score / text_features_for_clip_score.norm(dim=-1, keepdim=True)

    # --- 대회 제출용 최종 이미지 임베딩 추출 모델 로드 (대회 규정: ViT-L-14) ---
    clip_model_for_submission, _, clip_preprocess_for_submission = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai") 
    clip_model_for_submission.to(config.DEVICE)
    clip_model_for_submission.eval()

    # --- 테스트 데이터 로드 및 프롬프트 전처리 ---
    test_df = pd.read_csv(config.TEST_CSV)
    all_test_captions = test_df['caption'].astype(str).tolist()
    remove_phrases_inference = build_remove_phrases(all_test_captions, ngram_ns=(2,3,4), topk=100)

     # --- 추론 루프 및 최종 임베딩 추출 ---
    final_output_img_names = []
    final_output_embeddings_for_submission = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = str(row['ID'])
        caption = row['caption']
        input_img_filename = row.get('input_img_path', f"{img_id}.png")
        input_img_path = os.path.join(config.TEST_INPUT_DIR, input_img_filename)

        try:
            input_img_pil = Image.open(input_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {input_img_path}: {e}")
            continue

        # dominant color name 추출 (여기서 단 1회)
        dominant_colors = extract_dominant_colors(input_img_pil, topk=3)
        color_names = [rgb_to_simple_color_name(c) for c in dominant_colors]
        color_names = list(dict.fromkeys(color_names))[:3]
        color_str = ', '.join(color_names)

        # 캡션 전처리
        cleaned_caption_raw = clean_caption_full(
            caption, remove_phrases_inference, number_words, number_regex, max_tokens=70
        )

        # NSFW 처리 (없으면 원본 사용)
        is_nsfw = False
        if hasattr(config, 'NSFW_KEYWORDS') and hasattr(config, 'SFW_CAPTION_REPLACEMENT'):
            nsfw_keywords = [k.lower() for k in config.NSFW_KEYWORDS]
            sfw_caption_replacement = config.SFW_CAPTION_REPLACEMENT
            cleaned_caption_lower = cleaned_caption_raw.lower()
            for nsfw_kw in nsfw_keywords:
                if nsfw_kw in cleaned_caption_lower:
                    is_nsfw = True
                    break
            if is_nsfw:
                cleaned_caption = sfw_caption_replacement
            else:
                cleaned_caption = cleaned_caption_raw
        else:
            cleaned_caption = cleaned_caption_raw

        # 프롬프트 로그 1~2개 찍기
        if idx < 2:
            print(f"[Prompt Sample {idx+1}] dominant colors: {color_names} | caption: {cleaned_caption}")

        candidates = []
        for attempt in range(config.N_ATTEMPTS_PER_IMAGE):
            seed_everything(config.SEED + idx * config.N_ATTEMPTS_PER_IMAGE + attempt)

            # 프롬프트 리스트 초기화
            current_pos_prompt_parts = [color_str, cleaned_caption]
            enhancement_keywords_list = prompt_enhancer.get_enhancement_keywords(cleaned_caption)
            random.shuffle(enhancement_keywords_list)

            for keyword_phrase in enhancement_keywords_list:
                temp_prompt = ", ".join(current_pos_prompt_parts + [keyword_phrase])
                temp_token_ids = tokenizer.encode(
                    temp_prompt,
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors="pt"
                )[0]
                if len(temp_token_ids) <= config.MAX_PROMPT_TOKENS:
                    current_pos_prompt_parts.append(keyword_phrase)
                else:
                    break

            pos_prompt_str_raw = ", ".join(current_pos_prompt_parts)
            pos_prompt = safe_prompt_str(pos_prompt_str_raw, tokenizer, config.MAX_PROMPT_TOKENS)

            neg_prompt = prompt_enhancer.get_base_negative_prompt(cleaned_caption)
            neg_prompt = safe_prompt_str(neg_prompt, tokenizer, config.MAX_PROMPT_TOKENS)

            # 동적 파라미터 생성
            guidance_scale = dynamic_param_gen.get_optimal_guidance(cleaned_caption)
            num_inference_steps = dynamic_param_gen.get_optimal_steps(cleaned_caption)
            canny_low, canny_high = dynamic_param_gen.get_optimal_canny_params(cleaned_caption)

            # ControlNet 입력 이미지 전처리
            control_image = preprocess_for_controlnet(input_img_pil, detector_type="canny", low=canny_low, high=canny_high)

            # Stable Diffusion 추론 실행
            output = pipe(
                prompt=pos_prompt,
                image=control_image,
                negative_prompt=neg_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil"
            )
            current_img_pil = output.images[0]

            # CLIP Score 계산 및 HSV 채도 계산
            current_img_clip_embedding_for_score = calc_clip_embedding(current_img_pil, clip_model_for_score, clip_preprocess_for_score, config.DEVICE)
            current_img_hsv_saturation = calc_hsv(current_img_pil)
            candidates.append((current_img_pil, current_img_clip_embedding_for_score, current_img_hsv_saturation))

            torch.cuda.empty_cache(); gc.collect()

        # 여러 시도 중 최적의 이미지 선택
        best_img, best_embedding_for_score = pick_best(candidates, text_features_for_clip_score, config.DEVICE)

        if best_img is not None:
            # 생성된 이미지 저장
            file_name = f"{img_id}.png"
            best_img.save(os.path.join(config.SUB_DIR, file_name))
            final_output_img_names.append(file_name)

            # 대회 규정에 따른 최종 이미지 임베딩 추출 (ViT-L-14, L2 정규화 필수)
            processed_img_for_submission = clip_preprocess_for_submission(best_img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                feat_img_for_submission = clip_model_for_submission.encode_image(processed_img_for_submission)
                feat_img_for_submission /= feat_img_for_submission.norm(dim=-1, keepdim=True)  # L2 정규화 필수

            final_output_embeddings_for_submission.append(feat_img_for_submission.detach().cpu().numpy().reshape(-1))

    print('모든 이미지 생성 및 임베딩 추출 완료.')
    # 이하 동일...