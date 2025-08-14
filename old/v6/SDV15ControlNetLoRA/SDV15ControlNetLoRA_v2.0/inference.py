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
        self.OUTPUT_ROOT_DIR = "./output2"
        self.SUB_DIR = os.path.join(self.OUTPUT_ROOT_DIR, "submission")
        self.SUBMISSION_ZIP = os.path.join(self.OUTPUT_ROOT_DIR, "submission.zip")
        # LoRA 모델이 저장된 경로를 실제 경로에 맞게 설정해주세요.
        self.BEST_MODEL_DIR = os.path.join(self.OUTPUT_ROOT_DIR, 'lora_best_model')
        self.TEST_CSV = "../test.csv" # 테스트 CSV 파일 경로
        self.TEST_INPUT_DIR = "../" # 입력 이미지 폴더 경로 (test.csv와 같은 디렉토리)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.WEIGHT_DTYPE = torch.float16 if self.DEVICE == "cuda" else torch.float32
        self.num_inference_steps_for_submission = 50
        self.N_ATTEMPTS_PER_IMAGE = 3
        # 대회 규정에 따른 CLIP 임베딩 모델
        self.EMBED_MODEL = "ViT-L-14"
        self.EMBED_PRETRAINED = "openai"
        # Stable Diffusion 및 ControlNet 기본 모델 경로
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.MAX_PROMPT_TOKENS = 77
        # NSFW 관련 키워드 및 대체 문구 (선택 사항)
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "xxx", "erotic", "nude", "breast", "ass", "vagina", "penis", "groping", "rape", "molest"] 
        self.SFW_CAPTION_REPLACEMENT = "a person" 

CFG = Config()
os.makedirs(CFG.SUB_DIR, exist_ok=True)

# --- 유틸리티 함수 정의 ---

# 학습 코드에서 가져온 color_words 및 number_words, number_regex 정의
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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        current_total_score = 0.6 * hsv_similarity + 0.4 * clip_score # 예시 가중치
        
        if current_total_score > best_score:
            best_score = current_total_score
            best_image = img_pil
            best_embedding = img_embedding_np
    return best_image, best_embedding

# --- PEFT 래퍼 언랩핑 함수 (강화된 버전) ---
def get_peft_leaf_model(m):
    """
    PEFT (LoRA, PeftModel 등) 래퍼를 완전히 해제하고
    기반이 되는 원본 모델 (예: UNet2DConditionModel, ControlNetModel)을 반환합니다.
    """
    while True:
        m_type_str = str(type(m)).lower()
        if "peftmodel" in m_type_str or "loramodel" in m_type_str:
            if hasattr(m, "base_model") and m.base_model is not None:
                m = m.base_model
            elif hasattr(m, "model") and m.model is not None:
                m = m.model
            else:
                break # 더 이상 언랩핑할 속성이 없으면 종료
        else:
            break # PEFT 래퍼가 아니면 종료
    return m

# --- 추론 실행 함수 ---
def run_inference(config, prompt_enhancer, dynamic_param_gen, basic_transform):
    print("\n--- 추론 파이프라인 시작 ---")
    
    # 1. Stable Diffusion 및 ControlNet 기본 모델 로드
    base_controlnet = ControlNetModel.from_pretrained(config.CONTROLNET_PATH, torch_dtype=config.WEIGHT_DTYPE)
    base_unet = UNet2DConditionModel.from_pretrained(config.MODEL_PATH, subfolder="unet", torch_dtype=config.WEIGHT_DTYPE)
    
    # 2. LoRA 가중치 로드 (PEFT 사용)
    lora_unet_dir = os.path.join(config.BEST_MODEL_DIR, "unet_lora")
    lora_controlnet_dir = os.path.join(config.BEST_MODEL_DIR, "controlnet_lora")
    
    # PeftModel로 기본 모델에 LoRA 어댑터 적용
    unet_lora = PeftModel.from_pretrained(base_unet, lora_unet_dir).to(config.DEVICE)
    controlnet_lora = PeftModel.from_pretrained(base_controlnet, lora_controlnet_dir).to(config.DEVICE)

    # *** 파이프라인에 전달하기 전에 PEFT 래퍼를 언랩핑 ***
    unet_for_pipe = get_peft_leaf_model(unet_lora)
    controlnet_for_pipe = get_peft_leaf_model(controlnet_lora)

    print(f"UNet type for pipeline: {type(unet_for_pipe)}")
    print(f"ControlNet type for pipeline: {type(controlnet_for_pipe)}")
    # 예상 출력:
    # UNet type for pipeline: <class 'diffusers.models.unet_2d_condition.UNet2DConditionModel'>
    # ControlNet type for pipeline: <class 'diffusers.models.controlnet.ControlNetModel'>

    # 3. Stable Diffusion ControlNet 파이프라인 설정
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.MODEL_PATH,
        unet=unet_for_pipe,      # 언랩핑된 UNet 모델 전달
        controlnet=controlnet_for_pipe, # 언랩핑된 ControlNet 모델 전달
        torch_dtype=config.WEIGHT_DTYPE,
        safety_checker=None      # 안전 검사기 비활성화 (권장하지 않음, 대회 목적상 임시)
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

    # 4. CLIP Score 계산을 위한 CLIP 임베딩 모델 로드 (대회 규정: ViT-L-14)
    clip_model_for_score, _, clip_preprocess_for_score = open_clip.create_model_and_transforms(
        config.EMBED_MODEL, pretrained=config.EMBED_PRETRAINED)
    clip_model_for_score = clip_model_for_score.to(config.DEVICE)
    clip_model_for_score.eval()
    clip_tokenizer_for_score = open_clip.get_tokenizer(config.EMBED_MODEL)
    
    # CLIP Score 계산에 사용될 텍스트 임베딩 (정적인 기준 프롬프트)
    with torch.no_grad():
        text_prompt_for_score = "A vibrant and colorful image, photorealistic, high quality"
        tokenized_text = clip_tokenizer_for_score(text_prompt_for_score).to(config.DEVICE)
        text_features_for_clip_score = clip_model_for_score.encode_text(tokenized_text)
        text_features_for_clip_score = text_features_for_clip_score / text_features_for_clip_score.norm(dim=-1, keepdim=True)

    # 5. 대회 제출용 최종 이미지 임베딩 추출 모델 로드 (대회 규정: ViT-L-14)
    # CLIP Score 계산용과 동일한 모델이지만 역할이 다름
    clip_model_for_submission, _, clip_preprocess_for_submission = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai") 
    clip_model_for_submission.to(config.DEVICE)
    clip_model_for_submission.eval()

    # 6. 테스트 데이터 로드 및 프롬프트 전처리
    test_df = pd.read_csv(config.TEST_CSV)
    all_test_captions = test_df['caption'].astype(str).tolist()
    remove_phrases_inference = build_remove_phrases(all_test_captions, ngram_ns=(2,3,4), topk=100)

    # 7. 추론 루프 및 최종 임베딩 추출
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

        # 캡션 전처리 (불필요한 구문 제거, 숫자 제거 등)
        cleaned_caption_raw = clean_caption_full(caption, remove_phrases_inference, number_words, number_regex, max_tokens=70)
        
        # NSFW 키워드 처리 (Config에 정의된 경우)
        is_nsfw = False
        if hasattr(CFG, 'NSFW_KEYWORDS') and hasattr(CFG, 'SFW_CAPTION_REPLACEMENT'):
            nsfw_keywords = [k.lower() for k in CFG.NSFW_KEYWORDS]
            sfw_caption_replacement = CFG.SFW_CAPTION_REPLACEMENT
            cleaned_caption_lower = cleaned_caption_raw.lower()
            for nsfw_kw in nsfw_keywords:
                if nsfw_kw in cleaned_caption_lower:
                    is_nsfw = True
                    break
            if is_nsfw:
                current_cleaned_caption_for_processing = sfw_caption_replacement
            else:
                current_cleaned_caption_for_processing = cleaned_caption_raw
        else:
            current_cleaned_caption_for_processing = cleaned_caption_raw

        cleaned_caption = current_cleaned_caption_for_processing

        candidates = []
        for attempt in range(config.N_ATTEMPTS_PER_IMAGE):
            seed_everything(config.SEED + idx * config.N_ATTEMPTS_PER_IMAGE + attempt) # 시드 고정으로 재현성 확보
            
            current_pos_prompt_parts = [cleaned_caption]
            enhancement_keywords_list = prompt_enhancer.get_enhancement_keywords(cleaned_caption)
            random.shuffle(enhancement_keywords_list) # 키워드 순서 랜덤화

            # 최대 프롬프트 토큰 수에 맞춰 동적으로 키워드 추가
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

            # *** 대회 규정에 따른 최종 이미지 임베딩 추출 (ViT-L-14, L2 정규화 필수) ***
            processed_img_for_submission = clip_preprocess_for_submission(best_img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                feat_img_for_submission = clip_model_for_submission.encode_image(processed_img_for_submission)
                feat_img_for_submission /= feat_img_for_submission.norm(dim=-1, keepdim=True) # L2 정규화 필수
            
            final_output_embeddings_for_submission.append(feat_img_for_submission.detach().cpu().numpy().reshape(-1))

    print('모든 이미지 생성 및 임베딩 추출 완료.')

    # *** 평가 제출용 임베딩 CSV 파일 생성 및 저장 ***
    if final_output_img_names:
        feat_imgs_array = np.array(final_output_embeddings_for_submission)
        vec_columns = [f'vec_{i}' for i in range(feat_imgs_array.shape[1])]
        feat_submission = pd.DataFrame(feat_imgs_array, columns=vec_columns)
        feat_submission.insert(0, 'ID', final_output_img_names)
        csv_path = os.path.join(config.SUB_DIR, 'embed_submission.csv') 
        feat_submission.to_csv(csv_path, index=False)
        print(f"평가 제출용 임베딩 CSV 저장 완료: {csv_path}")

    # 최종 결과물 ZIP 파일로 압축
    print("ZIP 파일 생성 중...")
    if os.path.exists(config.SUB_DIR) and os.listdir(config.SUB_DIR):
        zip_path = config.SUBMISSION_ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.DEFLATED) as zipf:
            for file_name in os.listdir(config.SUB_DIR):
                file_path = os.path.join(config.SUB_DIR, file_name)
                # 숨김 파일 제외하고 .png와 .csv 파일만 포함
                if os.path.isfile(file_path) and not file_name.startswith('.') and (file_name.endswith('.png') or file_name.endswith('.csv')):
                    zipf.write(file_path, arcname=file_name)
        print(f"압축 완료: {zip_path}")
    else:
        print(f"생성된 파일이 없어 {config.SUB_DIR} 폴더 압축을 건너뜀.")


# --- 스크립트 실행 부분 ---
if __name__ == "__main__":
    # NLTK 데이터 다운로드 확인 (스크립트 시작 시 한 번만 수행)
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, OSError):
        nltk.download('punkt')

    prompt_enhancer = PromptEnhancer()
    dynamic_param_gen = DynamicParameterGenerator()
    
    run_inference(CFG, prompt_enhancer, dynamic_param_gen, basic_transform)