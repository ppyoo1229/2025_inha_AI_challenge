# Stable Diffusion XL + ControlNet + CLIP 점수 기반 앙상블 추론 파이프라인
# 프롬프트 정제/강화 및 동적 파라미터 조정
# ControlNet + SDXL 파이프라인을 통한 이미지 생성 / 앙상블 및 단일모드 
# CLIP 점수 기반 최적 샘플 선택 (성능 앙상블)
# Python 3.8+ / torch / diffusers / open_clip / PIL 
# 로컬 모델 파일 활용 (Stable Diffusion XL, ControlNet, OpenCLIP)

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import random
import os
import zipfile
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import open_clip
import warnings
import re

warnings.filterwarnings('ignore')

os.chdir('/home/guest01/colorize/')
ROOT_PATH = '/home/guest01/colorize/'
TEST_CSV_PATH = os.path.join(ROOT_PATH, 'test.csv')
IMG_ROOT = os.path.join(ROOT_PATH, 'test/input_image')
SUB_DIR = os.path.join(ROOT_PATH, 'submission')
ZIP_PATH = os.path.join(ROOT_PATH, 'submission.zip')
os.makedirs(SUB_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 설정 ---
CFG = {
    'SUB_DIR': './submission',
    'SEED': 42,
    'USE_ENHANCED_PROMPTS': True,
    'USE_NEGATIVE_PROMPTS': True,
    'USE_DYNAMIC_GUIDANCE': True,
    'USE_ENSEMBLE': True,
    'NUM_ENSEMBLE_SAMPLES': 2,
    'USE_ADAPTIVE_STEPS': True,
    'USE_CLIP_SCORING': True,
    'USE_CANNY_OPTIMIZATION': True,
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG['SEED'])

# ========== 1. 카테고리 사전 및 키워드 ==========
category_dict= {
    'color': [
        'white', 'black', 'red', 'green', 'blue', 'brown', 'yellow', 'orange', 'pink',
        'gray', 'silver', 'purple', 'grey', 'gold', 'beige', 'colorful', 'colour'
    ],
    'person': [
        'man', 'woman', 'person', 'people', 'boy', 'girl', 'persons', 'face', 'head',
        'hand', 'hands', 'hair', 'ear', 'ears', 'arm', 'arms', 'eye', 'eyes'
    ],
    'clothes': [
        'pants', 'shirt', 'jacket', 'dress', 'hat', 'cap', 'skirt', 'jeans', 'shorts',
        'coat', 'boots', 'shoes', 'scarf', 'bag'
    ],
    'animal': [
        'dog', 'cat', 'bear', 'elephant', 'cow', 'sheep', 'horse', 'giraffe'
    ],
    'action': [
        'sitting', 'standing', 'holding', 'riding', 'walking', 'smiling', 'running',
        'wearing', 'hanging', 'covering', 'eating', 'playing', 'watching', 'lying', 'leaning'
    ],
    'object': [
        'car', 'bus', 'train', 'truck', 'bicycle', 'bike', 'motorcycle', 'horse', 'dog',
        'cat', 'bear', 'elephant', 'cow', 'sheep', 'flower', 'flowers', 'tree', 'trees',
        'grass', 'pole', 'sign', 'bottle', 'bowl', 'plate', 'food', 'bread', 'cheese',
        'fruit', 'meat', 'container', 'paper', 'box', 'book', 'chair', 'table', 'bench',
        'window', 'clock', 'ball', 'bat', 'player', 'players', 'skateboard', 'skis', 'ski',
        'poles', 'vase', 'mat', 'tray', 'frame', 'mirror', 'device', 'phone', 'cup', 'spoon',
        'fork', 'knife', 'pot', 'pan', 'jar', 'can', 'flag', 'banner', 'bag', 'cart', 'trolley',
        'bed', 'sofa', 'couch', 'blanket', 'pillow', 'lamp', 'light', 'lights', 'neon', 'screen',
        'monitor', 'keyboard'
    ],
    'scene': [
        'background', 'wall', 'floor', 'ground', 'road', 'street', 'sidewalk', 'sky',
        'building', 'buildings', 'house', 'room', 'mountain', 'field', 'clouds', 'park',
        'window', 'door', 'roof', 'fence', 'tree', 'trees', 'bush', 'hill', 'water',
        'river', 'lake', 'snow', 'surface', 'bench', 'bridge', 'shelf', 'table', 'park',
        'grass', 'field', 'ground', 'track', 'post', 'sign', 'signboard', 'street',
        'sidewalk', 'railway', 'rails', 'tracks', 'platform', 'station'
    ]
}
ACTION_WORDS = [
    'sitting', 'standing', 'wearing', 'holding', 'riding', 'walking', 'smiling',
    'running', 'covering', 'eating', 'playing', 'watching', 'lying', 'leaning'
]
POSITION_WORDS = [
    'on', 'in', 'at', 'behind', 'near', 'under', 'over', 'inside', 'beside',
    'left', 'right', 'top', 'bottom', 'middle', 'center', 'front', 'back', 'around', 'between'
]
SCENE_WORDS = [
    'background', 'ground', 'sky', 'wall', 'field', 'mountain', 'building',
    'street', 'road', 'grass', 'clouds', 'snow', 'lake', 'river', 'outdoors', 'indoors'
]
PERSON_WORDS = ['man', 'woman', 'person', 'people', 'boy', 'girl']
OBJECT_WORDS = [
    'car', 'train', 'bus', 'bottle', 'dog', 'cat', 'table', 'chair', 'tree',
    'flower', 'food', 'animal', 'plate', 'bowl', 'window', 'sign', 'jacket', 'shirt'
]
CLOTHES_WORDS = [
    'shirt', 'jacket', 'dress', 'hat', 'coat', 'jeans', 'pants', 'boots', 'shoes'
]
COLOR_WORDS = [
    'white', 'black', 'red', 'green', 'blue', 'yellow', 'orange', 'pink',
    'gray', 'brown', 'silver', 'purple', 'grey'
]

# ========== 2. 클린 프롬프트 생성 ==========
def extract_keywords(text, category_dict):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    result = []
    for word, pos in tagged:
        word = word.lower()
        for cat in category_dict:
            if word in category_dict[cat] and word not in result:
                result.append(word)
    return result
def make_clean_prompt(text, category_dict, style_tail="vibrant colors, photorealistic, high quality"):
    kw = extract_keywords(text, category_dict)
    if kw:
        prompt = " ".join(kw).capitalize() + ", " + style_tail
    else:
        prompt = style_tail
    return prompt

# ========== 3. 프롬프트 강화 클래스 ==========
class PromptEnhancer:
    def __init__(self, category_dict):
        self.category_dict = category_dict
        self.color_enhancements = {
            'red':      ['vibrant red', 'crimson red', 'cherry red', 'ruby red'],
            'blue':     ['deep blue', 'azure blue', 'sapphire blue', 'navy blue'],
            'green':    ['lush green', 'emerald green', 'forest green', 'jade green'],
            'yellow':   ['bright yellow', 'golden yellow', 'sunny yellow', 'amber yellow'],
            'orange':   ['warm orange', 'burnt orange', 'tangerine orange', 'coral orange'],
            'purple':   ['rich purple', 'violet purple', 'lavender purple', 'amethyst purple'],
            'pink':     ['soft pink', 'rose pink', 'magenta pink', 'blush pink'],
            'brown':    ['warm brown', 'chocolate brown', 'coffee brown', 'chestnut brown'],
            'black':    ['deep black', 'charcoal black', 'ebony black', 'jet black'],
            'white':    ['pure white', 'ivory white', 'pearl white', 'snow white'],
            'gray':     ['light gray', 'silver gray', 'slate gray'],
        }
        self.quality_keywords = [
            "photorealistic, masterpiece", "ultra detailed, high resolution",
            "award winning, cinematic lighting", "professional photography, vivid colors",
            "DSLR, finely detailed, rich colors", "trending on artstation, beautiful lighting",
            "fine art, natural light, vibrant colors", "incredibly realistic, beautiful composition",
        ]
        self.texture_keywords = [
            "intricate textures", "realistic skin texture", "velvet surface",
            "subtle film grain", "smooth gradient", "oil painting effect",
        ]
        self.lighting_keywords = [
            "cinematic lighting", "soft golden backlight", "dramatic lighting",
            "studio lighting", "diffused natural light", "dynamic shadows",
        ]
        self.scene_keywords = [
            "detailed background", "lush scenery", "dynamic composition",
            "peaceful mood", "immersive environment", "artistic framing"
        ]
        self.negative_prompts = [
            "blurry, low quality, pixelated, artifacts, oversaturated",
            "monochrome, black and white, sepia, desaturated, dull",
            "cartoonish, anime style, artificial colors, flat lighting",
            "watermark, text, signature, logo, border, frame"
        ]
    def enhance_caption(self, caption, max_per_cat=1):
        caption_proc = caption.lower()
        for color, enhancements in self.color_enhancements.items():
            if color in caption_proc:
                caption_proc = caption_proc.replace(color, random.choice(enhancements))
        # 핵심 명사/장면 추출
        words = re.findall(r'\b[a-zA-Z]+\b', caption_proc)
        used = set()
        keywords = []
        for cat in ['color', 'person', 'clothes', 'object', 'scene', 'action']:
            picked = [w for w in words if w in self.category_dict[cat] and w not in used]
            if picked:
                keywords.extend(picked[:max_per_cat])
                used.update(picked[:max_per_cat])
        style_part = random.choice(self.quality_keywords)
        others = random.sample(self.texture_keywords + self.lighting_keywords + self.scene_keywords, k=2)
        prompt_parts = [style_part] + others + keywords + ["vibrant colors"]
        # 35토큰 이하로 제한
        prompt_txt = ", ".join(prompt_parts)
        tokens = re.findall(r'\b[a-zA-Z]+\b', prompt_txt)
        if len(tokens) > 35:
            trimmed = []
            cnt = 0
            for part in prompt_parts:
                words_in_part = re.findall(r'\b[a-zA-Z]+\b', part)
                if cnt + len(words_in_part) > 35:
                    break
                trimmed.append(part)
                cnt += len(words_in_part)
            prompt_txt = ", ".join(trimmed)
        negative_prompt = random.choice(self.negative_prompts)
        return prompt_txt, negative_prompt

# ========== 4. 파라미터 생성기 ==========
class DynamicParameterGenerator:
    def __init__(self):
        self.guidance_ranges = {
            'action': (7.8, 8.5),
            'scene': (7.3, 8.1),
            'person_object': (6.9, 7.6),
            'color': (6.7, 7.3),
            'default': (7.0, 7.5)
        }
        self.step_ranges = {
            'simple': (32, 38),
            'complex': (50, 58),
            'default': (42, 48)
        }
    def get_optimal_guidance(self, caption):
        cap = caption.lower()
        if any(w in cap for w in ACTION_WORDS + POSITION_WORDS):
            return random.uniform(*self.guidance_ranges['action'])
        if any(w in cap for w in SCENE_WORDS):
            return random.uniform(*self.guidance_ranges['scene'])
        if any(w in cap for w in PERSON_WORDS + OBJECT_WORDS + CLOTHES_WORDS):
            return random.uniform(*self.guidance_ranges['person_object'])
        if any(w in cap for w in COLOR_WORDS):
            return random.uniform(*self.guidance_ranges['color'])
        return random.uniform(*self.guidance_ranges['default'])
    def get_optimal_steps(self, caption):
        wc = len(caption.split())
        if wc <= 6:
            return random.randint(*self.step_ranges['simple'])
        elif wc >= 15:
            return random.randint(*self.step_ranges['complex'])
        else:
            return random.randint(*self.step_ranges['default'])
    def get_optimal_canny_params(self, caption=""):
        if any(w in caption.lower() for w in ['scene', 'field', 'mountain', 'park', 'background', 'sky']):
            return random.randint(120, 140), random.randint(220, 240)
        else:
            return random.randint(100, 120), random.randint(200, 220)

# ========== 5. CLIP 점수 계산기 ==========
class CLIPScorer:
    def __init__(self, clip_model, clip_preprocess):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
    def calculate_clip_score(self, image, caption):
        try:
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(device)
            text_tokens = open_clip.tokenize([caption]).to(device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = torch.cosine_similarity(image_features, text_features).item()
            return similarity
        except Exception as e:
            print(f"CLIP 점수 계산 실패: {e}")
            return 0.0

# ========== 6. 모델 로딩 ==========
CONTROLNET_PATH_LOCAL = '/home/guest01/model/controlnet-canny-sdxl-1.0'
SDXL_PATH_LOCAL = '/home/guest01/model/stabilityaistable-diffusion-xl-base-1.0'
CLIP_PATH_LOCAL = '/home/guest01/model/openclip-vit-large-patch14'  # CLIP 저장 경로

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ControlNet
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_PATH_LOCAL,
    local_files_only=True
).to(device)

# SDXL + ControlNet 파이프라인 로드
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SDXL_PATH_LOCAL,
    controlnet=controlnet,
    local_files_only=True
).to(device)

# CLIP
print("CLIP 모델 로딩 중...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained=CLIP_PATH_LOCAL,      
    local_files_only=True              
)
clip_model.to(device)
clip_model.eval()
print("모델 로딩 완료")

# ========== 7. 컴포넌트 초기화 ==========
prompt_enhancer = PromptEnhancer(category_dict)
clip_scorer = CLIPScorer(clip_model, clip_preprocess)
param_generator = DynamicParameterGenerator()

# ========== 8. Canny ==========
def preprocess_for_controlnet_advanced(image: Image.Image, low_threshold=100, high_threshold=200):
    canny_detector = CannyDetector()
    image_np = np.array(image)
    control_image_np = canny_detector(image_np, low_threshold=low_threshold, high_threshold=high_threshold)
    return Image.fromarray(control_image_np)

# ========== 9. Cleaned 프롬프트 생성 ==========
df = pd.read_csv(TEST_CSV_PATH)
patterns = [
    r'do you see', r'what is', r'is there', r'can you', r'please', r'maybe', r'possibly', r'probably',
    r'find', r'tell me', r'show me', r'could be', r'might be', r'should be', r'appears? to be', r'looks like',
    r'try to', r'look at', r'imagine', r'guess what', r'one can see', r'describe', r'identify', r'how many',
    r'which', r'where', r'it might', r'they might'
]
combined_pattern = r'|'.join(patterns)
bad_mask = df['caption'].str.contains(combined_pattern, case=False, regex=True)
# cleaned_prompt_list 생성
cleaned_prompts = [
    make_clean_prompt(caption, category_dict) if bad
    else caption + ", vibrant colors, photorealistic, high quality"
    for caption, bad in zip(df['caption'], bad_mask)
]
df['cleaned_prompt'] = cleaned_prompts

# ========== 10. 앙상블/단일 추론 ==========
# 상수 선언
INFER_SIZE = 1024
SUBMIT_SIZE = 512

# 업스케일/다운스케일 함수
def upscale(img, size=INFER_SIZE):
    return img.resize((size, size), resample=Image.BICUBIC)

def downscale(img, size=SUBMIT_SIZE):
    return img.resize((size, size), resample=Image.LANCZOS)

# 앙상블 이미지 생성 함수
def generate_ensemble(input_img, caption, img_id):
    print(f" 앙상블 생성 ({CFG['NUM_ENSEMBLE_SAMPLES']}개 샘플)")
    candidates, scores = [], []
    # --- 업샘플
    input_img_up = upscale(input_img, INFER_SIZE)
    for i in range(CFG['NUM_ENSEMBLE_SAMPLES']):
        try:
            # guidance, step 등 파라미터 생성
            guidance_scale = param_generator.get_optimal_guidance(caption)
            num_steps = param_generator.get_optimal_steps(caption)
            low_thresh, high_thresh = param_generator.get_optimal_canny_params(caption)
            # ControlNet 입력(업샘플) 생성
            control_image = preprocess_for_controlnet_advanced(
                input_img_up, low_threshold=low_thresh, high_threshold=high_thresh
            )
            # 프롬프트 생성
            enhanced_prompt, negative_prompt = prompt_enhancer.enhance_caption(caption)
            # 파이프라인 실행 (업샘플 이미지 사용)
            output_img = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=device).manual_seed(CFG['SEED'] + i)
            ).images[0]
            # --- 다운샘플
            output_img_down = downscale(output_img, SUBMIT_SIZE)
            candidates.append(output_img_down)
            # CLIP 점수 계산
            if CFG['USE_CLIP_SCORING']:
                clip_score = clip_scorer.calculate_clip_score(output_img_down, caption)
                scores.append(clip_score)
                print(f"      CLIP 점수: {clip_score:.3f}")
            else:
                scores.append(random.random())
        except Exception as e:
            print(f"    샘플 {i+1} 생성 실패: {e}")
            continue
    # 후보 중 최고 점수 선택
    if candidates:
        best_idx = np.argmax(scores)
        best_image = candidates[best_idx]
        best_score = scores[best_idx]
        print(f"   최고 점수: {best_score:.3f} (샘플 {best_idx+1})")
        return best_image
    else:
        print(f"  앙상블 실패, 기본 생성으로 대체")
        return generate_single_image(input_img, caption)

# 단일 이미지 생성 함수
def generate_single_image(input_img, caption):
    # --- 업샘플
    input_img_up = upscale(input_img, INFER_SIZE)
    # ControlNet 입력 생성
    control_image = preprocess_for_controlnet_advanced(input_img_up)
    enhanced_prompt, negative_prompt = prompt_enhancer.enhance_caption(caption)
    output_img = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        guidance_scale=7.5,
        num_inference_steps=50,
    ).images[0]
    # --- 다운샘플
    output_img_down = downscale(output_img, SUBMIT_SIZE)
    return output_img_down

# ========== 11. 최종 추론 ==========
out_imgs = []
out_img_names = []
for idx, (img_id, img_path, caption, cleaned_prompt) in enumerate(
    zip(df['ID'], df['input_img_path'], df['caption'], df['cleaned_prompt'])
):
    input_img = Image.open(img_path).convert("RGB")
    print(f"\n처리 중: {img_id} ({idx+1}/{len(df)})")
    print(f"캡션: {caption[:80]}{'...' if len(caption) > 80 else ''}")
    try:
        input_img = Image.open(img_path).convert("RGB")
        # 프롬프트 강화 적용: 의문형이면 cleaned_prompt → 프롬프트 강화, 일반 caption은 caption → 프롬프트 강화
        use_prompt = cleaned_prompt
        output_img = (
            generate_ensemble(input_img, use_prompt, img_id)
            if CFG['USE_ENSEMBLE'] else
            generate_single_image(input_img, use_prompt)
        )
        out_imgs.append(output_img)
        out_img_names.append(img_id)
        print(f"{img_id} 완료!")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{img_id} 실패: {e}")
        continue

# ========== 12. 제출 파일 생성 ==========
print("\n제출 파일 생성 중...")
os.makedirs(CFG['SUB_DIR'], exist_ok=True)
feat_imgs = []
for output_img, img_id in tqdm(zip(out_imgs, out_img_names), desc="임베딩 추출"):
    path_out_img = CFG['SUB_DIR'] + '/' + img_id + '.png'
    output_img.save(path_out_img)
    output_img_tensor = clip_preprocess(output_img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat_img = clip_model.encode_image(output_img_tensor)
        feat_img /= feat_img.norm(dim=-1, keepdim=True)
    feat_img = feat_img.detach().cpu().numpy().reshape(-1)
    feat_imgs.append(feat_img)
feat_imgs = np.array(feat_imgs)
vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
feat_submission.insert(0, 'ID', out_img_names)
feat_submission.to_csv(CFG['SUB_DIR']+'/embed_submission.csv', index=False)
zip_path = './submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            zipf.write(file_path, arcname=file_name)
print(f"2단계 제출 파일 생성 완료: {zip_path}")