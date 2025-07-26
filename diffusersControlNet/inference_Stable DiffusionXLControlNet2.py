"""
[Stable Diffusion XL + ControlNet + CLIP 임베딩] inference
- SDXL(Stable Diffusion XL) + ControlNet(Canny) 기반 텍스트-이미지 생성
- Stable Diffusion XL (SDXL) [stabilityai/stable-diffusion-xl-base-1.0, diffusers]
- ControlNet (Canny, SDXL 전용) [controlnet-canny-sdxl-1.0]
    · 엣지맵(윤곽) 기반 구조 보존, 로컬/허브 모두 지원
- CLIP 임베딩/평가 (ViT-L-14, openai)
    · open_clip 라이브러리, 제출용 임베딩 벡터 추출 (CLIP 점수 기반 best-of-N 선택 옵션도 있음)
- PromptEnhancer
    · 품질/질감/조명/장면/색상 등 프롬프트 자동 강화, 네거티브 프롬프트(negative prompt)도 동적 생성
- DynamicParameterGenerator
    · 캡션 키워드/길이 기반 guidance scale, step 수, canny 임계값 등 자동 샘플링
    · "scene", "object", "person" 등 상황별 파라미터 튜닝
-------------------------------------------------------------
- USE_ENSEMBLE: True/False로 단일 vs 앙상블(best-of-N, CLIP 점수 선택) 지원
- USE_DYNAMIC_GUIDANCE, USE_CANNY_OPTIMIZATION: 키워드 기반 동적 파라미터 자동 조정
- 프롬프트 클린(cleaned_prompts) 및 강화, 토큰 제한, 의문문 패턴 필터링 등 적용
- 로컬/허브 모델 모두 지원, torch.float16 최적화
- 단일/앙상블 모두 512x512 제출 (내부적으로는 1024x1024 생성, 다운샘플)
-------------------------------------------------------------
"""


import open_clip
from tqdm import tqdm
import re
import random
import os
import zipfile
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import warnings
import nltk

# 필요한 NLTK 데이터 다운로드
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

warnings.filterwarnings('ignore')

os.chdir('/home/guest01/colorize/')
ROOT_PATH = '/home/guest01/colorize/'
TEST_CSV_PATH = os.path.join(ROOT_PATH, 'test.csv')
IMG_ROOT = os.path.join(ROOT_PATH, 'test/input_image')
SUB_DIR = os.path.join(ROOT_PATH, 'submission')
ZIP_PATH = os.path.join(ROOT_PATH, 'submission.zip')
os.makedirs(SUB_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# --- 설정 ---
CFG = {
    'SUB_DIR': './submission',
    'SEED': 42,
    'USE_ENHANCED_PROMPTS': True,
    'USE_NEGATIVE_PROMPTS': True,

    'USE_DYNAMIC_GUIDANCE': True,       # 동적 guidance_scale
    'USE_ENSEMBLE': False,              # 앙상블 off
    'NUM_ENSEMBLE_SAMPLES': 2,          # 앙상블 샘플 수
    'USE_ADAPTIVE_STEPS': True,         # 적응적 스텝 수
    'USE_CLIP_SCORING': False,          # CLIP 점수로 최고 선택
    'USE_CANNY_OPTIMIZATION': True,     # Canny 파라미터 최적화
}

# --- 랜덤 시드 ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

# --- 사전학습 모델 로딩 - SDXL + ControlNet SDXL + CLIP-G/14 ---
def load_models():
    # ControlNet 모델 로컬 경로 정의
    controlnet_local_path = "/home/guest01/model/controlnet-canny-sdxl-1.0"
    # Hugging Face Hub에서 사용할 ControlNet 리포지토리 ID
    controlnet_hub_id = "diffusers/controlnet-canny-sdxl-1.0"

    # ControlNet 모델 로딩
    if os.path.exists(controlnet_local_path):
        print(f"ControlNet 모델 로드 중: {controlnet_local_path}")
        controlnet = ControlNetModel.from_pretrained(controlnet_local_path, torch_dtype=torch.float16).to(device)
    else:
        print(f"로컬 경로에 ControlNet 모델 없음 Hugging Face Hub에서 다운로드 중: {controlnet_hub_id}")
        print(f"다운로드된 모델은 {controlnet_local_path}에 저장됩니다.")
        controlnet = ControlNetModel.from_pretrained(controlnet_hub_id, torch_dtype=torch.float16).to(device)

    # Stable Diffusion XL Base 모델 로컬 경로에서 불러오기
    # Assuming stabilityai/stable-diffusion-xl-base-1.0 is in /home/guest01/model/stabilityaistable-diffusion-xl-base-1.0
    sdxl_base_path = "/home/guest01/model/stabilityaistable-diffusion-xl-base-1.0"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(sdxl_base_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
        ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # CLIP 모델은 일반적으로 자동으로 다운로드되므로, 로컬 경로 지정이 필요 없을 수 있습니다.
    # 만약 CLIP 모델도 로컬에 있다면 해당 경로를 지정해 주세요.
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device)
    clip_model.eval()
    return pipe, clip_model, clip_preprocess

pipe, clip_model, clip_preprocess = load_models()

# --- 함수 및 클래스 정의 ---
# 2. 키워드 추출 ---
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

# 3. 클린 프롬프트 생성
def make_clean_prompt(text, category_dict, style_tail="vibrant colors, photorealistic, high quality"):
    kw = extract_keywords(text, category_dict)
    if kw:
        prompt = " ".join(kw).capitalize() + ", " + style_tail
    else:
        prompt = style_tail
    return prompt

# 4. category_dict 정의
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

# 5. PromptEnhancer 클래스
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
            "photorealistic, masterpiece",
            "ultra detailed, high resolution",
            "award winning, cinematic lighting",
            "professional photography, vivid colors",
            "DSLR, finely detailed, rich colors",
            "trending on artstation, beautiful lighting",
            "fine art, natural light, vibrant colors",
            "incredibly realistic, beautiful composition",
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

    # 프롬프트 스타일/색상 강화, 토큰 수 제한
    def enhance_caption(self, caption, max_per_cat=1):
        # 색상 키워드 강화
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

        # 스타일/조명/텍스처/배경 랜덤으로 2개
        style_part = random.choice(self.quality_keywords)
        others = random.sample(
            self.texture_keywords + self.lighting_keywords + self.scene_keywords, k=2
        )
        prompt_parts = [style_part] + others + keywords + ["vibrant colors"]

        # 토큰 개수 제한 (35토큰 이하, 단어 기준)
        prompt_txt = ", ".join(prompt_parts)
        tokens = re.findall(r'\b[a-zA-Z]+\b', prompt_txt)
        if len(tokens) > 35:
            # 초과 시 뒤에서부터 삭제
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

# 6. 테스트셋 분석 기반 핵심 키워드 (동작, 위치, 장면, 인물, 오브젝트, 색상) 리스트
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

# 7. DynamicParameterGenerator 클래스
class DynamicParameterGenerator:
    def __init__(self):
        self.guidance_ranges = {
            'action': (7.8, 8.5),       # 동작/위치 강조
            'scene': (7.3, 8.1),        # 장면/환경
            'person_object': (6.9, 7.6), # 인물, 오브젝트, 의상
            'color': (6.7, 7.3),        # 색상 묘사만 있을 때
            'default': (7.0, 7.5)
        }
        self.step_ranges = {
            'simple': (32, 38),
            'complex': (50, 58),
            'default': (42, 48)
        }

    def get_optimal_guidance(self, caption):
        cap = caption.lower()
        # 1. 행동/동작/위치
        if any(w in cap for w in ACTION_WORDS + POSITION_WORDS):
            return random.uniform(*self.guidance_ranges['action'])
        # 2. scene/장면/환경
        if any(w in cap for w in SCENE_WORDS):
            return random.uniform(*self.guidance_ranges['scene'])
        # 3. 인물/오브젝트/의상
        if any(w in cap for w in PERSON_WORDS + OBJECT_WORDS + CLOTHES_WORDS):
            return random.uniform(*self.guidance_ranges['person_object'])
        # 4. 색상만 강조
        if any(w in cap for w in COLOR_WORDS):
            return random.uniform(*self.guidance_ranges['color'])
        # 기본
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
        # scene, field, mountain, sky 등 배경은 threshold 높임
        if any(w in caption.lower() for w in ['scene', 'field', 'mountain', 'park', 'background', 'sky']):
            return random.randint(120, 140), random.randint(220, 240)
        else:
            return random.randint(100, 120), random.randint(200, 220)

# 8. CLIP 점수 계산기
class CLIPScorer:
    def __init__(self, clip_model, clip_preprocess):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

    def calculate_clip_score(self, image, caption):
        """이미지와 캡션 간의 CLIP 점수 계산"""
        try:
            # 이미지 전처리
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(device)

            # 텍스트 토큰화
            text_tokens = open_clip.tokenize([caption]).to(device)

            with torch.no_grad():
                # 특징 추출
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)

                # 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 코사인 유사도
                similarity = torch.cosine_similarity(image_features, text_features).item()

            return similarity
        except Exception as e:
            print(f"CLIP 점수 계산 실패: {e}")
            return 0.0

# --- 클린 프롬프트 생성 ---
df = pd.read_csv(TEST_CSV_PATH)
patterns = [
    r'do you see', r'what is', r'is there', r'can you', r'please', r'maybe', r'possibly', r'probably',
    r'find', r'tell me', r'show me', r'could be', r'might be', r'should be', r'appears? to be', r'looks like',
    r'try to', r'look at', r'imagine', r'guess what', r'one can see', r'describe', r'identify', 'how many',
    r'which', r'where', r'it might', r'they might'
]
# Combine all patterns into a single regex for efficiency
combined_pattern = r'|'.join(patterns)
bad_mask = df['caption'].str.contains(combined_pattern, case=False, regex=True)

cleaned_prompts = [
    make_clean_prompt(caption, category_dict) if bad
    else caption + ", vibrant colors, photorealistic, high quality"
    for caption, bad in zip(df['caption'], bad_mask)
]

# --- Instantiate classes ---
prompt_enhancer = PromptEnhancer(category_dict)
param_generator = DynamicParameterGenerator()
clip_scorer = CLIPScorer(clip_model, clip_preprocess)

# --- 단일 이미지 생성 함수/ControlNet + SDXL 단일 추론 ---
def generate_single_image(input_img, prompt_text):
    control_image = preprocess_for_controlnet_advanced(input_img)
    enhanced_prompt, negative_prompt = prompt_enhancer.enhance_caption(prompt_text)

    output_img = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        guidance_scale=7.5,
        num_inference_steps=50,
    ).images[0]

    output_img = output_img.resize((512, 512))

    return output_img

# --- 앙상블 생성 함수/다양한 파라미터로 앙상블 후 최종 선택---
# 1. 앙상블 전처리/SDXL ControlNet용 Canny 전처리 (1024x1024)
def preprocess_for_controlnet_advanced(image: Image.Image, low_threshold=100, high_threshold=200):
    canny_detector = CannyDetector()
    image = image.resize((1024, 1024))
    image_np = np.array(image)
    control_image_np = canny_detector(image_np, low_threshold=low_threshold, high_threshold=high_threshold)
    return Image.fromarray(control_image_np)

# 2. 앙상블 생성 함수
def generate_ensemble(input_img, prompt_text, img_id):
    print(f" 앙상블 생성 ({CFG['NUM_ENSEMBLE_SAMPLES']}개 샘플)")

    candidates = []
    scores = []

    for i in range(CFG['NUM_ENSEMBLE_SAMPLES']):
        try:
            # 동적 파라미터 생성
            guidance_scale = param_generator.get_optimal_guidance(prompt_text)
            num_steps = param_generator.get_optimal_steps(prompt_text)
            low_thresh, high_thresh = param_generator.get_optimal_canny_params(prompt_text)

            print(f"     샘플 {i+1}: guidance={guidance_scale:.1f}, steps={num_steps}, canny=({low_thresh},{high_thresh})")

            # 제어 이미지 생성 (최적화된 파라미터)
            control_image = preprocess_for_controlnet_advanced(
                input_img, low_threshold=low_thresh, high_threshold=high_thresh
            )

            enhanced_prompt, negative_prompt = prompt_enhancer.enhance_caption(prompt_text)

            # 이미지 생성
            generation_kwargs = {
                'prompt': enhanced_prompt,
                'negative_prompt': negative_prompt,
                'image': control_image,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_steps,
                'generator': torch.Generator(device=device).manual_seed(CFG['SEED'] + i)
            }

            output_img = pipe(**generation_kwargs).images[0]
            candidates.append(output_img)

            # CLIP 점수 계산
            if CFG['USE_CLIP_SCORING']:
                original_caption_for_clip = df.loc[df['ID'] == img_id, 'caption'].iloc[0]
                clip_score = clip_scorer.calculate_clip_score(output_img, original_caption_for_clip)
                scores.append(clip_score)
                print(f"       CLIP 점수: {clip_score:.3f}")
            else:
                scores.append(random.random())  # 랜덤 점수

        except Exception as e:
            print(f" 샘플 {i+1} 생성 실패: {e}")
            continue

    # 최고 점수 선택
    if candidates:
        best_idx = np.argmax(scores)
        best_image = candidates[best_idx]
        best_score = scores[best_idx]
        print(f"   최고 점수: {best_score:.3f} (샘플 {best_idx+1})")
        return best_image
    else:
        # 모든 생성 실패시 기본 생성
        print(f" 앙상블 실패, 기본 생성으로 대체")
        return generate_single_image(input_img, prompt_text)

# --- 추론 루프 ---
test_df = pd.read_csv(TEST_CSV_PATH)
out_imgs = []
out_img_names = []

for idx, (img_id, img_path, prompt_text) in enumerate(zip(test_df['ID'], test_df['input_img_path'], cleaned_prompts)):
    print(f"\n처리 중: {img_id} ({idx+1}/{len(test_df)})")
    print(f"프롬프트: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")

    try:
        # 이미지 로딩: IMG_ROOT와 img_path를 결합하여 올바른 경로 생성
        input_img = Image.open(os.path.join(IMG_ROOT, img_path)).convert("RGB")

        # 단일 vs 앙상블 생성 선택
        if CFG['USE_ENSEMBLE']:
            output_img = generate_ensemble(input_img, prompt_text, img_id)
        else:
            output_img = generate_single_image(input_img, prompt_text)

        if output_img is not None:
            out_imgs.append(output_img)
            out_img_names.append(img_id)
            print(f"{img_id} 완료!")
        else:
            print(f"{img_id} 이미지 생성 실패 (None)")

    except Exception as e:
        print(f"{img_id} 실패: {e}")
    finally: # GPU 메모리 정리 (항상 실행되도록 finally 블록에 위치)
        torch.cuda.empty_cache()

os.makedirs(CFG['SUB_DIR'], exist_ok=True)

# --- 임베딩 추출 ---
feat_imgs = []
for output_img, img_id in tqdm(zip(out_imgs, out_img_names), desc="임베딩 추출"):
    img_filename = f"{img_id}.png"
    path_out_img = os.path.join(CFG['SUB_DIR'], img_filename)
    output_img.save(path_out_img)

    output_img_tensor = clip_preprocess(output_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_img = clip_model.encode_image(output_img_tensor)
        feat_img /= feat_img.norm(dim=-1, keepdim=True)
    feat_img = feat_img.detach().cpu().numpy().reshape(-1)
    feat_imgs.append(feat_img)

# CSV 저장
feat_imgs = np.array(feat_imgs)
vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
feat_submission.insert(0, 'ID', out_img_names)
feat_submission.to_csv(os.path.join(CFG['SUB_DIR'], 'embed_submission.csv'), index=False)

# --- 제출 ZIP 파일 생성---
zip_path = './submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            zipf.write(file_path, arcname=file_name)

print(f"제출 파일 생성 완료: {zip_path}")