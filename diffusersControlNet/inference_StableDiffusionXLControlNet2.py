"""
[Stable Diffusion XL + ControlNet(Canny) + CLIP-G/14 ] inference
- Stable Diffusion XL (SDXL) stabilityai/stable-diffusion-xl-base-1.0, diffusers]
- ControlNet (Canny, SDXL 전용) [diffusers/controlnet-canny-sdxl-1.0]
    · 구조/윤곽 정보 보존, SDXL 전용 사전학습 모델
- CLIP 임베딩/평가 (ViT-g-14, laion2b_s34b_b88k)
    · open_clip 사용, best-of-N 앙상블 샘플 선택 및 제출 임베딩 추출
- 프롬프트 강화 (PromptEnhancer)
    · 품질/질감/조명/배경/색감 등 자동 삽입
    · 네거티브 프롬프트(negative prompt)도 상황에 따라 자동 생성
- 동적 파라미터 제어 (DynamicParameterGenerator)
    · 캡션 키워드/길이 기반으로 guidance scale, step 수, canny 임계값 등 자동 샘플링
"""
from tqdm import tqdm
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
import open_clip
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/guest01/colorize/')
ROOT_PATH = '/home/guest01/colorize/'
TEST_CSV_PATH = os.path.join(ROOT_PATH, 'test.csv')
IMG_ROOT = './test/input_image'
SUB_DIR = os.path.join(ROOT_PATH, 'submission')
ZIP_PATH = os.path.join(ROOT_PATH, 'submission.zip')
os.makedirs(SUB_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2단계 설정 
CFG = {
    'SUB_DIR': './submission',
    'SEED': 42,
    'USE_ENHANCED_PROMPTS': True,
    'USE_NEGATIVE_PROMPTS': True,

    'USE_DYNAMIC_GUIDANCE': True,      # 동적 guidance_scale
    'USE_ENSEMBLE': True,              # 앙상블 생성
    'NUM_ENSEMBLE_SAMPLES': 2,         # 앙상블 샘플 수
    'USE_ADAPTIVE_STEPS': True,        # 적응적 스텝 수
    'USE_CLIP_SCORING': True,          # CLIP 점수로 최고 선택
    'USE_CANNY_OPTIMIZATION': True,    # Canny 파라미터 최적화
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

# 프롬프트 강화
class PromptEnhancer:
    def __init__(self):
        # 색상 강화 세트
        self.color_enhancements = {
            'red':    ['vibrant red', 'crimson red', 'cherry red', 'ruby red'],
            'blue':   ['deep blue', 'azure blue', 'sapphire blue', 'navy blue'],
            'green':  ['lush green', 'emerald green', 'forest green', 'jade green'],
            'yellow': ['bright yellow', 'golden yellow', 'sunny yellow', 'amber yellow'],
            'orange': ['warm orange', 'burnt orange', 'tangerine orange', 'coral orange'],
            'purple': ['rich purple', 'violet purple', 'lavender purple', 'amethyst purple'],
            'pink':   ['soft pink', 'rose pink', 'magenta pink', 'blush pink'],
            'brown':  ['warm brown', 'chocolate brown', 'coffee brown', 'chestnut brown'],
            'black':  ['deep black', 'charcoal black', 'ebony black', 'jet black'],
            'white':  ['pure white', 'ivory white', 'pearl white', 'snow white'],
            'gray':   ['light gray', 'silver gray', 'slate gray'],
            'grey':   ['light grey', 'silver grey', 'slate grey']
        }

        # 품질 키워드
        self.quality_keywords = [
            "photorealistic, high resolution, detailed textures, professional photography",
            "masterpiece quality, ultra detailed, sharp focus, vivid colors",
            "award winning photography, stunning colors, perfect lighting, cinematic",
            "high quality rendering, natural lighting, rich colors, fine details",
            "professional color grading, vibrant palette, exceptional detail"
        ]

        # 질감(재질) 키워드
        self.texture_keywords = [
            "intricate textures", "soft fabric texture", "subtle film grain",
            "realistic wood grain", "fine metallic shine"
        ]

        # 조명 키워드
        self.lighting_keywords = [
            "cinematic lighting", "soft golden backlight", "volumetric light",
            "diffused natural light", "dynamic shadows", "ambient occlusion"
            "natural lighting", "natural shadows"
        ]

        # 장면(배경) 키워드
        self.scene_keywords = [
            "detailed background", "clear midday sky", "subtle reflection",
            "dynamic composition", "intricate background elements", "4k", "peaceful mood"
            "energetic vibe", "beautiful background"
        ]

        # 네거티브 프롬프트
        self.negative_prompts = [
            "blurry, low quality, pixelated, artifacts, oversaturated",
            "monochrome, black and white, sepia, desaturated, dull",
            "cartoonish, anime style, artificial colors, flat lighting",
            "watermark, text, signature, logo, border, frame"
        ]

    def enhance_caption(self, caption):
        enhanced_caption = caption.lower()
        # 색상 키워드 강화
        for color, enhancements in self.color_enhancements.items():
            if color in enhanced_caption:
                enhanced_color = random.choice(enhancements)
                enhanced_caption = enhanced_caption.replace(color, enhanced_color)

        # 강화 프롬프트 조립
        positive_prompt = (
            f"{random.choice(self.quality_keywords)}, "
            f"{random.choice(self.texture_keywords)}, "
            f"{random.choice(self.lighting_keywords)}, "
            f"{random.choice(self.scene_keywords)}, "
            f"{enhanced_caption}, colorful, vibrant colors, do not change structure, only colorize"
        )
        negative_prompt = random.choice(self.negative_prompts)

        return positive_prompt, negative_prompt
# 2단계: CLIP 점수 계산기 
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

# 2단계: 동적 파라미터 생성기 
class DynamicParameterGenerator:
    def __init__(self):
        self.guidance_ranges = {
            'portrait': (6.0, 8.0),
            'landscape': (7.0, 9.0),
            'object': (6.5, 8.5),
            'default': (6.0, 9.0)
        }

        self.step_ranges = {
            'simple': (30, 40),
            'complex': (45, 55),
            'default': (40, 50)
        }

    def get_optimal_guidance(self, caption):
        """캡션 내용에 따라 최적 guidance_scale 선택"""
        caption_lower = caption.lower()

        if any(word in caption_lower for word in ['person', 'man', 'woman', 'face', 'portrait']):
            return random.uniform(*self.guidance_ranges['portrait'])
        elif any(word in caption_lower for word in ['landscape', 'sky', 'mountain', 'field']):
            return random.uniform(*self.guidance_ranges['landscape'])
        elif any(word in caption_lower for word in ['object', 'item', 'thing', 'apple', 'car']):
            return random.uniform(*self.guidance_ranges['object'])
        else:
            return random.uniform(*self.guidance_ranges['default'])

    def get_optimal_steps(self, caption):
        """캡션 복잡도에 따라 최적 스텝 수 선택"""
        word_count = len(caption.split())

        if word_count <= 5:
            return random.randint(*self.step_ranges['simple'])
        elif word_count >= 15:
            return random.randint(*self.step_ranges['complex'])
        else:
            return random.randint(*self.step_ranges['default'])

    def get_optimal_canny_params(self):
        """최적화된 Canny 파라미터"""
        low_threshold = random.randint(80, 120)
        high_threshold = random.randint(180, 220)
        return low_threshold, high_threshold

# 사전학습 모델 로딩 - SDXL + ControlNet SDXL + CLIP-G/14
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# ControlNet SDXL용 (canny, depth 등)
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
).to(device)
# SDXL 베이스 모델
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)
# 스케줄러
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# CLIP 모델 미리 로딩 (점수 계산용)
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-g-14", pretrained="laion2b_s34b_b88k"
)
clip_model.to(device)

# 컴포넌트 초기화
prompt_enhancer = PromptEnhancer()
clip_scorer = CLIPScorer(clip_model, clip_preprocess)
param_generator = DynamicParameterGenerator()

# 2단계: 앙상블 전처리 함수
def preprocess_for_controlnet_advanced(image: Image.Image, low_threshold=100, high_threshold=200):
    """SDXL ControlNet용 Canny 전처리 (1024x1024)"""
    canny_detector = CannyDetector()
    image = image.resize((1024, 1024))  # SDXL input size
    image_np = np.array(image)
    control_image_np = canny_detector(image_np, low_threshold=low_threshold, high_threshold=high_threshold)
    return Image.fromarray(control_image_np)

# 2단계: 앙상블 생성 함수 
def generate_ensemble(input_img, caption, img_id):
    """여러 설정으로 생성하여 최고 품질 선택"""

    print(f" 앙상블 생성 ({CFG['NUM_ENSEMBLE_SAMPLES']}개 샘플)")

    candidates = []
    scores = []

    for i in range(CFG['NUM_ENSEMBLE_SAMPLES']):
        try:
            # 동적 파라미터 생성
            guidance_scale = param_generator.get_optimal_guidance(caption)
            num_steps = param_generator.get_optimal_steps(caption)
            low_thresh, high_thresh = param_generator.get_optimal_canny_params()

            print(f"    샘플 {i+1}: guidance={guidance_scale:.1f}, steps={num_steps}, canny=({low_thresh},{high_thresh})")

            # 제어 이미지 생성 (최적화된 파라미터)
            control_image = preprocess_for_controlnet_advanced(
                input_img, low_threshold=low_thresh, high_threshold=high_thresh
            )

            # 프롬프트 생성
            enhanced_prompt, negative_prompt = prompt_enhancer.enhance_caption(caption)

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
                clip_score = clip_scorer.calculate_clip_score(output_img, caption)
                scores.append(clip_score)
                print(f"      CLIP 점수: {clip_score:.3f}")
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
        return generate_single_image(input_img, caption)

def generate_single_image(input_img, caption):
    """기본 단일 이미지 생성"""
    control_image = preprocess_for_controlnet_advanced(input_img)
    enhanced_prompt, negative_prompt = prompt_enhancer.enhance_caption(caption)

    output_img = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        guidance_scale=7.5,
        num_inference_steps=50,
    ).images[0]

    return output_img
# 2단계: 앙상블  
print("앙상블 추론 시작!")

test_df = pd.read_csv(TEST_CSV_PATH)
out_imgs = []
out_img_names = []

for idx, (img_id, img_path, caption) in enumerate(zip(test_df['ID'], test_df['input_img_path'], test_df['caption'])):
    input_img = Image.open(img_path).convert("RGB")
    print(f"\n처리 중: {img_id} ({idx+1}/{len(test_df)})")
    print(f"캡션: {caption[:80]}{'...' if len(caption) > 80 else ''}")

    try:
        # 이미지 로딩
        input_img = Image.open(img_path).convert("RGB")

        # 앙상블 vs 단일 생성 선택
        if CFG['USE_ENSEMBLE']:
            output_img = generate_ensemble(input_img, caption, img_id)
        else:
            output_img = generate_single_image(input_img, caption)

        out_imgs.append(output_img)
        out_img_names.append(img_id)
        print(f"✅ {img_id} 완료!")

        # GPU 메모리 정리
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"{img_id} 실패: {e}")
        continue

print('2단계 고급 추론 완료!')

# 제출 파일 생성
print("\제출 파일 생성 중")

os.makedirs(CFG['SUB_DIR'], exist_ok=True)

# 임베딩 추출
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

# ZIP 파일 생성
zip_path = './submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            zipf.write(file_path, arcname=file_name)

print(f"2단계 제출 파일 생성 완료: {zip_path}")