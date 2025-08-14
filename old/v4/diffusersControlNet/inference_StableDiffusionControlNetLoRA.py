"""
[Stable Diffusion + ControlNet + LoRA] inference
일반 RGB 이미지, 단일 추론, 효율 및 클린 코드, n-gram clean 등 강화
테스트셋 이미지에 대해 Stable Diffusion + ControlNet + LoRA + 커스텀 프롬프트로 컬러 이미지를 생성
1. Stable Diffusion v1-5** (runwayml/stable-diffusion-v1-5, diffusers 기반)
- 텍스트 → 이미지 생성 메인 프레임워크
2. ControlNet (Canny)** (lllyasviel/sd-controlnet-canny)
- 엣지맵 기반의 추가 조건(흑백 구조 유지) 제공, 사전학습+대회 자체 파인튜닝(best_controlnet.pth)
3. LoRA Adapter
- UNet에 LoRA 어댑터(r=8, lora_alpha=32 등) 삽입, 별도 학습 가중치(best_unet.pth) 로드
- 주요 모듈: ["to_q","to_v","to_k","to_out.0"]
4. CLIP 임베딩/평가 (openai/clip-vit-base-patch32, 임베딩은 ViT-L-14)
- 제출 임베딩/평가용, open_clip 라이브러리
---------
- 프롬프트 강화: PromptEnhancer 클래스에서 품질/질감/조명/장면/색감 등 자동 삽입 + 부정 프롬프트(negative prompt)도 자동 생성
- 동적 파라미터: 캡션 키워드 및 길이 기반 guidance scale, step 수, Canny 임계값을 자동 조정
- Canny 엣지맵: 입력 이미지를 Canny 엣지로 변환 후 ControlNet 조건 입력
- 모델 파라미터: ControlNet/UNet 모두 대회용 best checkpoint 수동 로드, float16 최적화(GPU시)
- 파이프라인 완전 freeze: 추론 시 모든 모델 파라미터 고정
- 입력: test.csv (ID, input_img_path, caption), 각 이미지 파일
- 출력: ./test_submission/ (이미지, embed_submission.csv), ./submission.zip (최종 제출용 압축)
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import zipfile
from tqdm import tqdm
from torchvision import transforms
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig
import cv2
import re
import string
import random
from collections import Counter
import nltk

# NLTK punkt 토크나이저 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Config/Paths ---
class TestConfig:
    def __init__(self):
        self.IMG_SIZE = 512
        self.TEST_CSV = "/home/guest01/colorize/test.csv"  # 테스트셋 경로
        self.SUB_DIR = "./test_submission"
        self.CONTROLNET_CKPT = "/home/guest01/colorize/diffusersControlNet/output/best_controlnet.pth"
        self.UNET_CKPT = "/home/guest01/colorize/diffusersControlNet/output/best_unet.pth"
        self.PIPELINE_MODEL = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"  
        self.EMBED_MODEL = "ViT-L-14"  
        self.EMBED_PRETRAINED = "openai"
        self.SUBMISSION_ZIP = "./submission.zip"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = TestConfig()

os.makedirs(CFG.SUB_DIR, exist_ok=True)

# --- 유틸/정제 (이전과 동일) ---
color_words = set([
    'white', 'black', 'gray', 'grey', 'red', 'blue', 'green', 'yellow', 'orange', 'pink',
    'purple', 'brown', 'tan', 'silver', 'gold', 'beige', 'violet', 'cyan', 'magenta',
    "navy", "olive", "burgundy", "maroon", "teal", "lime", "indigo", "charcoal",
    "peach", "cream", "ivory", "turquoise", "mint", "mustard", "coral"
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
        ngrams = list(nltk.ngrams(tokens, n))
        ngram_counter.update(ngrams)
    return [' '.join(k) for k, v in ngram_counter.most_common(topk)]

def build_remove_phrases(captions, ngram_ns=(2,3,4), topk=100):
    remove_phrases = set()
    for n in ngram_ns:
        remove_phrases |= set(get_top_ngrams(captions, n, topk))
    return list(remove_phrases)

def clean_caption_full(caption, remove_phrases, number_words, number_regex, max_tokens=30):
    c = str(caption).lower()
    for phrase in remove_phrases:
        c = re.sub(r'[\s,.!?;:]*' + re.escape(phrase) + r'[\s,.!?;:]*', ' ', c)
    c = c.translate(str.maketrans('', '', string.punctuation))
    c = number_regex.sub(' ', c)
    c = ' '.join([w for w in c.split() if w not in number_words])
    c = re.sub(r'\s+', ' ', c).strip()
    # 중복 단어 제거
    seen = set()
    result = []
    for word in c.split():
        if word not in seen:
            result.append(word)
            seen.add(word)
    # 토큰 제한
    return ' '.join(result[:max_tokens])

# --- PromptEnhancer (이전과 동일) ---
class PromptEnhancer:
    def __init__(self):
        self.quality_keywords = ["masterpiece", "best quality", "high resolution", "4k", "8k"]
        self.texture_keywords = ["detailed texture", "smooth texture", "realistic texture"]
        self.lighting_keywords = ["dramatic lighting", "soft lighting", "cinematic lighting", "studio lighting"]
        self.scene_keywords = ["wide angle", "close up", "full body shot", "dynamic pose", "indoor scene", "outdoor scene"]
        self.color_enhancements = {
            'red': ['vivid red', 'crimson', 'scarlet'],
            'blue': ['sky blue', 'deep blue', 'azure'],
            'green': ['emerald green', 'forest green', 'lime green']
        }
        self.negative_prompts_general = [
            "bad anatomy", "disfigured", "extra limbs", "mutated hands", "blurry", "low resolution", "ugly", "distorted"
        ]
        self.negative_prompts_person = [
            "deformed face", "unnatural skin", "extra fingers", "too many limbs", "pale skin"
        ]
        self.negative_prompts_landscape = [
            "washed out colors", "low contrast", "color shift", "color bleeding", "over-saturated colors"
        ]
        self.negative_prompts_object = [
            "wrong perspective", "misshapen object"
        ]
    def enhance_caption(self, caption):
        enhanced_caption = (
            f"{random.choice(self.quality_keywords)}, "
            f"{random.choice(self.texture_keywords)}, "
            f"{random.choice(self.lighting_keywords)}, "
            f"{random.choice(self.scene_keywords)}, "
            f"photorealistic, high resolution, best quality, {caption}"
        )
        for color_word, enhancements in self.color_enhancements.items():
            if color_word in enhanced_caption:
                enhanced_caption += f", {random.choice(enhancements)}"
        enhanced_caption += ", colorful, vibrant colors, do not change structure, only colorize"
        negs = random.sample(self.negative_prompts_general, k=random.randint(2, 4))
        if any(word in caption for word in ['person', 'man', 'woman', 'face', 'people']):
            negs += random.sample(self.negative_prompts_person, k=2)
        if any(word in caption for word in ['tree', 'sky', 'field', 'mountain', 'river', 'cloud']):
            negs += random.sample(self.negative_prompts_landscape, k=2)
        if any(word in caption for word in ['car', 'train', 'bus', 'object', 'table', 'chair']):
            negs += random.sample(self.negative_prompts_object, k=1)
        neg_prompt = ', '.join(set(negs)) + ", bad quality, grayscale, monochromatic, desaturated, unrealistic colors"
        return enhanced_caption, neg_prompt

# --- DynamicParameterGenerator (이전과 동일) ---
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
            self.TYPE_COMPLEX_DETAIL: ((20, 80), (50, 120)),
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

prompt_enhancer = PromptEnhancer()
dynamic_param_gen = DynamicParameterGenerator()

# --- Helper: Canny Edge for ControlNet (OpenCV) ---
def preprocess_for_controlnet(img_pil, detector_type="canny", low=100, high=200):
    tfm = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    img = tfm(img_pil)
    img = np.array(img.convert("RGB"))
    if detector_type == "canny":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, low, high)
        edges = Image.fromarray(edges).convert("RGB")
        return edges
    else:
        return img_pil

# --- Load Pipeline & Model ---
# ControlNet 초기화 및 가중치 로드
controlnet = ControlNetModel.from_pretrained(CFG.CONTROLNET_MODEL).to(CFG.DEVICE)
controlnet.load_state_dict(torch.load(CFG.CONTROLNET_CKPT, map_location=CFG.DEVICE))
if CFG.DEVICE == "cuda":
    controlnet.half() # <<< Keep this: Ensures ControlNet's parameters are float16

# StableDiffusionControlNetPipeline 초기화
# The torch_dtype argument here is crucial for the overall pipeline components
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    CFG.PIPELINE_MODEL,
    controlnet=controlnet, # Pass the already casted controlnet instance
    torch_dtype=torch.float16 if CFG.DEVICE == "cuda" else torch.float32,
).to(CFG.DEVICE) # .to(CFG.DEVICE) also handles device placement and initial dtype casting

# LoRA config 적용 및 UNet 가중치 로드
lora_cfg = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05,
    init_lora_weights="gaussian",
    target_modules=["to_q","to_v","to_k","to_out.0"]
)
pipe.unet.add_adapter(lora_cfg)
pipe.unet.load_state_dict(torch.load(CFG.UNET_CKPT, map_location=CFG.DEVICE))
if CFG.DEVICE == "cuda":
    pipe.unet.half() 

# 스케줄러 설정
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# 필요시 freeze
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)

# --- Test Inference ---
try:
    test_df = pd.read_csv(CFG.TEST_CSV) 
    # test_df = test_df.head(5) # 실제 추론 시에는 이 라인을 주석 처리하거나 제거해야 합니다.
    if test_df.empty:
        raise ValueError("test.csv 파일이 비어있습니다. 경로와 내용을 확인해주세요.")
    print(f"test.csv에서 {len(test_df)}개의 항목을 찾았습니다.")
    # 첫 번째 항목의 예시를 출력하여 데이터 로딩 확인
    if not test_df.empty:
        print("첫 번째 항목 예시:")
        print(test_df.iloc[0])

except FileNotFoundError:
    print(f"오류: test.csv 파일을 찾을 수 없습니다: {CFG.TEST_CSV}")
    exit() 
except Exception as e:
    print(f"test.csv 로딩 중 오류 발생: {e}")
    exit() 

all_test_captions = test_df['caption'].astype(str).tolist()
remove_phrases = build_remove_phrases(all_test_captions, ngram_ns=(2,3,4), topk=100)

out_imgs = []
out_img_names = []

print("Generating images...")
script_dir = os.path.dirname(os.path.abspath(__file__)) 
csv_dir = os.path.dirname(CFG.TEST_CSV)

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_id = row['ID']
    
    if 'caption' not in row or pd.isna(row['caption']):
        print(f"경고: ID {img_id} 에 대한 캡션이 없거나 유효하지 않습니다. 이 항목을 건너뜁니다.")
        continue

    caption = row['caption']
    relative_img_path_from_csv = row['input_img_path']

    input_img_path = os.path.join(csv_dir, relative_img_path_from_csv)
    
    try:
        input_img = Image.open(input_img_path).convert("RGB")
    except FileNotFoundError:
        print(f"오류: ID {img_id} 에 대한 이미지 파일을 찾을 수 없습니다: {input_img_path}. 이 항목을 건너킵니다.")
        continue
    except Exception as e:
        print(f"오류: ID {img_id} 의 이미지 로딩 중 예상치 못한 오류 발생: {e}. 경로: {input_img_path}. 이 항목을 건너킵니다.")
        continue

    cleaned_caption = clean_caption_full(caption, remove_phrases, number_words, number_regex, max_tokens=30)
    pos_prompt, neg_prompt = prompt_enhancer.enhance_caption(cleaned_caption)
    guidance_scale = dynamic_param_gen.get_optimal_guidance(cleaned_caption)
    num_inference_steps = dynamic_param_gen.get_optimal_steps(cleaned_caption)
    canny_low, canny_high = dynamic_param_gen.get_optimal_canny_params(cleaned_caption)

    control_image = preprocess_for_controlnet(input_img, detector_type="canny", low=canny_low, high=canny_high)

    output = pipe(
        prompt=pos_prompt,
        image=control_image,
        negative_prompt=neg_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    output_img = output.images[0]

    file_name = f"{img_id}.png"
    out_imgs.append(output_img)
    out_img_names.append(file_name)
    output_img.save(os.path.join(CFG.SUB_DIR, file_name))

print('Test 데이터셋에 대한 모든 이미지 생성 완료.')

# --- 제출용 임베딩 추출 (이전과 동일) ---
print('CLIP 임베딩 추출 및 저장')
import open_clip

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(CFG.EMBED_MODEL, pretrained=CFG.EMBED_PRETRAINED)
clip_model = clip_model.to(CFG.DEVICE)
clip_model.eval()

feat_imgs = []
if not out_imgs:
    print("경고: 생성된 이미지가 없습니다. 임베딩 추출을 건너킵니다.")
else:
    for output_img, img_id in tqdm(zip(out_imgs, out_img_names), total=len(out_imgs)):
        preprocessed_img = clip_preprocess(output_img).unsqueeze(0).to(CFG.DEVICE)
        with torch.no_grad():
            feat_img = clip_model.encode_image(preprocessed_img)
            feat_img = feat_img / feat_img.norm(dim=-1, keepdim=True)
        feat_img_np = feat_img.detach().cpu().numpy().reshape(-1)
        feat_imgs.append(feat_img_np)

    feat_imgs = np.array(feat_imgs)
    vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
    feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
    feat_submission.insert(0, 'ID', [os.path.splitext(n)[0] for n in out_img_names])
    csv_path = os.path.join(CFG.SUB_DIR, 'embed_submission.csv')
    feat_submission.to_csv(csv_path, index=False)
    print(f"임베딩 CSV 파일 저장 완료: {csv_path}")

# --- 제출용 ZIP 생성 (이전과 동일) ---
print("ZIP 파일 생성")
if not os.path.exists(CFG.SUB_DIR) or not os.listdir(CFG.SUB_DIR):
    print(f"경고: {CFG.SUB_DIR} 디렉토리가 비어있거나 존재하지 않습니다. ZIP 파일을 생성할 수 없습니다.")
else:
    with zipfile.ZipFile(CFG.SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in os.listdir(CFG.SUB_DIR):
            file_path = os.path.join(CFG.SUB_DIR, file_name)
            if os.path.isfile(file_path) and not file_name.startswith('.'):
                zipf.write(file_path, arcname=file_name)
    print(f"압축 완료: {CFG.SUBMISSION_ZIP}")

print("모든 테스트셋 추론 및 제출 포맷 완성!")