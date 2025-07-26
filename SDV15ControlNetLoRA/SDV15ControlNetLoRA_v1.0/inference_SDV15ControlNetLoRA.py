import gc
import os
import random
import re
import cv2
import string
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import nltk
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model 
from torchvision import transforms
import zipfile
import open_clip

# Config 
class Config:
    def __init__(self):
        self.IMG_SIZE = 512
        self.SEED = 42
        
        # --- Paths and Directories ---
        self.OUTPUT_ROOT_DIR = "./output"
        self.SUB_DIR = os.path.join(self.OUTPUT_ROOT_DIR, "submission")
        self.SUBMISSION_ZIP = os.path.join(self.OUTPUT_ROOT_DIR, "submission.zip")

        self.BEST_MODEL_DIR = os.path.join(self.OUTPUT_ROOT_DIR, 'best_model') 

 
        self.TEST_CSV = "../test.csv" 
        self.TEST_INPUT_DIR = "../" 
        # --- Inference Specific Parameters ---
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.WEIGHT_DTYPE = torch.float16 if self.DEVICE == "cuda" else torch.float32 
        self.num_inference_steps_for_submission = 50
        self.N_ATTEMPTS_PER_IMAGE = 3
        self.EMBED_MODEL = "ViT-L-14" 
        self.EMBED_PRETRAINED = "openai" 
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5" 
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
CFG = Config()
os.makedirs(CFG.SUB_DIR, exist_ok=True)

number_words = set([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "a", "an"
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

def clean_caption_full(caption, remove_phrases, number_words, number_regex, max_tokens=28):
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

class PromptEnhancer:
    def __init__(self):
        self.quality_keywords = ["masterpiece", "best quality", "high resolution", "4k", "8k"]
        self.texture_keywords = ["detailed texture", "smooth texture", "realistic texture"]
        self.lighting_keywords = ["dramatic lighting", "soft lighting", "cinematic lighting", "studio lighting"]
        self.scene_keywords = ["wide angle", "close up", "full body shot", "dynamic pose", "indoor scene", "outdoor scene"]
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
            "tan": ["tan", "sandy tan", "desert tan"],
            "silver": ["shimmering silver", "polished silver", "chrome silver"],
            "gold": ["golden", "shiny gold", "metallic gold"],
            "beige": ["soft beige", "light beige", "sandy beige"],
            "violet": ["deep violet", "vibrant violet", "amethyst violet"],
            "cyan": ["bright cyan", "electric cyan", "aqua cyan"],
            "magenta": ["bright magenta", "vibrant magenta", "fuchsia magenta"],
            "navy": ["deep navy", "dark navy", "midnight navy"],
            "olive": ["dark olive", "military olive", "earthy olive"],
            "burgundy": ["rich burgundy", "deep burgundy", "wine burgundy"],
            "maroon": ["dark maroon", "deep maroon", "reddish maroon"],
            "teal": ["bright teal", "ocean teal", "emerald teal"],
            "lime": ["bright lime", "vibrant lime", "electric lime"],
            "indigo": ["deep indigo", "dark indigo", "royal indigo"],
            "charcoal": ["dark charcoal", "deep charcoal", "smoky charcoal"],
            "peach": ["soft peach", "blush peach", "light peach"],
            "cream": ["rich cream", "vanilla cream", "ivory cream"],
            "ivory": ["soft ivory", "warm ivory", "pale ivory"],
            "turquoise": ["bright turquoise", "ocean turquoise", "sky turquoise"],
            "mint": ["cool mint", "fresh mint", "light mint"],
            "mustard": ["spicy mustard", "golden mustard", "deep mustard"],
            "coral": ["bright coral", "pinkish coral", "orange coral"],
            "gray": ["muted gray", "cool gray", "steel gray"],
            "grey": ["muted grey", "cool grey", "steel grey"], 
            "colorful": ["vibrant colors", "rich color palette", "brightly colored", "rainbow colors", "full color"]
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
            "wrong perspective", "misshapen object", "deformed objects"
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
        enhanced_caption += ", colorful, vibrant colors, maintain original structure, do not change structure, only colorize"
        negs = random.sample(self.negative_prompts_general, k=random.randint(2, 4))
        if any(word in caption for word in ['person', 'man', 'woman', 'face', 'people']):
            negs += random.sample(self.negative_prompts_person, k=2)
        if any(word in caption for word in ['tree', 'sky', 'field', 'mountain', 'river', 'cloud']):
            negs += random.sample(self.negative_prompts_landscape, k=2)
        if any(word in caption for word in ['car', 'train', 'bus', 'object', 'table', 'chair']):
            negs += random.sample(self.negative_prompts_object, k=1)
        neg_prompt = ', '.join(set(negs)) + ", bad quality, grayscale, monochromatic, desaturated, unrealistic colors"
        return enhanced_caption, neg_prompt

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

prompt_enhancer = PromptEnhancer()
dynamic_param_gen = DynamicParameterGenerator()
basic_transform = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        hsv_similarity = hsv_saturation_mean 
        img_embedding_tensor = torch.tensor(img_embedding_np, device=device, dtype=torch.float32).unsqueeze(0) 
        clip_score = F.cosine_similarity(img_embedding_tensor, text_features_for_clip_score).item()
        current_total_score = 0.6 * hsv_similarity + 0.4 * clip_score
        if current_total_score > best_score:
            best_score = current_total_score
            best_image = img_pil
            best_embedding = img_embedding_np
    return best_image, best_embedding

def run_inference(config, prompt_enhancer, dynamic_param_gen, basic_transform):
    print("\n--- 추론 파이프라인 시작 ---")

    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_PATH, torch_dtype=config.WEIGHT_DTYPE
    ).to(config.DEVICE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.MODEL_PATH, controlnet=controlnet, torch_dtype=config.WEIGHT_DTYPE
    ).to(config.DEVICE)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    base_tokenizer = pipe.tokenizer
    base_text_encoder = pipe.text_encoder
    base_vae = pipe.vae
    base_scheduler_config = pipe.scheduler.config

    try:
        pipe.controlnet = ControlNetModel.from_pretrained(
            os.path.join(config.BEST_MODEL_DIR, "controlnet_best"), 
            torch_dtype=config.WEIGHT_DTYPE
        ).to(config.DEVICE)
        pipe.unet = pipe.unet.__class__.from_pretrained(
            config.MODEL_PATH, subfolder="unet", torch_dtype=config.WEIGHT_DTYPE
        ).to(config.DEVICE)
        try:
            pipe.unet.load_adapter(os.path.join(config.BEST_MODEL_DIR, "unet_best"), adapter_name="color_lora")
            pipe.unet.set_adapter("color_lora") 
        except Exception as e_lora:
            print(f"경고: LoRA 어댑터 로드 오류: {e_lora}")
            pass 
        try:
            pipe.tokenizer = type(base_tokenizer).from_pretrained(
                os.path.join(config.BEST_MODEL_DIR, "tokenizer_best")
            )
        except Exception:
            pipe.tokenizer = base_tokenizer 
        try:
            pipe.text_encoder = type(base_text_encoder).from_pretrained(
                os.path.join(config.BEST_MODEL_DIR, "text_encoder_best"),
                torch_dtype=config.WEIGHT_DTYPE
            ).to(config.DEVICE)
        except Exception:
            pipe.text_encoder = base_text_encoder 
        pipe.scheduler = UniPCMultistepScheduler.from_config(base_scheduler_config)
    except Exception as e:
        print(f"치명적 오류: 모델 로드 중 오류 발생: {e}")
        return 

    pipe.unet.eval()
    pipe.controlnet.eval()
    if pipe.text_encoder: pipe.text_encoder.eval()
    if pipe.vae: pipe.vae.eval()
    pipe.scheduler.set_timesteps(config.num_inference_steps_for_submission)

    clip_model_for_embed, _, clip_preprocess_for_embed = open_clip.create_model_and_transforms(config.EMBED_MODEL, pretrained=config.EMBED_PRETRAINED)
    clip_model_for_embed = clip_model_for_embed.to(config.DEVICE)
    clip_model_for_embed.eval()
    with torch.no_grad():
        clip_tokenizer = open_clip.get_tokenizer(config.EMBED_MODEL)
        text_prompt_for_score = "A vibrant and colorful image, photorealistic, high quality" 
        tokenized_text = clip_tokenizer(text_prompt_for_score).to(config.DEVICE)
        text_features_for_clip_score = clip_model_for_embed.encode_text(tokenized_text)
        text_features_for_clip_score = text_features_for_clip_score / text_features_for_clip_score.norm(dim=-1, keepdim=True)

    try:
        test_df = pd.read_csv(config.TEST_CSV) 
        if test_df.empty:
            raise ValueError(f"test.csv 파일이 비어있거나 올바르지 않습니다: {config.TEST_CSV}")
        print(f"test.csv에서 {len(test_df)}개 항목 발견")
    except Exception as e:
        print(f"test.csv 로딩 오류: {e}")
        return
    
    all_test_captions = test_df['caption'].astype(str).tolist()
    remove_phrases_inference = build_remove_phrases(all_test_captions, ngram_ns=(2,3,4), topk=100)

    final_output_img_names = []
    final_output_embeddings = []

    print("제출 이미지 생성 중...")
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_id = str(row['ID'])
            if 'caption' not in row or pd.isna(row['caption']):
                continue
            caption = row['caption']
            input_img_filename = row['input_img_path'] if 'input_img_path' in row and pd.notna(row['input_img_path']) else f"{img_id}.png"
            input_img_path = os.path.join(config.TEST_INPUT_DIR, input_img_filename)
            try:
                input_img_pil = Image.open(input_img_path).convert("RGB")
            except Exception:
                continue

            cleaned_caption = clean_caption_full(caption, remove_phrases_inference, number_words, number_regex, max_tokens=30)
            candidates = []
            for attempt in range(config.N_ATTEMPTS_PER_IMAGE):
                seed_everything(config.SEED + idx * config.N_ATTEMPTS_PER_IMAGE + attempt) 
                pos_prompt, neg_prompt = prompt_enhancer.enhance_caption(cleaned_caption)
                guidance_scale = dynamic_param_gen.get_optimal_guidance(cleaned_caption)
                num_inference_steps = dynamic_param_gen.get_optimal_steps(cleaned_caption)
                canny_low, canny_high = dynamic_param_gen.get_optimal_canny_params(cleaned_caption)
                control_image = preprocess_for_controlnet(input_img_pil, detector_type="canny", low=canny_low, high=canny_high)
                control_image_tensor = basic_transform(control_image).unsqueeze(0).to(dtype=config.WEIGHT_DTYPE, device=config.DEVICE)
                output = pipe(
                    prompt=[pos_prompt], 
                    image=control_image_tensor, 
                    negative_prompt=[neg_prompt], 
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="pil"
                )
                current_img_pil = output.images[0]
                current_img_clip_embedding = calc_clip_embedding(current_img_pil, clip_model_for_embed, clip_preprocess_for_embed, config.DEVICE)
                current_img_hsv_saturation = calc_hsv(current_img_pil)
                candidates.append((current_img_pil, current_img_clip_embedding, current_img_hsv_saturation))
                del control_image_tensor
                torch.cuda.empty_cache()
                gc.collect()
            best_img, best_embedding = pick_best(candidates, text_features_for_clip_score, config.DEVICE)
            if best_img is not None and best_embedding is not None:
                file_name = f"{img_id}.png" 
                best_img.save(os.path.join(config.SUB_DIR, file_name)) 
                final_output_img_names.append(file_name)
                final_output_embeddings.append(best_embedding)

    print('모든 이미지 생성 완료.')

    if final_output_img_names:
        feat_imgs_array = np.array(final_output_embeddings)
        vec_columns = [f'vec_{i}' for i in range(feat_imgs_array.shape[1])]
        feat_submission = pd.DataFrame(feat_imgs_array, columns=vec_columns)
        feat_submission.insert(0, 'ID', final_output_img_names)
        csv_path = os.path.join(config.SUB_DIR, 'embed_submission.csv') 
        feat_submission.to_csv(csv_path, index=False)
        print(f"임베딩 CSV 저장 완료: {csv_path}")

    print("ZIP 파일 생성 중...")
    if os.path.exists(config.SUB_DIR) and os.listdir(config.SUB_DIR):
        zip_path = config.SUBMISSION_ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name in os.listdir(config.SUB_DIR):
                file_path = os.path.join(config.SUB_DIR, file_name)
                if os.path.isfile(file_path) and not file_name.startswith('.'):
                    zipf.write(file_path, arcname=file_name)
        print(f"압축 완료: {zip_path}")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    print("추론 프로세스 시작...")
    run_inference(CFG, prompt_enhancer, dynamic_param_gen, basic_transform)
    print("추론 프로세스 완료.")
