import os
import gc
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DConditionModel, ControlNetModel
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from scipy.spatial.distance import cosine
import open_clip
import cv2
import spacy
from zipfile import ZipFile, ZIP_DEFLATED
import re
import string
import webcolors

# --- Config ---
class Config:
    def __init__(self):
        self.ROOT_PATH = '/home/guest01/colorize'
        self.NO = "NO.9"
        self.LORA_UNET_VERSION = "unet_lora_280"
        self.CLIP_MODEL = "openai/clip-vit-base-patch32"
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"
        self.PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        self.TEST_CSV = os.path.join(self.ROOT_PATH, 'test.csv')
        self.TEST_INPUT_DIR = os.path.join(self.ROOT_PATH, 'test/input_image')
        self.LORA_UNET_DIR = os.path.join(self.ROOT_PATH, self.NO, self.LORA_UNET_VERSION)
        self.SUB_DIR = os.path.join(self.ROOT_PATH, 'submission')
        self.SUBMISSION_ZIP = os.path.join(self.ROOT_PATH, 'NO.9_280_submission_4E.zip')
        self.IMG_SIZE = 512
        self.WEIGHT_DTYPE = torch.float16
        self.USE_ENSEMBLE = True
        self.NUM_ENSEMBLE_SAMPLES = 3
        self.USE_CLIP_SCORING = True
        self.HSV_CLIP_RATIO = (0.6, 0.4)
        self.SEED = 42
        self.MAX_PROMPT_TOKENS = 77
        self.EMBED_MODEL = "ViT-L-14"
        self.EMBED_PRETRAINED = "openai"
        self.CONTROLNET_STRENGTH = 1.0
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

nlp = spacy.load("en_core_web_sm")

color_words = set([
    'white', 'black', 'gray', 'grey', 'red', 'blue', 'green', 'yellow', 'orange', 'pink',
    'purple', 'brown', 'tan', 'silver', 'gold', 'beige', 'violet', 'cyan', 'magenta',
    "navy", "olive", "burgundy", "maroon", "teal", "lime", "indigo", "charcoal",
    "peach", "cream", 'ivory', 'turquoise', 'mint', 'mustard', 'coral', 'colorful'
])
number_words = set([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
])
number_regex = re.compile(r'\b(\d+|[aA]n?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b')
irregular_plural_map = {
    "people": "person", "men": "man", "women": "woman", "children": "child", "teeth": "tooth",
    "feet": "foot", "mice": "mouse", "geese": "goose", "cacti": "cactus", "fungi": "fungus",
    "nuclei": "nucleus", "syllabi": "syllabus", "analyses": "analysis", "diagnoses": "diagnosis", "theses": "thesis",
    "crises": "crisis", "phenomena": "phenomenon", "criteria": "criterion", "data": "datum",
    "appendices": "appendix", "oxen": "ox", "media": "medium", "axes": "axis", "bases": "basis",
    "alumni": "alumnus", "stimuli": "stimulus", "curricula": "curriculum", "indices": "index", "leaves": "leaf", "lives": "life",
}
EXTRA_STOPWORDS = {
    "this", "image", "the", "a", "an", "i", "you", "it", "which", "that", "we",
    "many", "some", "few", "what", "is", "are", "was", "were", "in", "on", "at",
    "to", "by", "with", "for", "of", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
}
STOPWORDS = set(nlp.Defaults.stop_words) | EXTRA_STOPWORDS

def lemma_with_irregular(token):
    text = token.text.lower()
    return irregular_plural_map.get(text, token.lemma_.lower())

def pre_clean_caption(caption):
    patterns = [
        r'\bin this image\b', r'\bin the picture\b', r'\bin this photo\b',
        r'\bare there\b', r'\bthere are\b', r'\bis there\b', r'\bthere is\b',
        r'\bthis is\b', r'\bwhat is\b', r'\bwhat do .* have in common\b',
        r'\bimage of\b', r'\bphoto of\b', r'\bpicture of\b',
        r'\bcaptured from\b', r'\brecorded from\b',
    ]
    cleaned_caption = caption.lower()
    for pat in patterns:
        cleaned_caption = re.sub(pat, '', cleaned_caption)
    cleaned_caption = cleaned_caption.translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    cleaned_caption = number_regex.sub(' ', cleaned_caption)
    cleaned_caption = ' '.join([w for w in cleaned_caption.split() if w not in number_words])
    cleaned_caption = ' '.join([w for w in cleaned_caption.split() if w not in EXTRA_STOPWORDS])
    cleaned_caption = re.sub(r'\s+', ' ', cleaned_caption).strip()
    doc = nlp(cleaned_caption)
    lemma_words = []
    for token in doc:
        word_lemma = lemma_with_irregular(token)
        if word_lemma not in STOPWORDS and len(word_lemma) > 1:
            lemma_words.append(word_lemma)
    cleaned_caption_final = ' '.join(lemma_words)
    return cleaned_caption_final

def extract_nouns_only(caption, n=10):
    doc = nlp(pre_clean_caption(caption))
    nouns = []
    used = set()
    for token in doc:
        lemma = lemma_with_irregular(token)
        if (token.pos_ in ["NOUN", "PROPN"]
            and lemma not in used
            and lemma not in STOPWORDS
            and len(lemma) > 1
        ):
            nouns.append(lemma)
            used.add(lemma)
        if len(nouns) >= n:
            break
    return ', '.join(nouns)

def safe_prompt_str(prompt_str, tokenizer, max_len=77):
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=True, truncation=True, return_tensors="pt")[0]
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len-1]
    final_prompt_str = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return final_prompt_str

def get_sfw_template(caption, fallback="a high quality photo"):
    doc = nlp(caption.lower())
    key_phrases = []
    seen_phrases = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text
        if phrase not in seen_phrases:
            key_phrases.append(phrase)
            seen_phrases.add(phrase)
    for i in range(len(doc)-1):
        if doc[i].pos_ == "ADJ":
            j = i
            while j+1 < len(doc) and doc[j+1].pos_ == "ADJ":
                j += 1
            if j+1 < len(doc) and doc[j+1].pos_ == "NOUN":
                phrase = " ".join([t.text for t in doc[i:j+2]])
                if phrase not in seen_phrases:
                    key_phrases.append(phrase)
                    seen_phrases.add(phrase)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} or token.tag_ in {"VBG", "VBN"}:
            if token.text not in seen_phrases:
                key_phrases.append(token.text)
                seen_phrases.add(token.text)
    if not key_phrases:
        return fallback
    return f"a photo of {', '.join(key_phrases)}"

def extract_dominant_colors(image, topk=3):
    from sklearn.cluster import KMeans
    img = image.resize((32, 32)).convert('RGB')
    arr = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=topk, n_init='auto').fit(arr)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

def print_dominant_colors_with_names(image_pil):
    dominant_colors = extract_dominant_colors(image_pil, topk=3)
    color_names = []
    for rgb_tuple in dominant_colors:
        try:
            name = webcolors.rgb_to_name(rgb_tuple, spec='css3')
        except Exception:
            # 최신 webcolors에서는 CSS3_NAMES_TO_HEX가 있는 경우도, 없는 경우도 있음
            try:
                names_map = webcolors.CSS3_NAMES_TO_HEX
            except AttributeError:
                names_map = {
                    # fallback 기본값 (필요시 여기에 140 CSS3 이름 추가)
                    'white': '#ffffff', 'black': '#000000', 'red': '#ff0000',
                    'blue': '#0000ff', 'green': '#008000', 'yellow': '#ffff00', 'orange': '#ffa500', 'pink': '#ffc0cb',
                    'purple': '#800080', 'brown': '#a52a2a', 'gray': '#808080', 'grey': '#808080'
                }
            min_dist = float("inf")
            closest_name = None
            for name, hex_code in names_map.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
                dist = (r_c - rgb_tuple[0]) ** 2 + (g_c - rgb_tuple[1]) ** 2 + (b_c - rgb_tuple[2]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_name = name
            name = closest_name
        color_names.append(name)
    print(f"[dominant RGB] {dominant_colors} | [color name] {color_names}")
    return color_names

class PromptEnhancer:
    def __init__(self):
        self.fixed_tail_template = (
            "high detail, neutral tone, photorealistic, real people, sharp, natural color, original color, actual person,"
            "preserve structure, balanced tone, realistic coloration, unexaggerated colors, not oversaturated"
        )
        self.base_negative_prompts = (
            "bad quality, vivid, uncanny, vibrant, sketch, monochrome, grayscale, low detail, deformed, distorted, "
            "missing face, blurry, overexposed, oversaturated, too bright, high saturation, mutated hands, low contrast, artificial, neon, "
            "unrealistic, burnt, posterization, color artifact, noisy"
        )
    def make_enhanced_prompt(self, subject_text):
        return self.fixed_tail_template
    def get_base_negative_prompt(self):
        return self.base_negative_prompts

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
            self.TYPE_PERSON: (6.5, 8.5),
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
            self.TYPE_PERSON: ((50, 90), (100, 150)),
            self.TYPE_COMPLEX_DETAIL:((30, 70), (90, 140)),
            self.TYPE_DEFAULT: ((70, 120), (140, 200)),
            self.TYPE_SIMPLE_OUTLINE: ((80, 180), (120, 220))
        }
        self.guidance_keywords_map = {
            self.TYPE_CARTOON: ['cartoon', 'drawing', 'illustration', 'anime'],
            self.TYPE_PERSON: ['person', 'people', 'man', 'woman', 'face', 'shirt', 'jacket', 'hat', 'boy', 'girl', 'child', 'people'],
            self.TYPE_LANDSCAPE: ['tree', 'trees', 'sky', 'mountain', 'field', 'grass', 'clouds', 'building', 'buildings', 'city', 'street', 'road', 'river', 'lake', 'ocean'],
            self.TYPE_OBJECT: ['car', 'bus', 'train', 'table', 'chair', 'cow', 'bowl', 'dog', 'cat', 'book', 'bottle', 'cup', 'food', 'flower', 'clock', 'sign', 'window', 'door']
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
        caption_str = str(caption).lower()
        caption_str = caption_str.translate(str.maketrans('', '', string.punctuation))
        caption_str = re.sub(r'\s+', ' ', caption_str).strip()
        return caption_str

    def _get_category(self, caption, category_map):
        cleaned_caption_for_keywords = self._clean_caption_for_keywords(caption)
        for category, keywords in category_map.items():
            if any(word in cleaned_caption_for_keywords for word in keywords):
                return category
        return self.TYPE_DEFAULT

    def get_optimal_guidance(self, caption):
        category = self._get_category(caption, self.guidance_keywords_map)
        return random.uniform(*self.guidance_ranges[category])

    def get_optimal_steps(self, caption):
        cleaned_caption_for_keywords = self._clean_caption_for_keywords(caption)
        word_count = len(cleaned_caption_for_keywords.split())
        if any(word in cleaned_caption_for_keywords for word in self.guidance_keywords_map[self.TYPE_CARTOON]):
            return random.randint(*self.step_ranges[self.TYPE_CARTOON])
        elif word_count < 8:
            return random.randint(*self.step_ranges[self.TYPE_SHORT_CAPTION])
        elif word_count > 16:
            return random.randint(*self.step_ranges[self.TYPE_LONG_CAPTION])
        else:
            return random.randint(*self.step_ranges[self.TYPE_DEFAULT])

    def get_optimal_canny_params(self, caption=""):
        cleaned_caption_for_keywords = self._clean_caption_for_keywords(caption)
        if any(word in cleaned_caption_for_keywords for word in self.canny_complex_keywords):
            low_range, high_range = self.canny_thresholds[self.TYPE_COMPLEX_DETAIL]
        elif any(word in cleaned_caption_for_keywords for word in self.canny_simple_keywords):
            low_range, high_range = self.canny_thresholds[self.TYPE_SIMPLE_OUTLINE]
        elif any(word in cleaned_caption_for_keywords for word in self.guidance_keywords_map[self.TYPE_PERSON]):
            low_range, high_range = self.canny_thresholds[self.TYPE_PERSON]
        else:
            low_range, high_range = self.canny_thresholds[self.TYPE_DEFAULT]
        low_threshold = random.randint(low_range[0], low_range[1])
        high_threshold = random.randint(high_range[0], high_range[1])
        return low_threshold, high_threshold

class CLIPScorer:
    def __init__(self, clip_model, clip_preprocess, device):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

    def calculate_clip_score(self, image, caption):
        try:
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = open_clip.tokenize([caption]).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
                text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
                return torch.cosine_similarity(image_features, text_features).item()
        except Exception as e:
            print("CLIP 계산 오류:", e)
            return 0.0

def hsv_hist_similarity(img1, img2, bins=32):
    hsv1 = np.array(img1.convert('HSV')).flatten()
    hsv2 = np.array(img2.convert('HSV')).flatten()
    hist1 = np.histogram(hsv1, bins=bins, range=(0,255))[0]
    hist2 = np.histogram(hsv2, bins=bins, range=(0,255))[0]
    hist1 = hist1 / (np.linalg.norm(hist1) + 1e-8)
    hist2 = hist2 / (np.linalg.norm(hist2) + 1e-8)
    sim = 1 - cosine(hist1, hist2)
    return sim

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

def get_clip_nsfw_score(caption, clip_model, clip_tokenizer, device, threshold=0.8):
    candidate_labels = [
        "safe",
        "nsfw"
    ]
    text_inputs = clip_tokenizer(candidate_labels).to(device)
    caption_inputs = clip_tokenizer([caption]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        caption_features = clip_model.encode_text(caption_inputs)
        caption_features /= caption_features.norm(dim=-1, keepdim=True)
        similarity = (caption_features @ text_features.T).squeeze(0)
        probs = similarity.softmax(dim=0)
    return float(probs[1]) >= threshold

def run_inference(config, prompt_enhancer, dynamic_param_gen):
    final_output_img_names = []
    final_output_embeddings_for_submission = []
    all_hsv_scores = []
    all_clip_scores = []
    all_combined_scores = []
    print("\n--- 추론 시작 ---")
    unet = UNet2DConditionModel.from_pretrained(
        config.MODEL_PATH,
        subfolder="unet",
        torch_dtype=config.WEIGHT_DTYPE
    )
    print("DEBUG: UNet 로드 완료")

    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_PATH,
        torch_dtype=config.WEIGHT_DTYPE
    )
    print("DEBUG: ControlNet 로드 완료")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.MODEL_PATH,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=config.WEIGHT_DTYPE,
        safety_checker=None,
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("DEBUG: xFormers memory efficient attention 활성화 시도.")
    except Exception as e:
        print(f"WARN: xFormers 활성화 실패 또는 xFormers 미설치: {e}. 다른 메모리 최적화를 시도합니다.")
    pipe.enable_vae_slicing()
    print("DEBUG: VAE slicing 활성화")
    pipe.to(config.DEVICE)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.check_inputs = lambda *args, **kwargs: None # !!!

    lora_unet_weights_path = os.path.join(config.LORA_UNET_DIR, "adapter_model.safetensors")
    assert os.path.exists(lora_unet_weights_path), f"LoRA weights file not found: {lora_unet_weights_path}"
    pipe.load_lora_weights(
        lora_unet_weights_path,
        adapter_name="default",
        force_merge=True
    )
    print(f"DEBUG: LoRA '{lora_unet_weights_path}'병합 성공")
    torch.cuda.empty_cache(); gc.collect()
    pipe.unet.eval()
    pipe.controlnet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    tokenizer_for_sd = CLIPTokenizer.from_pretrained(config.MODEL_PATH, subfolder="tokenizer")
    clip_model_for_submission, _, clip_preprocess_for_submission = open_clip.create_model_and_transforms(
        config.EMBED_MODEL, pretrained=config.EMBED_PRETRAINED)
    clip_model_for_submission = clip_model_for_submission.to(config.DEVICE).eval()
    clip_tokenizer_for_nsfw = open_clip.get_tokenizer(config.EMBED_MODEL)

    clip_scorer = CLIPScorer(clip_model_for_submission, clip_preprocess_for_submission, CFG.DEVICE)
    test_df = pd.read_csv(config.TEST_CSV)
    print("DEBUG: 이미지 생성 루프 시작")

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating Images"):
        img_id = str(row['ID'])
        caption_original = row['caption']
        original_img_path_in_csv = row.get('input_img_path', f"{img_id}.png")
        input_img_base_filename = os.path.basename(original_img_path_in_csv)
        input_img_path = os.path.join(config.TEST_INPUT_DIR, input_img_base_filename)

        try:
            input_img_pil = Image.open(input_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {input_img_path}: {e}")
            continue

        color_names = print_dominant_colors_with_names(input_img_pil)
        nouns_only = extract_nouns_only(caption_original, n=10)

        is_nsfw = get_clip_nsfw_score(
            caption_original, clip_model_for_submission, open_clip.get_tokenizer(config.EMBED_MODEL), config.DEVICE, threshold=0.5
        )

        if is_nsfw:
            main_part = get_sfw_template(caption_original)
        else:
            if color_names:
                main_part = ', '.join(color_names + [nouns_only])
            else:
                main_part = nouns_only

        tail_prompt = prompt_enhancer.make_enhanced_prompt('')
        positive_prompt = f"{main_part}, {tail_prompt}"

        positive_prompt = safe_prompt_str(positive_prompt, CLIPTokenizer.from_pretrained(config.MODEL_PATH, subfolder="tokenizer"), max_len=config.MAX_PROMPT_TOKENS)
        negative_prompt = prompt_enhancer.get_base_negative_prompt()
        negative_prompt = safe_prompt_str(negative_prompt, CLIPTokenizer.from_pretrained(config.MODEL_PATH, subfolder="tokenizer"), max_len=config.MAX_PROMPT_TOKENS)

        print(f"[{img_id}] 원본 caption: {caption_original}")
        print(f"[{img_id}] 최종 pos_prompt: {positive_prompt} (token count: {len(tokenizer_for_sd.encode(positive_prompt, add_special_tokens=True))})")

        candidate_images = []
        candidate_hsv_scores = []
        candidate_clip_scores = []

        for sample_idx in range(config.NUM_ENSEMBLE_SAMPLES if config.USE_ENSEMBLE else 1):
            guidance_scale_value = dynamic_param_gen.get_optimal_guidance(caption_original)
            num_steps_inference = dynamic_param_gen.get_optimal_steps(caption_original)
            canny_low_val, canny_high_val = dynamic_param_gen.get_optimal_canny_params(caption_original)

            control_image_processed = preprocess_for_controlnet(input_img_pil, detector_type="canny", low=canny_low_val, high=canny_high_val)
            generator_seed = torch.Generator(device=config.DEVICE).manual_seed(config.SEED + idx * 100 + sample_idx)
            with torch.autocast(device_type=config.DEVICE.type, dtype=config.WEIGHT_DTYPE):
                output_generation = pipe(
                    prompt=positive_prompt,
                    image=control_image_processed,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale_value,
                    num_inference_steps=num_steps_inference,
                    output_type="pil",
                    generator=generator_seed,
                    controlnet_conditioning_scale=config.CONTROLNET_STRENGTH
                )
            generated_img = output_generation.images[0]
            candidate_images.append(generated_img)
            hsv_score_current = hsv_hist_similarity(generated_img, input_img_pil)
            candidate_hsv_scores.append(hsv_score_current)
            if config.USE_CLIP_SCORING:
                clip_score_current = clip_scorer.calculate_clip_score(generated_img, caption_original)
            else:
                clip_score_current = 0.0
            candidate_clip_scores.append(clip_score_current)
            print(f"  샘플{sample_idx+1} | HSV: {hsv_score_current:.3f} | CLIP: {clip_score_current:.3f}")

        hsv_weight, clip_weight = config.HSV_CLIP_RATIO
        combined_scores_for_candidates = np.array(candidate_hsv_scores) * hsv_weight + np.array(candidate_clip_scores) * clip_weight
        best_candidate_idx = int(np.argmax(combined_scores_for_candidates))

        all_hsv_scores.append(candidate_hsv_scores[best_candidate_idx])
        all_clip_scores.append(candidate_clip_scores[best_candidate_idx])
        all_combined_scores.append(combined_scores_for_candidates[best_candidate_idx])

        best_image_for_output = candidate_images[best_candidate_idx]
        print(f"  [{img_id}] 선택된 샘플: {best_candidate_idx+1}, HSV={candidate_hsv_scores[best_candidate_idx]:.3f}, CLIP={candidate_clip_scores[best_candidate_idx]:.3f}")

        output_file_name = f"{img_id}.png"
        save_path_image = os.path.join(config.SUB_DIR, output_file_name)
        best_image_for_output.save(save_path_image)
        final_output_img_names.append(output_file_name)

        processed_img_for_submission_embed = clip_preprocess_for_submission(best_image_for_output).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            image_features_for_submission = clip_model_for_submission.encode_image(processed_img_for_submission_embed)
            image_features_for_submission /= image_features_for_submission.norm(dim=-1, keepdim=True)
        final_output_embeddings_for_submission.append(image_features_for_submission.detach().cpu().numpy().reshape(-1))

        torch.cuda.empty_cache(); gc.collect()

    if final_output_img_names:
        feature_images_array = np.array(final_output_embeddings_for_submission)
        vector_columns = [f'vec_{i}' for i in range(feature_images_array.shape[1])]
        feature_submission_df = pd.DataFrame(feature_images_array, columns=vector_columns)
        feature_submission_df.insert(0, 'ID', [os.path.splitext(n)[0] for n in final_output_img_names])
        csv_output_path = os.path.join(config.SUB_DIR, 'embed_submission.csv')
        feature_submission_df.to_csv(csv_output_path, index=False)
        print(f"임베딩 CSV 저장 완료: {csv_output_path}")

        with ZipFile(config.SUBMISSION_ZIP, 'w', ZIP_DEFLATED) as submission_zip_file:
            for fname in final_output_img_names:
                submission_zip_file.write(os.path.join(config.SUB_DIR, fname), fname)
            submission_zip_file.write(csv_output_path, 'embed_submission.csv')
        print(f"최종 결과 압축 완료: {config.SUBMISSION_ZIP}")

    if len(all_hsv_scores) > 0:
        mean_hsv_score = np.mean(all_hsv_scores)
        mean_clip_score = np.mean(all_clip_scores)
        mean_combined_score = np.mean(all_combined_scores)
    else:
        mean_hsv_score = mean_clip_score = mean_combined_score = 0

    print("="*50)
    print(f"전체 평균 HSV 점수:   {mean_hsv_score:.4f}")
    print(f"전체 평균 CLIP 점수:  {mean_clip_score:.4f}")
    print(f"최종 앙상블 평균점수: {mean_combined_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    seed_everything(CFG.SEED)
    prompt_enhancer = PromptEnhancer()
    dynamic_param_gen = DynamicParameterGenerator()
    run_inference(CFG, prompt_enhancer, dynamic_param_gen)
    torch.cuda.empty_cache()
    gc.collect()