'''
프로젝트 전체 설정 및 하이퍼파라미터
'''
import os
import torch

# from src.config import CFG
# os.makedirs(CFG.SUB_DIR, exist_ok=True)

class Config:
    # 경로
    ROOT_PATH = '/home/guest01/colorize'
    NO = "NO.9" # 실험 번호 폴더 (로라 저장)
    LORA_UNET_VERSION = "unet_lora_280" # 실험 폴더 내 로라_스텝(체크포인트)
    MODEL_PATH = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"

    TEST_CSV = os.path.join(ROOT_PATH, 'test.csv')
    TEST_INPUT_DIR = os.path.join(ROOT_PATH, 'test/input_image')
    LORA_UNET_DIR = os.path.join(ROOT_PATH, NO, LORA_UNET_VERSION)
    SUB_DIR = os.path.join(ROOT_PATH, 'submission')
    SUBMISSION_ZIP = os.path.join(ROOT_PATH, 'NO.9_280_submission_4E.zip')

    # 이미지/모델 관련
    IMG_SIZE = 512
    WEIGHT_DTYPE = torch.float16
    CONTROLNET_STRENGTH = 1.0 

    # 앙상블/score
    USE_ENSEMBLE = True
    NUM_ENSEMBLE_SAMPLES = 3
    USE_CLIP_SCORING = True
    HSV_CLIP_RATIO = (0.6, 0.4)

    # 기타 하이퍼파라미터
    SEED = 42
    MAX_PROMPT_TOKENS = 77

    # CLIP 임베딩
    EMBED_MODEL = "ViT-L-14"
    EMBED_PRETRAINED = "openai"

    # 디바이스
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config

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
