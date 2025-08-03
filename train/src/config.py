class Config:
    def __init__(self):
        self.IMG_SIZE = 512  # 입력/출력 이미지 크기
        self.GUIDANCE_SCALE = 7.5  # Diffusion guidance scale (classifier-free guidance)
        self.NUM_INFERENCE_STEPS = 35  # Diffusion sampling 단계 수 (추론 시)
        self.SEED = 42  # 고정
        
        self.OUTPUT_DIR = "./output7"  # 결과물(모델, 샘플, 로그) 저장 루트 디렉토리
        self.TRAIN_CSV = "../train.csv"  # 학습 데이터 경로(csv)
        self.INPUT_DIR = ".."  # 입력 이미지 루트 디렉토리
        self.GT_DIR = ".."  # GT(정답) 이미지 루트 디렉토리
        
        self.LR = 1e-4  # 학습률
        self.BATCH_SIZE = 10  # 배치 크기
        self.NUM_WORKERS = 2  # DataLoader의 worker 개수
        self.EPOCHS = 30  # 전체 학습 에폭 수
        
        self.MAX_DATA = None  # 사용할 데이터 최대 개수 (None이면 전체)
        self.LAMBDA_L1 = 2.0  # L1 손실 가중치
        self.LAMBDA_CLIP = 0.0  # CLIP 손실 가중치
        self.LAMBDA_LPIPS = 0.5  # LPIPS 손실 가중치
        self.LAMBDA_SSIM = 0.5  # SSIM 손실 가중치

        self.CLIP_MODEL = "openai/clip-vit-base-patch32"  # 사용할 CLIP 모델 이름/경로
        self.MODEL_PATH = "runwayml/stable-diffusion-v1-5"  # 메인 diffusion 모델 경로
        self.PRETRAINED_MODEL_NAME_OR_PATH = "runwayml/stable-diffusion-v1-5"  # 사전학습 모델
        self.CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"  # ControlNet 모델 경로

        self.PROJECT_NAME = "colorization_training"  # 로그/트래커용 프로젝트 이름
        self.PATIENCE = 9999  # Early stopping patience (intervals)

        self.MAX_PROMPT_TOKENS = 77  # 프롬프트 최대 토큰 수 (CLIP 토큰 기준)
        self.NSFW_KEYWORDS = ["naked", "sex", "porn", "erotic", "nude", "breast", "ass", "penis", "vagina"]  # NSFW 필터용 키워드
        self.SFW_CAPTION_REPLACEMENT = "a high quality image, realistic, clean, beautiful, bright, colorful"  # NSFW 감지 시 대체 프롬프트

        self.GRADIENT_ACCUMULATION_STEPS = 2  # Gradient Accumulation Step 수
        self.MAX_GRAD_NORM = 1.0  # Gradient clipping 최대 norm

        self.LR_SCHEDULER_TYPE = "constant"  # 스케줄러 타입
        self.LR_WARMUP_STEPS = 0  # LR warmup step 수
        self.ADAM_BETA1 = 0.9  # AdamW beta1
        self.ADAM_BETA2 = 0.999  # AdamW beta2
        self.ADAM_WEIGHT_DECAY = 1e-2  # AdamW weight decay
        self.ADAM_EPSILON = 1e-08  # AdamW epsilon

        self.MIXED_PRECISION = "bf16"  # mixed precision 옵션 ("no", "fp16", "bf16")
        self.REPORT_TO = "tensorboard"  # 로깅 툴 ("tensorboard", "wandb", "all")

        self.MAX_TRAIN_STEPS = 100  # 전체 학습 스텝 수 (None이면 EPOCHS로 자동 계산)
        self.RESUME_FROM_CHECKPOINT = ""  # 체크포인트에서 재시작 경로 (예: "./output5/checkpoint-40")

        self.SAMPLE_SAVE_START_STEP = 5  # 샘플 이미지 저장 시작 스텝
        self.SAMPLE_SAVE_END_STEP = 100  # 샘플 이미지 저장 종료 스텝
        self.NUM_SAMPLES_TO_SAVE = None  # 검증/샘플 저장시 실제 사용할 이미지 개수

        self.MAX_CHECKPOINTS_TO_KEEP = 30  # 남길 체크포인트 최대 개수
        self.LOG_INTERVAL = 10  # train_loss 등 로그 저장 step 간격
        self.VAL_INTERVAL = 1  # 에폭당 검증 주기 (사용 X, 아래로 통합)
        self.CONTROLNET_STRENGTH = 1.0  # ControlNet conditioning scale
        self.SAVE_AND_VAL_INTERVAL = 5  # 저장/검증을 실행할 step 간격
CFG = Config()

# --- 색상 단어 리스트 (프롬프트/클린용) ---
color_words = set([
    'white', 'black', 'gray', 'grey', 'red', 'blue', 'green', 'yellow', 'orange', 'pink',
    'purple', 'brown', 'tan', 'silver', 'gold', 'beige', 'violet', 'cyan', 'magenta',
    "navy", "olive", "burgundy", "maroon", "teal", "lime", "indigo", "charcoal",
    "peach", "cream", 'ivory', 'turquoise', 'mint', 'mustard', 'coral', 'colorful'
])
 # --- 숫자 관련 단어 (프롬프트/클린용) ---
number_words = set([
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "a", "an"
])
# --- 숫자/순서 표현 정규표현식 (프롬프트/클린용) ---
number_regex = re.compile(r'\b(\d+|[aA]n?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b')
