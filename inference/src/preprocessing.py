'''
전처리·정규화·토큰화 관련 함수
'''
import re
import string
import numpy as np
import cv2
from PIL import Image

# from src.preprocessing import (
#     lemma_with_irregular,
#     pre_clean_caption,
#     extract_nouns_only,
#     safe_prompt_str,
#     preprocess_for_controlnet
# )

# --- 불규칙 복수/단수 변환 --- 
def lemma_with_irregular(token):
    text = token.text.lower()
    return irregular_plural_map.get(text, token.lemma_.lower())

# --- 관용 표현, 특수문자, 숫자, number_words 제거 ---
def pre_clean_caption(caption):
    patterns = [
        r'\bin this image\b', r'\bin the picture\b', r'\bin this photo\b',
        r'\bare there\b', r'\bthere are\b', r'\bis there\b', r'\bthere is\b',
        r'\bthis is\b', r'\bwhat is\b', r'\bwhat do .* have in common\b',
        r'\bimage of\b', r'\bphoto of\b', r'\bpicture of\b', # 추가
        r'\bcaptured from\b', r'\brecorded from\b', # 추가
    ]
    c = caption.lower()
    for pat in patterns:
        c = re.sub(pat, '', c)
    # 특수문자/숫자/number_words 제거
    c = c.translate(str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    c = number_regex.sub(' ', c)
    c = ' '.join([w for w in c.split() if w not in number_words])
    # stopwords 제거 (EXTRA_STOPWORDS)
    c = ' '.join([w for w in c.split() if w not in EXTRA_STOPWORDS])
    c = re.sub(r'\s+', ' ', c).strip()
    return c

# --- 캡션에서 명사만 최대 n개 추출/,로 연결 ---
def extract_nouns_only(caption, nlp, n=10):
    doc = nlp(caption)
    nouns = []
    used = set()
    for token in doc:
        lemma = lemma_with_irregular(token)
        if (token.pos_ in ["NOUN", "PROPN"]
            and lemma not in used
            and len(lemma) > 1
        ):
            nouns.append(lemma)
            used.add(lemma)
        if len(nouns) >= n:
            break
    return ', '.join(nouns)

# --- clip 기준 max_len 토큰 조절 77 ---
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

# --- ControlNet 입력용 Cann 엣지 이미지 생성 ---
def preprocess_for_controlnet(image_pil, detector_type="canny", low=100, high=200):
    import numpy as np
    import cv2
    from PIL import Image
    image_np = np.array(image_pil)
    if detector_type == "canny":
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edged_image = cv2.Canny(gray_image, low, high)
        control_image = Image.fromarray(edged_image).convert("RGB")
    else:
        control_image = image_pil
    return control_image
