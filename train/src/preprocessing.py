'''
텍스트/ 이미지 전처리 캡션 클린, 토큰화, 숫자/색상 정제 함수
'''
def simple_caption_clean(
    caption,
    number_words,
    number_regex,
    remove_phrases=None,
    color_words=None
):
    c = str(caption).lower()
    c = c.translate(str.maketrans('', '', string.punctuation))
    c = number_regex.sub(' ', c)
    c = ' '.join([w for w in c.split() if w not in number_words])
    c = re.sub(r'\s+', ' ', c).strip()

    # ngram 제거(색상 단어 포함된 phrase는 남김)
    if remove_phrases and color_words:
        non_color_phrases = [
            p for p in remove_phrases if not any(color in p for color in color_words)
        ]
        for phrase in non_color_phrases:
            c = re.sub(r'[\s,.!?;:]*' + re.escape(phrase) + r'[\s,.!?;:]*', ' ', c)
        c = re.sub(r'\s+', ' ', c).strip()
    return c

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

def safe_prompt_str(prompt_str, tokenizer, max_len=77):
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=True, truncation=True, max_length=max_len, return_tensors="pt")[0]
    prompt_str = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
        )
    return prompt_str

def tensor_to_pil(tensor):
    pil_images = []
    if tensor.dim() == 4:
        # 배치 처리
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i].cpu().float()
            img_tensor = (img_tensor + 1) / 2.0 if img_tensor.min() < 0 or img_tensor.max() > 1 else img_tensor
            img_tensor = torch.clamp(img_tensor, 0, 1)
            image_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(image_np))
    elif tensor.dim() == 3:
        # 단일 이미지 처리 (기존 동작 유지)
        img_tensor = tensor.cpu().float()
        img_tensor = (img_tensor + 1) / 2.0 if img_tensor.min() < 0 or img_tensor.max() > 1 else img_tensor
        img_tensor = torch.clamp(img_tensor, 0, 1)
        image_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(image_np))
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Expected 3 or 4.")
    return pil_images if len(pil_images) > 1 else pil_images[0]
