import torch
import re
import spacy

# from src.preprocessing import (
#     pre_clean_caption, extract_nouns_only, safe_prompt_str, preprocess_for_controlnet
# )
# from src.prompt_enhancer import PromptEnhancer, get_sfw_template

# nlp = spacy.load("en_core_web_sm")
# clean_caption = pre_clean_caption(caption, nlp, number_words, number_regex, EXTRA_STOPWORDS, STOPWORDS)
# nouns = extract_nouns_only(caption, nlp)
# prompt_enhancer = PromptEnhancer()
# pos_prompt = prompt_enhancer.make_enhanced_prompt(clean_caption)
# neg_prompt = prompt_enhancer.get_base_negative_prompt()
# sfw_prompt = get_sfw_template(caption, nlp)

# --- fix tail 프롬프트 강화 템플릿 ---
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
    
# --- NSFW 판별 ---
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

# --- nsfw용 안전 프롬프트 템플릿 ---
def get_sfw_template(caption, fallback="a high quality photo"):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(caption.lower())
    key_phrases = []
    seen = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text
        if phrase not in seen:
            key_phrases.append(phrase)
            seen.add(phrase)
    for i in range(len(doc)-1):
        if doc[i].pos_ == "ADJ":
            j = i
            while j+1 < len(doc) and doc[j+1].pos_ == "ADJ":
                j += 1
            if j+1 < len(doc) and doc[j+1].pos_ == "NOUN":
                phrase = " ".join([t.text for t in doc[i:j+2]])
                if phrase not in seen:
                    key_phrases.append(phrase)
                    seen.add(phrase)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} or token.tag_ in {"VBG", "VBN"}:
            if token.text not in seen:
                key_phrases.append(token.text)
                seen.add(token.text)
    if not key_phrases:
        return fallback
    return f"a photo of {', '.join(key_phrases)}"
