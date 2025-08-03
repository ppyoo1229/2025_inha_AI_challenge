'''
기타 비교/분석/시각화 관련 함수들
'''

# --------- CLIP 점수 계산기 ---------
class CLIPScorer:
    def __init__(self, clip_model, clip_preprocess, device):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

    def calculate_clip_score(self, image, caption):
        try:
            img_t = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            txt = open_clip.tokenize([caption]).to(self.device)
            with torch.no_grad():
                img_f = self.clip_model.encode_image(img_t)
                txt_f = self.clip_model.encode_text(txt)
                img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-8)
                txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-8)
                return torch.cosine_similarity(img_f, txt_f).item()
        except Exception as e:
            print("CLIP 계산 오류:", e)
            return 0.0

# --------- HSV 유사도 계산기 ---------
def hsv_hist_similarity(img1, img2, bins=32):
    hsv1 = np.array(img1.convert('HSV')).flatten()
    hsv2 = np.array(img2.convert('HSV')).flatten()
    hist1 = np.histogram(hsv1, bins=bins, range=(0,255))[0]
    hist2 = np.histogram(hsv2, bins=bins, range=(0,255))[0]
    hist1 = hist1 / (np.linalg.norm(hist1) + 1e-8)
    hist2 = hist2 / (np.linalg.norm(hist2) + 1e-8)
    sim = 1 - cosine(hist1, hist2)
    return sim

# --- 원본 비교 확인용 2분할 사진 ---
def save_2up_image(original_img_path, generated_img, save_dir, img_id):
    original_img = Image.open(original_img_path).convert("RGB")
    generated_img = generated_img.resize(original_img.size)
    w, h = original_img.size
    result_img = Image.new('RGB', (w * 2, h))
    result_img.paste(original_img, (0, 0))
    result_img.paste(generated_img, (w, 0))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{img_id}_2up.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)