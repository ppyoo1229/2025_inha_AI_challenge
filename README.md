# 🎨 Language-Guided Image Colorization (2025 인하 인공지능 챌린지)

> 흑백 이미지를 **자연어 지시**로 색채화하는 멀티모달 모델  
> Stable Diffusion + ControlNet(Canny) + ResUNet + LoRA + BLIP-2(텍스트 임베딩) + CLIPScore 후처리

<p align="center">
  <a href="https://dacon.io/competitions/official/236499/overview/description">대회 바로가기</a>
</p>

---

## 📌 대회 정보

**2025 인하 인공지능 챌린지 – 언어 정보 기반 이미지 색채화** 대회 기록물들 입니다. 
텍스트와 흑백 이미지를 결합하여, 원본 구조를 보존하면서 **의미 기반 색감**을 복원하는 것을 목표로 합니다.

---

## 🧮 평가

- **HSV_Similarity**: 실제 vs 생성 이미지의 **HSV 히스토그램 유사도 평균**
- **CLIP_Score**: 프롬프트 vs 생성 이미지의 **코사인 유사도** (ViT-L/14 기준, 베이스라인 참조)
- **리더보드**
  - Public: 테스트 30%  
  - Private: 테스트 100% (최종 순위 산정)

규칙 전문: https://dacon.io/competitions/official/236499/overview/rules

---
## ⚙️ 환경

- Python ≥ 3.10
- PyTorch, diffusers, transformers, peft, accelerate  
- open-clip-torch, lpips, pytorch-msssim, opencv-python, scikit-image, safetensors 등
- 학습:
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install diffusers==0.27.2
pip install transformers==4.36.2
pip install accelerate==0.29.2
pip install peft==0.7.1
pip install xformers==0.0.25
pip install lpips
pip install torchmetrics
pip install scikit-image pandas matplotlib tqdm pillow opencv-python
pip install nltk
pip install tensorboard
pip install open_clip_torch==3.0.0
pip install pytorch-msssim

- 추론:
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install numpy==1.26.4
pip install diffusers==0.27.2
pip install transformers==4.36.2
pip install huggingface-hub==0.20.2

---

## 🏗️ 방법(Our Approach)

### 전체 파이프라인
