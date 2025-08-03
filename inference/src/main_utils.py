'''
파이프라인 로딩/ LoRA
'''

import os
import gc
import torch
from diffusers import UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from safetensors.torch import load_file
from transformers import CLIPTokenizer
import open_clip

# from src.model_utils import (
#     load_sd_controlnet_pipeline,
#     merge_lora_weights,
#     set_eval_mode,
#     load_tokenizer,
#     load_clip_model,
#     clear_cuda_cache
# )

# --- 1) UNet/ ControlNet/ SD+ControlNet 파이프라인 로드 ---
def load_sd_controlnet_pipeline(config):
    unet = UNet2DConditionModel.from_pretrained(
        config.MODEL_PATH,
        subfolder="unet",
        torch_dtype=config.WEIGHT_DTYPE
    )
    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_PATH,
        torch_dtype=config.WEIGHT_DTYPE
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.MODEL_PATH,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=config.WEIGHT_DTYPE,
        safety_checker=None,
    )

    # --- xFormers 등 메모리 최적화 옵션 ---
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
    pipe.check_inputs = lambda *args, **kwargs: None
    print("DEBUG: 파이프라인 생성 완료")
    return pipe

# --- 2) LoRA 어댑터 가중치 병합 ---
def merge_lora_weights(pipe, lora_weights_path):
    assert os.path.exists(lora_weights_path), f"LoRA weights file not found: {lora_weights_path}"
    pipe.load_lora_weights(
        lora_weights_path,
        adapter_name="default",
        force_merge=True
    )
    return pipe

# --- 3) 파이프라인 내 모든 모델을 eval() 모드 전환
def set_eval_mode(pipe):
    pipe.unet.eval()
    pipe.controlnet.eval()
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.eval()
    if hasattr(pipe, "vae"):
        pipe.vae.eval()

# --- 4) 토크나이저 로드 함수 ---
def load_tokenizer(model_path, subfolder="tokenizer"):
    return CLIPTokenizer.from_pretrained(model_path, subfolder=subfolder)

# --- 5) CLIP 임베딩 모델 및 preprocesser 로드 ---
def load_clip_model(embed_model, embed_pretrained, device):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        embed_model, pretrained=embed_pretrained)
    clip_model = clip_model.to(device).eval()
    return clip_model, clip_preprocess

# --- 6) 캐시 정리 ---
def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()