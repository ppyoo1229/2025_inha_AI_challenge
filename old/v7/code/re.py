from diffusers import UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline

from peft import PeftModel, merge_adapter_weights



# 1. UNet base + LoRA merge

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

peft_unet = PeftModel.from_pretrained(unet, "output5/best_model/unet_lora")

peft_unet = merge_adapter_weights(peft_unet)

merged_unet = peft_unet.base_model



# 2. ControlNet: pretrained만 로드

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")



# 3. 나머지 파이프라인 컴포넌트 (vae, text_encoder 등) 원본으로

pipe = StableDiffusionControlNetPipeline(
    unet=merged_unet,
    controlnet=controlnet,
    vae=AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae"),
    text_encoder=CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder"),
    tokenizer=CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer"),
    scheduler=UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler"),
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
pipe.to("cuda")