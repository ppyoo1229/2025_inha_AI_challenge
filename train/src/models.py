'''
UNet, VAE, ControlNet, TextEncoder, etc. 모델 load 함수
LoRA 어댑터/병합, 가중치 저장/불러오기
'''

def save_lora_model_overwriting(model_dict, dir_path, subfolder_unet="unet_lora", is_main_process=True):
    if is_main_process:
        abs_dir_path = os.path.abspath(dir_path)

        unet_lora_path = os.path.join(abs_dir_path, subfolder_unet)
        if os.path.exists(unet_lora_path):
            shutil.rmtree(unet_lora_path)
        os.makedirs(unet_lora_path, exist_ok=True)

        if isinstance(model_dict['unet'], PeftModel):
            model_dict['unet'].save_pretrained(unet_lora_path)
            print(f"UNet LoRA saved to {unet_lora_path}")
        else:
            print(f"Warning: model_dict['unet'] is not a PeftModel. Skipping UNet LoRA saving.")

def get_peft_leaf_model(m):
    if hasattr(m, "base_model") and isinstance(m.base_model, torch.nn.Module):
        return get_peft_leaf_model(m.base_model)
    return m

def merge_adapter_weights()