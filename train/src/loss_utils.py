
def get_clip_features(image_tensor, clip_processor, clip_model, accelerator_device, weight_dtype):
    pil_list = tensor_to_pil(image_tensor)
    if not isinstance(pil_list, list):
        pil_list = [pil_list]

    inputs = clip_processor(images=pil_list, return_tensors="pt")
    inputs = {k: v.to(accelerator_device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=weight_dtype)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)

def ssim_loss(img1, img2, data_range=2.0, size_average=True):
    return 1 - msssim(img1.float(), img2.float(), data_range=data_range, size_average=size_average)

def l1_loss(pred, target):
    """
    L1 Loss (Mean Absolute Error) -> 함수 없이 코드로 호출함...
    l1_loss = F.l1_loss(generated_image, gt_pixel_values, reduction='mean')
    """
    return F.l1_loss(pred, target, reduction='mean')

def clip_loss(features_generated, features_gt):
    """
    clip_features_generated = get_clip_features(generated_image, clip_processor, clip_model, accelerator.device, weight_dtype)
    clip_features_gt = get_clip_features(gt_pixel_values, clip_processor, clip_model, accelerator.device, weight_dtype)
    clip_loss = 1 - F.cosine_similarity(clip_features_generated, clip_features_gt, dim=-1).mean()
    """
    return 1 - F.cosine_similarity(features_generated, features_gt, dim=-1).mean()

def lpips_loss(pred, target, lpips_model):
    """
    LPIPS perceptual loss -> 함수 없이 코드로 호출함...
    lpips_loss_train = torch.tensor(0.0, device=accelerator.device) # (실제 학습 시 비활성화)
    검증(run_validation):
    lpips_loss = lpips_loss_fn(
        ((images + 1) / 2.0),     # -1~1 → 0~1
        ((gt_rgb_tensors + 1) / 2.0)
    ).mean()
    """
    return lpips_model(
        ((pred + 1) / 2.0),    # -1~1 -> 0~1
        ((target + 1) / 2.0)
    ).mean()

def diffusion_loss_fn(pred, target, reduction='mean'): 
    """
    Diffusion loss (MSE Loss) -> 함수 없이 코드로 호출함...
    loss_diffusion = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    """
    return F.mse_loss(pred.float(), target.float(), reduction=reduction)