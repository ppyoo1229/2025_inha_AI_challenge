# main_utils.py

# ---- 1) Checkpoint 및 LoRA 저장 ----
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

# --- 2) save_latest_checkpoint 함수 추가 ---
def save_latest_checkpoint(unet, output_dir, global_step, accelerator, cfg):
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        # save_lora_model_overwriting 함수를 사용하여 UNet LoRA 모델 저장
        save_lora_model_overwriting({"unet": unet}, ckpt_dir, is_main_process=True)
        # 상태 (옵티마이저, 스케줄러 등) 저장
        accelerator 
        accelerator.save_state(ckpt_dir)
        # global_step 저장
        torch.save(global_step, os.path.join(ckpt_dir, "global_step.pt"))
        print(f"Checkpoint saved to {ckpt_dir} at step {global_step}")
        # 오래된 체크포인트 삭제 
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))], 
                         key=lambda x: int(x.split('-')[-1]))
        if len(ckpts) > cfg.MAX_CHECKPOINTS_TO_KEEP:
            for i in range(len(ckpts) - cfg.MAX_CHECKPOINTS_TO_KEEP):
                old_ckpt_path = os.path.join(output_dir, ckpts[i])
                print(f"Removing old checkpoint: {old_ckpt_path}")
                shutil.rmtree(old_ckpt_path, ignore_errors=True)

# --- 3) 검증 함수 ---
@torch.no_grad()
def run_validation(pipeline,
                   accelerator,
                   epoch,
                   global_step,
                   val_dataloader,
                   clip_processor,
                   clip_model,
                   weight_dtype,
                   output_dir,
                   num_samples_to_save
                   ):
    global lpips_loss_fn  

    if lpips_loss_fn is None:
        try:
            lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
            lpips_loss_fn.eval() # LPIPS - 학습에서 X
            for param in lpips_loss_fn.parameters(): # 파라미터 업데이트 방지
                param.requires_grad = False
            print("LPIPS model initialized and moved to device.")
        except ImportError:
            print("Warning: LPIPS library not found. LPIPS loss will be skipped.")
            lpips_loss_fn = "skipped"
    print("\n---Running validation---")
    pipeline.unet.eval()
    pipeline.controlnet.eval()

    val_output_dir = os.path.join(output_dir, "validation_samples", f"step_{global_step}")
    os.makedirs(val_output_dir, exist_ok=True)

    total_l1_loss = 0.0
    total_clip_loss = 0.0
    total_lpips_loss = 0.0
    total_ssim_loss = 0.0
    # Diffusion Loss는 run_validation에서 파이프라인으로 생성된 최종 이미지에 대한 손실이 아니므로 여기서는 계산하지 않음
    num_processed_samples = 0

    batches_to_process = 0
    if num_samples_to_save is not None:
        batches_to_process = math.ceil(num_samples_to_save / CFG.BATCH_SIZE)
    for i, batch in enumerate(val_dataloader):
        if num_samples_to_save is not None and i >= batches_to_process:
            break

        conditioning_images_pil_list = tensor_to_pil(batch["conditioning_pixel_values"])
        gt_rgb_tensors = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
        pos_prompt_strs = batch["pos_prompt_str_for_pipe"]
        neg_prompt_strs = batch["neg_prompt_str_for_pipe"]
        num_inference_steps = batch["num_inference_steps"]
        file_names = batch["file_names"]
        captions = batch["captions"]
        current_batch_size = gt_rgb_tensors.shape[0]
        print(f"\n--- Validation Batch {i+1} (Size: {current_batch_size}) ---")
        
        # pipeline을 통해 이미지 생성
        images = pipeline(
            image=conditioning_images_pil_list,
            prompt=pos_prompt_strs,
            negative_prompt=neg_prompt_strs,
            guidance_scale=CFG.GUIDANCE_SCALE,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=CFG.CONTROLNET_STRENGTH,
            output_type="pt", # 텐서 출력을 위해 "pt" 명시
        ).images.to(accelerator.device)

        # Losses 계산 (Diffusion Loss 제외)
        l1_loss = F.l1_loss(images, gt_rgb_tensors, reduction='mean')
        total_l1_loss += l1_loss.item() * current_batch_size
        clip_features_generated = get_clip_features(images, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_features_gt = get_clip_features(gt_rgb_tensors, clip_processor, clip_model, accelerator.device, weight_dtype)
        clip_loss = 1 - F.cosine_similarity(clip_features_generated, clip_features_gt, dim=-1).mean()
        total_clip_loss += clip_loss.item() * current_batch_size
        if lpips_loss_fn != "skipped":
            lpips_loss = lpips_loss_fn(
                ((images + 1) / 2.0).to(accelerator.device), # -1~1 -> 0~1
                ((gt_rgb_tensors + 1) / 2.0).to(accelerator.device) # -1~1 -> 0~1
            ).mean()
            total_lpips_loss += lpips_loss.item() * current_batch_size
        else:
            lpips_loss = torch.tensor(0.0)

        ssim_val = ssim_loss(images, gt_rgb_tensors, data_range=2.0, size_average=True)
        total_ssim_loss += ssim_val.item() * current_batch_size
        num_samples_to_save = 3  # 대표 저장 개수(컨피그에서 받아도 됨)
        save_count = 0           # 저장한 샘플 수
        for j in range(current_batch_size):
            if save_count < num_samples_to_save:
                generated_img_pil = tensor_to_pil(images[j:j+1])
                gt_img_pil = tensor_to_pil(gt_rgb_tensors[j:j+1])
                canny_img_pil = conditioning_images_pil_list[j]

                combined_width = canny_img_pil.width + generated_img_pil.width + gt_img_pil.width
                combined_height = canny_img_pil.height
                combined_img = Image.new('RGB', (combined_width, combined_height))

                combined_img.paste(canny_img_pil, (0, 0))
                combined_img.paste(generated_img_pil, (canny_img_pil.width, 0))
                combined_img.paste(gt_img_pil, (canny_img_pil.width + generated_img_pil.width, 0))

                save_path = os.path.join(val_output_dir, f"step_{global_step}_sample_{num_processed_samples + j}_{file_names[j]}.png")
                combined_img.save(save_path)
                save_count += 1
            # else: 저장 안 함
        num_processed_samples += current_batch_size

    if num_processed_samples > 0:
        avg_l1_loss = total_l1_loss / num_processed_samples
        avg_clip_loss = total_clip_loss / num_processed_samples
        avg_lpips_loss = total_lpips_loss / num_processed_samples
        avg_ssim_loss = total_ssim_loss / num_processed_samples
    else:
        avg_l1_loss = avg_clip_loss = avg_lpips_loss = avg_ssim_loss = 0.0

    log_message = (
        f"Validation Results (Epoch {epoch}, Global Step {global_step}):\n"
        f"   Average L1 Loss: {avg_l1_loss:.4f}\n"
        f"   Average CLIP Loss: {avg_clip_loss:.4f}\n"
        f"   Average LPIPS Loss: {avg_lpips_loss:.4f}\n"
        f"   Average SSIM Loss: {avg_ssim_loss:.4f}"
    )
    print(log_message)

    avg_combined_val_loss = (CFG.LAMBDA_L1 * avg_l1_loss +
                             CFG.LAMBDA_CLIP * avg_clip_loss +
                             CFG.LAMBDA_LPIPS * avg_lpips_loss +
                             CFG.LAMBDA_SSIM * avg_ssim_loss)

    accelerator.log({
        "val_avg_l1_loss": avg_l1_loss,
        "val_avg_clip_loss": avg_clip_loss,
        "val_avg_lpips_loss": avg_lpips_loss,
        "val_avg_ssim_loss": avg_ssim_loss,
        "val_avg_combined_loss": avg_combined_val_loss,
    }, step=global_step)

    pipeline.unet.train()
    pipeline.controlnet.train()

    return avg_combined_val_loss

# --- 4) Main Training Loop ---
def train_loop(
    pretrained_model_name_or_path: str,
    controlnet_path: str,
    output_dir: str,
    train_data_df: pd.DataFrame,
    cfg: Config,
):
    global lpips_loss_fn # LPIPS loss 전역 변수 선언
    lpips_loss_fn = None

    # 1. 분산 학습, mixed precision 등 accelerator 초기화
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=cfg.MIXED_PRECISION,
        log_with=cfg.REPORT_TO,
        project_dir=os.path.join(output_dir, cfg.PROJECT_NAME),
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers(cfg.PROJECT_NAME, config=filter_config_types(vars(cfg)))
    
    # 2. 모델/토크나이저/프로세서 로드
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_path)
    clip_processor = CLIPProcessor.from_pretrained(cfg.CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(cfg.CLIP_MODEL)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # 3. LPIPS 손실 함수 준비 (최초 1회만)
    if lpips_loss_fn is None: 
        try:
            lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
            lpips_loss_fn.eval()
            for param in lpips_loss_fn.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: LPIPS library not found. LPIPS loss will be skipped.")
            lpips_loss_fn = "skipped"

    # 4. mixed precision 타입 지정
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 5. 파인튜닝하지 않을 모듈 freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False) # ControlNet은 Fine-tuning에서 고정

    # 6. LoRA Config로 UNet에 LoRA 적용
    unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters() # trainable 파라미터만 출력

    # 7. 옵티마이저/스케줄러 세팅
    params_to_optimize = list(unet.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=cfg.LR,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        weight_decay=cfg.ADAM_WEIGHT_DECAY,
        eps=cfg.ADAM_EPSILON,
    )
    if cfg.MAX_TRAIN_STEPS is None:
        cfg.MAX_TRAIN_STEPS = cfg.EPOCHS * (len(train_data_df) // cfg.BATCH_SIZE)
    lr_scheduler = get_scheduler(
        cfg.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=cfg.LR_WARMUP_STEPS * cfg.GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=cfg.MAX_TRAIN_STEPS,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # 8. 데이터셋/로더 준비
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_df, val_df = train_test_split(train_data_df, test_size=0.1, random_state=cfg.SEED)
    # MAX_DATA 처리
    if cfg.MAX_DATA is not None:
        train_df = train_df.head(min(cfg.MAX_DATA, len(train_df)))
        val_df = val_df.head(min(cfg.MAX_DATA // 10 if cfg.MAX_DATA // 10 > 0 else 1, len(val_df)))
    else:
        # MAX_DATA 미설정 시, val_df는 300장만 샘플링
        val_df = val_df.sample(n=300, random_state=cfg.SEED).reset_index(drop=True)
    
    enhancer = PromptEnhancer()
    train_dataset = ColorizationDataset(
        df=train_df, input_dir=cfg.INPUT_DIR, gt_dir=cfg.GT_DIR, transform=transform,
        tokenizer=tokenizer, enhancer=enhancer, img_size=cfg.IMG_SIZE
    )
    val_dataset = ColorizationDataset(
        df=val_df, input_dir=cfg.INPUT_DIR, gt_dir=cfg.GT_DIR, transform=transform,
        tokenizer=tokenizer, enhancer=enhancer, img_size=cfg.IMG_SIZE
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn, worker_init_fn=worker_init_fn, pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=True,
    )

     # 9. 분산/가속 환경에 모델, 옵티마이저, 로더 등록
    unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, val_dataloader
    )
    controlnet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    # 10. Diffusion 파이프라인 생성 (추론 및 검증에 사용)
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=get_peft_leaf_model(accelerator.unwrap_model(unet)) if isinstance(unet, PeftModel) else accelerator.unwrap_model(unet),
        controlnet=controlnet, scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.check_inputs = lambda *args, **kwargs: None # 입력 검증 건너뛰기

    # 11. 체크포인트, epoch, early stopping 등 학습 변수 초기화
    global_step = 0
    first_epoch = 0

    # 12. 체크포인트에서 학습 재개
    if cfg.RESUME_FROM_CHECKPOINT:
        if cfg.RESUME_FROM_CHECKPOINT != "latest":
            path = cfg.RESUME_FROM_CHECKPOINT
        else:
            all_checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if not all_checkpoints:
                raise ValueError("No checkpoints found to resume from 'latest'.")
            all_checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            path = os.path.join(output_dir, all_checkpoints[-1])
            print(f"Resuming from latest checkpoint: {path}")

        accelerator.load_state(path)
        global_step_path = os.path.join(path, "global_step.pt")
        if os.path.exists(global_step_path):
            global_step = torch.load(global_step_path)
        else:
            global_step = 0 # Default to 0 if not found

        first_epoch = global_step // len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Resumed training state from {path}, starting at global_step {global_step}, epoch {first_epoch}")

    total_batch_size = cfg.BATCH_SIZE * accelerator.num_processes * cfg.GRADIENT_ACCUMULATION_STEPS
    print("***** Running training *****")
    print(f"   Num examples = {len(train_dataset)}")
    print(f"   Num epochs = {cfg.EPOCHS}")
    print(f"   Instantaneous batch size per device = {cfg.BATCH_SIZE}")
    print(f"   Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"   Gradient Accumulation steps = {cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Total optimization steps = {cfg.MAX_TRAIN_STEPS}")

    progress_bar = tqdm(
        range(global_step, cfg.MAX_TRAIN_STEPS),
        disable=not accelerator.is_main_process,
        initial=global_step
    )
    progress_bar.set_description("Steps")
    best_combined_val_loss = float('inf')
    intervals_no_improve = 0

    # 13. 학습 메인 루프 (epoch, batch)
    for epoch in range(first_epoch, cfg.EPOCHS):
        unet.train()
        controlnet.eval() # ControlNet은 학습되지 않으므로 eval 모드 유지
        train_loss_this_interval = 0.0
        
        for step_in_epoch, batch in enumerate(train_dataloader):
            if global_step >= cfg.MAX_TRAIN_STEPS:
                break
            with accelerator.accumulate(unet):
                # CLIP text encoder (positive prompt)
                encoder_hidden_states_pos = text_encoder(batch["pos_prompt_input_ids"])[0]

                # Noise & Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # ControlNet Input (Canny image)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                
                # ControlNet forward pass (no_grad)
                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents, timesteps, encoder_hidden_states_pos,
                        controlnet_cond=controlnet_image, return_dict=False
                    )
                
                # UNet forward pass (LoRA 학습)
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states_pos,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample

                # Diffusion Loss (MSE Loss)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError("Unknown prediction type")
                
                loss_diffusion = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # VAE 디코딩 for L1/CLIP Loss (그래디언트 계산)
                # pred_original_sample은 중간 계산이므로 no_grad 유지, 하지만 vae.decode 결과는 그래디언트 연결
                with torch.no_grad():
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    sqrt_alpha_t = alpha_t.sqrt()
                    sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
                    if noise_scheduler.config.prediction_type == "epsilon":
                        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        pred_original_sample = noise_scheduler.get_velocity(noisy_latents, model_pred, timesteps)
                    else:
                        raise ValueError("Unknown prediction type")
                
                # VAE 디코딩 (VAE는 freeze 되어 있어도 Unet과의 연결은 유지)
                decoded_latents = 1 / vae.config.scaling_factor * pred_original_sample
                generated_image = vae.decode(decoded_latents.to(dtype=weight_dtype)).sample
                generated_image = generated_image.clamp(-1, 1) # -1 ~ 1 범위로 클램핑

                gt_pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                # L1 Loss 계산
                l1_loss = F.l1_loss(generated_image, gt_pixel_values, reduction='mean')
                
                # CLIP Loss 계산
                clip_features_generated = get_clip_features(generated_image, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_features_gt = get_clip_features(gt_pixel_values, clip_processor, clip_model, accelerator.device, weight_dtype)
                clip_loss = 1 - F.cosine_similarity(clip_features_generated, clip_features_gt, dim=-1).mean()

                lpips_loss_train = torch.tensor(0.0, device=accelerator.device)
                ssim_val_train = torch.tensor(0.0, device=accelerator.device)

                # --- total Loss ---- > 계속 바뀜
                total_loss = cfg.LAMBDA_L1 * l1_loss + \
                             loss_diffusion
                
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, cfg.MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # 15. Loss 로깅
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss_this_interval += total_loss.item()
                if global_step % cfg.LOG_INTERVAL == 0:
                    avg_train_loss = train_loss_this_interval / (cfg.LOG_INTERVAL * cfg.GRADIENT_ACCUMULATION_STEPS)
                    accelerator.log({
                        "train_loss": avg_train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "diffusion_loss": loss_diffusion.item(),
                        "l1_loss": l1_loss.item(),
                        "clip_loss": clip_loss.item(),
                        "lpips_loss": lpips_loss_train.item(), 
                        "ssim_loss": ssim_val_train.item(),     
                    }, step=global_step)
                    train_loss_this_interval = 0.0
            # 16. 검증 및 체크포인트 저장 (SAVE_AND_VAL_INTERVAL마다)
            if (
                global_step % cfg.SAVE_AND_VAL_INTERVAL == 0
                and global_step >= cfg.SAMPLE_SAVE_START_STEP
                and accelerator.is_main_process
            ):
                # 파이프라인의 구성 요소를 unwrap하여 최신 가중치 적용
                pipeline.unet = get_peft_leaf_model(accelerator.unwrap_model(unet))
                pipeline.controlnet = accelerator.unwrap_model(controlnet)
                pipeline.vae = accelerator.unwrap_model(vae)
                pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
                # Ensure pipeline is on the correct device with correct dtype before validation
                pipeline.to(accelerator.device, dtype=weight_dtype)
                current_combined_val_loss = run_validation(
                    pipeline, accelerator, epoch, global_step, val_dataloader,
                    clip_processor, clip_model, weight_dtype, output_dir, cfg.NUM_SAMPLES_TO_SAVE
                )
                accelerator.wait_for_everyone()

                if current_combined_val_loss < best_combined_val_loss:
                    best_combined_val_loss = current_combined_val_loss
                    intervals_no_improve = 0
                    print(f"New best validation combined loss: {best_combined_val_loss:.4f}. Saving best model.")
                    best_model_dir = os.path.join(output_dir, f"best_model_step{global_step}")
                    best_models = sorted(
                        [d for d in os.listdir(output_dir) if d.startswith("best_model_step")],
                        key=lambda x: int(x.split("step")[-1])
                    )
                    if len(best_models) > 5:
                        for d in best_models[:-5]:
                            shutil.rmtree(os.path.join(output_dir, d), ignore_errors=True)

                    if accelerator.is_main_process:
                        os.makedirs(best_model_dir, exist_ok=True)
                        save_lora_model_overwriting(
                            {"unet": accelerator.unwrap_model(unet)}, best_model_dir,
                            is_main_process=True
                        )
                        # torch.save(global_step, os.path.join(best_model_dir, "global_step.pt"))
                else:
                    intervals_no_improve += 1
                    print(f"Validation combined loss did not improve. Intervals without improvement: {intervals_no_improve}")
                    if intervals_no_improve >= cfg.PATIENCE:
                        print(f"Early stopping triggered after {cfg.PATIENCE} intervals.")
                        break
                save_latest_checkpoint(accelerator.unwrap_model(unet), output_dir, global_step, accelerator, cfg)
            
            # 17. 학습 조기 종료 조건(Early stopping 등)
            if global_step >= cfg.MAX_TRAIN_STEPS or intervals_no_improve >= cfg.PATIENCE:
                break

        if global_step >= cfg.MAX_TRAIN_STEPS or intervals_no_improve >= cfg.PATIENCE:
            break

    # 18. 학습 종료 후 최종 모델 저장
    if accelerator.is_main_process:
        print("Training finished. Saving final model.")
        save_lora_model_overwriting(
            {"unet": accelerator.unwrap_model(unet)}, os.path.join(output_dir, "final_model"),
            is_main_process=accelerator.is_main_process,
        )
    accelerator.end_training()

    # 메모리 해제
    del unet, controlnet, vae, text_encoder, tokenizer, optimizer, lr_scheduler
    del train_dataloader, train_dataset, val_dataset
    del clip_processor, clip_model
    if lpips_loss_fn != "skipped" and lpips_loss_fn is not None: 
        del lpips_loss_fn
    lpips_loss_fn = None 
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()