# ColorizationDataset,

# collate_fn,

# worker_init_fn

# --- ColorizationDataset ---
class ColorizationDataset(Dataset):
    def __init__(self, df, input_dir, gt_dir, transform, tokenizer, enhancer, img_size=512):
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.enhancer = enhancer
        self.img_size = img_size
        self.max_tokens = CFG.MAX_PROMPT_TOKENS
        self.nsfw_keywords = [k.lower() for k in CFG.NSFW_KEYWORDS]
        self.sfw_caption_replacement = CFG.SFW_CAPTION_REPLACEMENT
        self.printed_count = 0 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cleaned_input_path_from_csv = os.path.normpath(row['input_img_path'])
        cleaned_gt_path_from_csv = os.path.normpath(row['gt_img_path'])
        
        input_image_path = os.path.join(self.input_dir, cleaned_input_path_from_csv)
        gt_image_path = os.path.join(self.gt_dir, cleaned_gt_path_from_csv)
        original_input_pil = Image.open(input_image_path).convert("RGB")
        input_image_np = np.array(original_input_pil)
        gray_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)
       
        raw_caption = str(row['caption'])
        cleaned_caption_raw = simple_caption_clean(raw_caption, number_words, number_regex)

        is_nsfw = any(nsfw_kw in cleaned_caption_raw for nsfw_kw in self.nsfw_keywords)
        if is_nsfw:
            cleaned_caption = self.sfw_caption_replacement
        else:
            cleaned_caption = cleaned_caption_raw

        # dominant color name 추출
        dominant_colors = extract_dominant_colors(original_input_pil, topk=3)
        color_names = [rgb_to_simple_color_name(c) for c in dominant_colors]
        color_names = list(dict.fromkeys(color_names))[:3]
        color_str = ', '.join(color_names)
        # 프롬프트에 dominant color name 삽입
        pos_prompt_parts = [color_str, cleaned_caption]  # 색상명을 맨 앞에 추가
        enhancement_keywords_list = self.enhancer.get_enhancement_keywords(cleaned_caption)
        for keyword_phrase in enhancement_keywords_list:
            temp_prompt = ", ".join(pos_prompt_parts + [keyword_phrase])
            temp_token_ids = self.tokenizer.encode(
                temp_prompt,
                add_special_tokens=True,
                truncation=True,
                return_tensors="pt"
            )[0]
            if len(temp_token_ids) <= self.max_tokens:
                pos_prompt_parts.append(keyword_phrase)
            else:
                break
        pos_prompt_str_raw = ", ".join(pos_prompt_parts)
        final_pos_prompt_str_for_pipe = safe_prompt_str(pos_prompt_str_raw, self.tokenizer, self.max_tokens)

        if self.printed_count < 2:
            print(f"[프롬프트 샘플 {self.printed_count+1}] dominant colors: {color_names} | caption: {cleaned_caption}")
            print(f"[프롬프트 전체] {final_pos_prompt_str_for_pipe}\n")
            self.printed_count += 1

        pos_tokenized_output = self.tokenizer(
            final_pos_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        final_pos_input_ids = pos_tokenized_output.input_ids[0]

        base_neg_prompt_str = self.enhancer.get_base_negative_prompt(cleaned_caption)
        final_neg_prompt_str_for_pipe = safe_prompt_str(base_neg_prompt_str, self.tokenizer, self.max_tokens)
        neg_tokenized_output = self.tokenizer(
            final_neg_prompt_str_for_pipe,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        final_neg_input_ids = neg_tokenized_output.input_ids[0]

        # --- Canny 이미지 생성 및 정규화 ---
        canny_low, canny_high = 50, 150
        canny_image_np = cv2.Canny(gray_image_np, canny_low, canny_high)
        canny_image_pil = Image.fromarray(canny_image_np).convert("RGB")
        input_control_image = self.transform(canny_image_pil)

        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        gt_rgb_tensor = self.transform(gt_image_pil)

        guidance = CFG.GUIDANCE_SCALE
        steps = CFG.NUM_INFERENCE_STEPS
        return {
            "conditioning_pixel_values": input_control_image,
            "gt_rgb_tensor": gt_rgb_tensor,
            "caption": raw_caption,
            "cleaned_caption_raw": cleaned_caption,
            "pos_prompt_input_ids": final_pos_input_ids,
            "neg_prompt_input_ids": final_neg_input_ids,
            "pos_prompt_str_for_pipe": final_pos_prompt_str_for_pipe,
            "neg_prompt_str_for_pipe": final_neg_prompt_str_for_pipe,
            "guidance": guidance,
            "steps": steps,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "file_name": os.path.basename(cleaned_input_path_from_csv)
        }
    
def collate_fn(examples):
    pixel_values = torch.stack([example["gt_rgb_tensor"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    pos_prompt_input_ids = torch.stack([example["pos_prompt_input_ids"] for example in examples])
    neg_prompt_input_ids = torch.stack([example["neg_prompt_input_ids"] for example in examples])

    pos_prompt_str_for_pipe = [str(example["pos_prompt_str_for_pipe"]) for example in examples]
    neg_prompt_str_for_pipe = [str(example["neg_prompt_str_for_pipe"]) for example in examples]

    guidance_scales = torch.tensor([example["guidance"] for example in examples])
    num_inference_steps = examples[0]["steps"]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "pos_prompt_input_ids": pos_prompt_input_ids,
        "neg_prompt_input_ids": neg_prompt_input_ids,
        "pos_prompt_str_for_pipe": pos_prompt_str_for_pipe,
        "neg_prompt_str_for_pipe": neg_prompt_str_for_pipe,
        "guidance_scales": guidance_scales,
        "num_inference_steps": num_inference_steps,
        "captions": [example["caption"] for example in examples],
        "file_names": [example["file_name"] for example in examples],
    }

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    set_seed(worker_seed)