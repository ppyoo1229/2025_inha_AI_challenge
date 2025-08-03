

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
    def get_enhancement_keywords(self, cleaned_caption):
        return [self.fixed_tail_template]  

    def get_base_negative_prompt(self, cleaned_caption=None):
        return self.base_negative_prompts