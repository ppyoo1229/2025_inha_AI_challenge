from diffusers import ControlNetModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

for name, module in controlnet.named_modules():
    print(name)