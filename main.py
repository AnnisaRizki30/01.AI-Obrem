import torch
from torch.cuda.amp import autocast
import cv2
from PIL import Image
import numpy as np
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DEISMultistepScheduler


# Load the model
controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16)
pipe_obrem = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)
pipe_obrem.scheduler = DEISMultistepScheduler.from_config(pipe_obrem.scheduler.config)
pipe_obrem.enable_xformers_memory_efficient_attention()
pipe_obrem.to('cuda')

def resize_image_obrem(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)

def inference_obrem(input_dict, prompt):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    image = input_dict["image"].convert("RGB")
    input_image = input_dict["mask"].convert("RGB")
    input_image = resize_image(input_image, 768)
    image = resize_image(image, 768)

    image_np = np.array(image)
    input_image_np = np.array(input_image)

    mask_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY) / 255.0

    inpainted_image_np = cv2.inpaint(image_np, (mask_np * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    blended_image_np = image_np * (1 - mask_np)[:, :, None] + inpainted_image_np * mask_np[:, :, None]

    blended_image = Image.fromarray(np.uint8(blended_image_np))

    blended_image_np = np.array(blended_image)
    low_threshold = 800
    high_threshold = 900
    canny = cv2.Canny(blended_image_np, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)

    generator = torch.manual_seed(0)
    with torch.no_grad():
        with torch.cuda.amp.autocast(): 
            output = pipe_obrem(
                prompt=prompt,  
                num_inference_steps=50,
                generator=generator,
                image=blended_image_np,
                control_image=canny_image,
                controlnet_conditioning_scale=0.9,
                mask_image=input_image
            ).images[0]
    
    return output
