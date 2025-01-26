import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from torchvision.transforms import functional
sys.modules["torchvision.transforms.functional_tensor"] = functional

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

import torch
from torch.cuda.amp import autocast
import cv2
from PIL import Image
import numpy as np
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DEISMultistepScheduler

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)


# Load the model
controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16)
pipe_obrem = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)
pipe_obrem.scheduler = DEISMultistepScheduler.from_config(pipe_obrem.scheduler.config)
pipe_obrem.enable_xformers_memory_efficient_attention()
pipe_obrem.to('cuda')

def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)


def upscaler(img):
    try:
        if not isinstance(img, np.ndarray):
            raise ValueError("Input image is not a valid NumPy array.")
        
        model_path = 'image_upscaler/realesr-general-x4v3.pth'
        half = True if torch.cuda.is_available() else False

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

        img = img.astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        try:
            face_enhancer = GFPGANer(
                model_path='image_upscaler/GFPGANv1.4.pth',
                upscale=10,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )
        except Exception as e:
            print("Error initializing GFPGANer:", e)
            return {"error": f"Error initializing GFPGANer: {str(error)}"}

        try:
            with torch.no_grad():
                with autocast():
                     enhance_result = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                     if isinstance(enhance_result, tuple) and len(enhance_result) == 3:
                          _, _, output = enhance_result
                     else:
                        raise ValueError("Unexpected output from face_enhancer.enhance")
        except RuntimeError as error:
            print('Error during enhancement:', error)
        except Exception as error:
            print("Unexpected error during enhancement:", error)

        if output is None:
            raise ValueError("Enhancement failed, output is None.")

        interpolation = cv2.INTER_LANCZOS4
        h, w = img.shape[0:2]
        output = cv2.resize(output, (int(w * 10 / 2), int(h * 10 / 2)), interpolation=interpolation)
        if len(output.shape) == 3 and output.shape[2] == 3:
            pass  
        else:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output 
    
    except Exception as error:
        print('Global exception:', error)
        return {"error": f"Global exception: {str(error)}"}
        

def inference_obrem(input_dict):
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
    if blended_image.mode == 'RGBA':
        blended_image = blended_image.convert('RGB')

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
                prompt='',  
                num_inference_steps=20,
                generator=generator,
                image=blended_image_np,
                control_image=canny_image,
                controlnet_conditioning_scale=0.9,
                mask_image=input_image
            ).images[0]
            
    upscaled_image = upscaler(np.array(output))

    if len(upscaled_image.shape) == 3 and upscaled_image.shape[2] == 3:
        upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB)

    return upscaled_image
