from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import random
import imageio

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetPipeline, AutoencoderKL, \
    DiffusionPipeline
from diffusers.utils import load_image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux.processor import Processor



class sdxl_model:
    def __init__(self, theme, control_type="inpaint", safety_checker=False):
        theme_cfg = theme_to_model_map[theme]
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(theme_cfg["vae"], torch_dtype=torch.float16)
        self.pipe = None
        self.refiner = None
        self.processor = None
        # Base pipe
        if(control_type=="inpaint"):
            controlnet = ControlNetModel.from_pretrained(model_name_mapper[control_type][1], torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(theme_cfg["base_model"],
                controlnet=controlnet, vae=self.vae, torch_dtype=torch.float16)
            self.processor = Processor(ctrl_type_to_processor_id[control_type])
        elif(control_type):
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(theme_cfg["base_model"], 
                    torch_dtype=torch.float16, vae=vae, use_safetensors=True)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(theme_cfg["base_model"], 
                    torch_dtype=torch.float16, use_safetensors=True)
        
        self.pipe = load_lora_weight(pipe, theme_cfg["lora_list"])
        self.pipe.enable_model_cpu_offload()

        # Refiner pipe
        if(theme_cfg["use_refiner"] and is_inpaint):
            self.refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, 
                torch_dtype=torch.float16, use_safetensors=True)
            self.refiner.enable_model_cpu_offload()
        elif(theme_cfg["use_refiner"]):
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, 
                torch_dtype=torch.float16, use_safetensors=True)
            self.refiner.enable_model_cpu_offload()

        # base_model, control_type=None, lora_list=[], ti_list=[], n_prompt='', num_inf_steps=30,
        # vae_path=None, clip_skip=0, safety_checker=False
        self.control_type = control_type
        self.theme = theme
        self.num_inf_steps = 75
        return
    
    def gen_img(self, prompt, op_fld, ctrl_img_path, ctrl_img, n_prompt="", height=1024, width=1024, seed=-1, num_images=1):
        # print("Min and max ", np.max(ctrl_img))
        generator = seed_to_generator(seed)

        if(self.control_type == "inpaint"):
            return gen_inpaint_img()

        if(self.processor):
            image = self.gen_ctrl_img()
        else:
            image = pipe(
                prompt, prompt_2=prompt, num_inference_steps=num_inf_steps, denoising_end=high_noise_frac,
                generator=generator, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
                height=height, width=width, output_type="latent" if self.refiner else "pil"
            ).images

        if(self.refiner):
            image = refiner(prompt=prompt, prompt_2=prompt, image=image, num_inference_steps=num_inf_steps,
                denoising_start=high_noise_frac, 
                negative_prompt=n_prompt, negative_prompt_2=n_prompt,
                generator=generator).images

            
    def gen_inpaint_img():
        image = self.pipe(prompt=prompt, prompt_2=prompt, image=init_image, mask_image=mask_image,
            num_inference_steps=num_inf_base_steps, denoising_end=denoising_end,
            generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
            height=height, width=width, output_type="latent" if self.refiner else "pil",).images
        if(self.refiner):
            image = self.refiner(prompt=prompt, prompt_2=prompt, image=image, mask_image=mask_image,
                num_inference_steps=num_inf_refiner_steps, denoising_start=denoising_start,
                generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
                height=height, width=width).images
    
    def gen_ctrl_img():
        if(ctrl_img_path):
            ctrl_img = Image.open(ctrl_img_path).convert("RGB")#.resize((512, 512))
        else:
            ctrl_img = Image.fromarray(ctrl_img)
        ctrl_img = processor(ctrl_img, to_pil=True)
        # generate image
        image = self.pipe(prompt, prompt_2=prompt_2, controlnet_conditioning_scale=model_name_mapper[control_type][0], image=ctrl_img,
            num_inference_steps=num_inf_steps, generator=generator, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
            height=height, width=width, output_type= "latent" if self.refiner else "pil"
        ).images
        return
    

        

def gen_one_img(state, prompt, op_fld, prompt_2, ctrl_img_path, ctrl_img, op_fld, control_type="canny",
    n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inf_steps=75):

    pipe, refiner, processor = state
    

    # pipe_list.append(pipe)        
    
    # image = refiner(prompt=prompt, prompt_2=prompt, image=image, num_inference_steps=num_inf_steps, 
    #     negative_prompt=n_prompt, negative_prompt_2=n_prompt,generator=generator).images

    
def gen_sdxl_img(prompt, op_fld, lora_path=None, lora_scale = 0.8, n_prompt="", 
    height=1024, width=1024, seed=-1, use_refiner=True,
    is_eed=False, high_noise_frac = 0.7, num_images=1, num_inf_steps=75,):

    if(not use_refiner):
        is_eed=False

    pipe = create_base_model(lora_path=lora_path, lora_scale = lora_scale, is_inpaint=False)
    generator = seed_to_generator(seed)

    if(not is_eed):
        high_noise_frac = 1.0
    
    image = pipe(
        prompt, prompt_2=prompt, num_inference_steps=num_inf_steps, denoising_end=high_noise_frac,
        generator=generator, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
        height=height, width=width, output_type="latent" if use_refiner else "pil"
    ).images

    if(use_refiner):
        if(not is_eed):
            num_inf_steps = int(num_inf_steps*0.25)
            high_noise_frac = 0.0
        # num_inf_steps = 25 
        
        
    os.makedirs(op_fld, exist_ok=True)
    op_name = os.path.join(op_fld, "generated.png")
    image[0].save(op_name)
    return [{'seed':seed, 'path':op_name}]


def gen_sdxl_controlnet(prompt, ctrl_img_path, op_fld, control_type="canny", lora_path=None, lora_scale = 0.8,
    n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inf_steps=75, use_refiner=True,
    safety_checker=None, collect_cache=True): 

    pipe, processor = create_controlnet(control_type="canny", lora_path=None, lora_scale=0.8)
    gen_one_img(pipe, refiner, processor, prompt, prompt_2, ctrl_img_path, ctrl_img, op_fld, control_type="canny",
        n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inf_steps=75)


    if(use_refiner):
        # num_inf_steps = 25 
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, 
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16",)
        refiner.enable_model_cpu_offload()
        image = refiner(prompt=prompt, prompt_2=prompt, 
            image=image, num_inference_steps=num_inf_steps, 
            negative_prompt=n_prompt, negative_prompt_2=n_prompt,
            generator=generator).images

    os.makedirs(op_fld, exist_ok=True)
    op_name = os.path.join(op_fld, "generated.png")
    image[0].save(op_name)
    return [{'seed':seed, 'path':op_name}]

# The first stage outputs a latents and then the refiner takes the latents and produces the output
def sdxl_inpaint(theme, prompt, op_fld, lora_path=None, lora_scale = 0.8,
    n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inference_steps=75, use_refiner=True,
    is_eed=True, high_noise_frac = 0.7, safety_checker=None, collect_cache=True):#prompt, neg_prompt, img_path, mask_path, vae_path=None,):

    if(not use_refiner):
        is_eed = False

    pipe = create_base_model(lora_path=lora_path, lora_scale = lora_scale, is_inpaint=True)
    # generate image
    generator = seed_to_generator(seed)

    if(is_eed):
        num_inf_base_steps = 75
        denoising_end = 0.7
        num_inf_refiner_steps = 75
        denoising_start = 0.7
    else:
        num_inf_base_steps = 75
        denoising_end = 1.0
        num_inf_refiner_steps = 25
        denoising_start = 0.0

    image = pipe(prompt=prompt, prompt_2=prompt, image=init_image, mask_image=mask_image,
        num_inference_steps=num_inf_base_steps, denoising_end=denoising_end,
        generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
        height=height, width=width, output_type="latent" if use_refiner else "pil",).images
    if(use_refiner):
        refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        
        

    os.makedirs(op_fld, exist_ok=True)
    op_name = os.path.join(op_fld, "generated.png")
    image.save(op_name)
    return [{'seed':seed, 'path':op_name}]


