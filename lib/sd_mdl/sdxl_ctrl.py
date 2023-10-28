from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import random
import imageio

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetPipeline, \
    DiffusionPipeline, AutoPipelineForInpainting
from diffusers.utils import load_image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux.processor import Processor

from .sdxl_theme import theme_to_model_map, model_name_mapper, ctrl_type_to_processor_id
from .cmn import lora_base_path, load_vae, seed_to_generator

def load_lora_weight(pipe, lora_list):
    # if(lora_path):
    #     pipe = config_lora(pipe, lora_path, lora_scale)
    # alpha_wgt = 0.8
    for lora_name in lora_list:
        print("Lora ", lora_name)
        lora_path = os.path.join(lora_base_path, lora_name)
        pipe.load_lora_weights(lora_path)
    return pipe

def config_lora(pipe, lora_list, lora_scale):
    for lora_name in lora_list:
        lora_path = os.path.join(lora_base_path, lora_name)
        state_dict, network_alphas = pipe.lora_state_dict(lora_path, unet_config=pipe.unet.config,)
        pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet)
        pipe.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=pipe.text_encoder,
            lora_scale=lora_scale #self.lora_scale,
        )
    return pipe


class sdxl_model:
    def __init__(self, theme, control_type="inpaint", safety_checker=False):
        # print("Theme ", theme_to_model_map.keys())
        theme_cfg = theme_to_model_map[theme]
        
        # VAE
        self.vae = load_vae(theme_cfg["vae"])
        self.pipe = None
        self.refiner = None
        self.processor = None
        # Base pipe
        if(control_type=="inpaint"):
            # self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(theme_cfg["base_model"], 
            #         torch_dtype=torch.float16, vae=self.vae, use_safetensors=True)
            self.pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
                torch_dtype=torch.float16, variant="fp16")

        elif(control_type):
            controlnet = ControlNetModel.from_pretrained(model_name_mapper[control_type][1], torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(theme_cfg["base_model"],
                controlnet=controlnet, vae=self.vae, torch_dtype=torch.float16)
            self.processor = Processor(ctrl_type_to_processor_id[control_type])
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(theme_cfg["base_model"], 
                    torch_dtype=torch.float16, use_safetensors=True)
        
        self.pipe = config_lora(self.pipe, theme_cfg["lora_list"], 0.8) #load_lora_weight(self.pipe, theme_cfg["lora_list"])
        self.pipe.enable_model_cpu_offload()

        # Refiner pipe
        if(theme_cfg["use_refiner"] and control_type=="inpaint"):
            self.refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.pipe.text_encoder_2, vae=self.pipe.vae, 
                torch_dtype=torch.float16, use_safetensors=True)
            self.refiner.enable_model_cpu_offload()
        elif(theme_cfg["use_refiner"]):
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.pipe.text_encoder_2, vae=self.pipe.vae, 
                torch_dtype=torch.float16, use_safetensors=True)
            self.refiner.enable_model_cpu_offload()

        # base_model, control_type=None, lora_list=[], ti_list=[], n_prompt='', num_inf_steps=30,
        # vae_path=None, clip_skip=0, safety_checker=False
        self.control_type = control_type
        self.theme = theme
        self.num_inf_steps = 75
        self.high_noise_frac = 0.8
        return
    
    def gen_img(self, prompt, ctrl_img=None, n_prompt="", height=1024, width=1024, seed=-1, 
        num_images=1):
        # print("Min and max ", np.max(ctrl_img))
        seed, generator = seed_to_generator(seed)

        if(self.control_type == "inpaint"):
            mask_image = np.zeros_like(ctrl_img[0])
            mask_image[ctrl_img[0]==-1] = 1
            mask_image = Image.fromarray((255 * mask_image).astype(np.uint8))
            ctrl_img[0][ctrl_img[0]==-1] = 0
            print("Mask and image shape ", ctrl_img.shape)
            ctrl_img = Image.fromarray((255*ctrl_img[0]).astype(np.uint8))
            image = self.gen_inpaint_img(ctrl_img, mask_image, generator, prompt, n_prompt, height, width)
            return {"seed":seed, "image":image[0]}

        if(self.processor):
            ctrl_img = Image.fromarray(ctrl_img)
            image = self.gen_ctrl_img(ctrl_img, generator, prompt, n_prompt, height, width)
        else:
            image = self.pipe(prompt, prompt_2=prompt, num_inference_steps=self.num_inf_steps, 
                denoising_end=self.high_noise_frac, generator=generator, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
                height=height, width=width, output_type="latent" if self.refiner else "pil").images

        if(self.refiner):
            image = self.refiner(prompt=prompt, prompt_2=prompt, image=image, num_inference_steps=self.num_inf_steps,
                denoising_start=self.high_noise_frac, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
                generator=generator).images

        return {"seed":seed, "image":image[0]}
    def gen_inpaint_img(self, init_image, mask_image, generator, prompt, n_prompt, height, width):
        image = self.pipe(prompt=prompt, prompt_2=prompt, image=init_image, mask_image=mask_image,
            num_inference_steps=self.num_inf_steps, denoising_end=self.high_noise_frac,
            generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
            height=height, width=width, output_type="latent" if self.refiner else "pil",).images
        
        # steps between 15 and 30 work well for us # make sure to use `strength` below 1.0
        # image = self.pipe(prompt=prompt, prompt_2=prompt, image=init_image, mask_image=mask_image, guidance_scale=8.0,
        #     num_inference_steps=20, strength=0.99, generator=generator,
        #     negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
        #     height=height, width=width, output_type="latent" if self.refiner else "pil",).images
        if(self.refiner):
            image = self.refiner(prompt=prompt, prompt_2=prompt, image=image, mask_image=mask_image,
                num_inference_steps=self.num_inf_steps, denoising_start=self.high_noise_frac,
                generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
                height=height, width=width).images
        return image
    
    def gen_ctrl_img(self, ctrl_img, generator, prompt, n_prompt, height, width):
        
        ctrl_img = self.processor(ctrl_img, to_pil=True)
        # generate image
        image = self.pipe(prompt, prompt_2=prompt, controlnet_conditioning_scale=model_name_mapper[self.control_type][0], 
            image=ctrl_img, num_inference_steps=self.num_inf_steps, generator=generator, negative_prompt=n_prompt, 
            negative_prompt_2=n_prompt, height=height, width=width, output_type= "latent" if self.refiner else "pil"
        ).images
        return image


# def gen_sdxl_img(prompt, op_fld, lora_path=None, lora_scale = 0.8, n_prompt="", 
#     height=1024, width=1024, seed=-1, use_refiner=True,
#     is_eed=False, high_noise_frac = 0.7, num_images=1, num_inf_steps=75,):

#     if(not use_refiner):
#         is_eed=False

#     pipe = create_base_model(lora_path=lora_path, lora_scale = lora_scale, is_inpaint=False)
#     generator = seed_to_generator(seed)

#     if(not is_eed):
#         high_noise_frac = 1.0
    
#     image = pipe(
#         prompt, prompt_2=prompt, num_inference_steps=num_inf_steps, denoising_end=high_noise_frac,
#         generator=generator, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
#         height=height, width=width, output_type="latent" if use_refiner else "pil"
#     ).images

#     if(use_refiner):
#         if(not is_eed):
#             num_inf_steps = int(num_inf_steps*0.25)
#             high_noise_frac = 0.0
#         # num_inf_steps = 25 
        
        
#     os.makedirs(op_fld, exist_ok=True)
#     op_name = os.path.join(op_fld, "generated.png")
#     image[0].save(op_name)
#     return [{'seed':seed, 'path':op_name}]


# def gen_sdxl_controlnet(prompt, ctrl_img_path, op_fld, control_type="canny", lora_path=None, lora_scale = 0.8,
#     n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inf_steps=75, use_refiner=True,
#     safety_checker=None, collect_cache=True): 

#     pipe, processor = create_controlnet(control_type="canny", lora_path=None, lora_scale=0.8)
#     gen_one_img(pipe, refiner, processor, prompt, prompt_2, ctrl_img_path, ctrl_img, op_fld, control_type="canny",
#         n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inf_steps=75)


#     if(use_refiner):
#         # num_inf_steps = 25 
#         refiner = DiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, 
#             torch_dtype=torch.float16, use_safetensors=True, variant="fp16",)
#         refiner.enable_model_cpu_offload()
#         image = refiner(prompt=prompt, prompt_2=prompt, 
#             image=image, num_inference_steps=num_inf_steps, 
#             negative_prompt=n_prompt, negative_prompt_2=n_prompt,
#             generator=generator).images

#     os.makedirs(op_fld, exist_ok=True)
#     op_name = os.path.join(op_fld, "generated.png")
#     image[0].save(op_name)
#     return [{'seed':seed, 'path':op_name}]

# # The first stage outputs a latents and then the refiner takes the latents and produces the output
# def sdxl_inpaint(theme, prompt, op_fld, lora_path=None, lora_scale = 0.8,
#     n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inference_steps=75, use_refiner=True,
#     is_eed=True, high_noise_frac = 0.7, safety_checker=None, collect_cache=True):#prompt, neg_prompt, img_path, mask_path, vae_path=None,):

#     if(not use_refiner):
#         is_eed = False

#     pipe = create_base_model(lora_path=lora_path, lora_scale = lora_scale, is_inpaint=True)
#     # generate image
#     generator = seed_to_generator(seed)

#     if(is_eed):
#         num_inf_base_steps = 75
#         denoising_end = 0.7
#         num_inf_refiner_steps = 75
#         denoising_start = 0.7
#     else:
#         num_inf_base_steps = 75
#         denoising_end = 1.0
#         num_inf_refiner_steps = 25
#         denoising_start = 0.0

#     image = pipe(prompt=prompt, prompt_2=prompt, image=init_image, mask_image=mask_image,
#         num_inference_steps=num_inf_base_steps, denoising_end=denoising_end,
#         generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
#         height=height, width=width, output_type="latent" if use_refiner else "pil",).images
#     if(use_refiner):
#         refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=pipe.text_encoder_2,
#             vae=pipe.vae, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        
        

#     os.makedirs(op_fld, exist_ok=True)
#     op_name = os.path.join(op_fld, "generated.png")
#     image.save(op_name)
#     return [{'seed':seed, 'path':op_name}]


