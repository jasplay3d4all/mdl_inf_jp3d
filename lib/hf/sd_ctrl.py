from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import random
import imageio

from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler

from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, \
    ControlNetModel, UniPCMultistepScheduler, StableDiffusionInpaintPipeline, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel

# from .data_loader import FolderLoad
from lib.lora.convert_lora_safetensors_to_diffuser import convert 



def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

class sd_model:
    def __init__(self, base_model, control_type=None, lora_list=[], ti_list=[], n_prompt='', num_inf_steps=30,
        vae_path=None, clip_skip=0, safety_checker=False):

        def gen_pipe():
            controlnet_conditioning_scale = []
            if(control_type):
                model_name_mapper = {
                    "pidiedge" : [1.0, "lllyasviel/control_v11p_sd15_softedge"],
                    "hededge" : [1.0, "lllyasviel/control_v11p_sd15_softedge"],
                    "inpaint" : [1.0, "lllyasviel/control_v11p_sd15_inpaint"],
                    "canny" : [1.0, "lllyasviel/control_v11p_sd15_canny"],
                    "openpose" : [1.0, "lllyasviel/control_v11p_sd15_openpose"],
                    "midasdepth" : [1.0, "lllyasviel/control_v11f1p_sd15_depth"],
                    "zoedepth" : [1.0, "lllyasviel/control_v11f1p_sd15_depth"],
                } 

                controlnet_list = []
                # Prepare inpaint model
                controlnet_list.append(
                    ControlNetModel.from_pretrained(model_name_mapper[control_type][1], torch_dtype=torch.float16))
                controlnet_conditioning_scale.append(model_name_mapper[control_type][0])

                if(os.path.isfile(base_model)):
                    pipe = StableDiffusionControlNetPipeline.from_single_file(
                        base_model, controlnet=controlnet_list[0], safety_checker=safety_checker, torch_dtype=torch.float16)
                    # print("The model loaded from file ", base_model)
                else:
                    # https://discuss.huggingface.co/t/how-to-enable-safety-checker-in-stable-diffusion-2-1-pipeline/31286
                    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        base_model, controlnet=controlnet_list, safety_checker=safety_checker, torch_dtype=torch.float16)
                # print("Model created ", len(controlnet_list))
                self.use_ctrl_net = True
            else:
                if(os.path.isfile(base_model)):
                    pipe = StableDiffusionPipeline.from_single_file(
                        base_model, safety_checker=safety_checker, torch_dtype=torch.float16) #revision="fp16", 
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        base_model, safety_checker=safety_checker, torch_dtype=torch.float16) #revision="fp16", 
                self.use_ctrl_net = False

            self.lora_base_path = "./share_vol/models/lora/"
            alpha_wgt = 0.8
            for lora_name in lora_list:
                print("Lora ", lora_name)
                lora_path = os.path.join(self.lora_base_path, lora_name)
                pipe = convert(pipe, lora_path, LORA_PREFIX_UNET="lora_unet", 
                    LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=alpha_wgt)

            self.ti_base_path = "./share_vol/models/ti/"
            for ti_path in ti_list:
                print("TI ", ti_path)
                pipe.load_textual_inversion(os.path.join(self.ti_base_path, ti_path))

            # Load VAE:
            if(vae_path and os.path.isfile(vae_path)):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16).to("cuda")
                pipe.vae = vae
            elif(vae_path):
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to("cuda")
                pipe.vae = vae
            print("VAE Path ", vae_path)#vae

            # Load TextEncoder:
            # text_encoder = CLIPTextModel.from_pretrained(base_model, 
            #     subfolder="text_encoder", num_hidden_layers=12-clip_skip, torch_dtype=torch.float16).to("cuda")
            # pipe.text_encoder = text_encoder
            # print("Clip skip ", 12-clip_skip)


            
            
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()

                            
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            if(not safety_checker):
                pipe.safety_checker = disabled_safety_checker            
            return pipe, controlnet_conditioning_scale
        
        self.pipe, self.controlnet_conditioning_scale = gen_pipe()
        self.num_inf_steps = num_inf_steps

        self.n_prompt = n_prompt
        self.prompt = ''
        return

    def set_prompt(self, prompt):
        self.prompt = prompt
        return
    def append_n_prompt(self, n_prompt):
        self.n_prompt += n_prompt
        return

    def gen_img_lst(self, op_fld=None, height=512, width=512, seed=-1, num_images=1, ctrl_img=None):

        if(self.use_ctrl_net and ctrl_img is None):
            print("Error: Expected a control image")
            return -1

        output_info_list = []
        for idx in range(num_images):
            if seed == -1:
                seed_val = random.randint(0, 65535)
            else:
                seed_val = seed
            generator = torch.Generator(device="cpu").manual_seed(seed_val)

            if(self.use_ctrl_net):
                # print("Input image dimension ", bg_dpt_mask.shape, len(img_lst))
                image = self.pipe(self.prompt, image=[ctrl_img], 
                    num_inference_steps=self.num_inf_steps,
                    generator=generator, negative_prompt=self.n_prompt,
                    controlnet_conditioning_scale=self.controlnet_conditioning_scale[0],
                    height=height, width=width, guidance_scale=10).images[0]
            else:
                image = self.pipe(self.prompt, num_inference_steps=self.num_inf_steps,
                    generator=generator, negative_prompt=self.n_prompt,
                    height=height, width=width, guidance_scale=10).images[0]

            if(op_fld):
                os.makedirs(op_fld, exist_ok=True)
                op_name = os.path.join(op_fld, str(idx).zfill(5)+".png")
                image.save(op_name)
                output_info_list.append({'seed':seed_val, 'path':op_name})
            else:
                output_info_list.append({'seed':seed_val, 'image':image})
            
        return output_info_list
    
