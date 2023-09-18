from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import imageio

from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler

from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, \
    ControlNetModel, UniPCMultistepScheduler, StableDiffusionInpaintPipeline
from transformers import AutoTokenizer, CLIPTextModel
from controlnet_aux.processor import Processor

# from .data_loader import FolderLoad
from lib.lora.convert_lora_safetensors_to_diffuser import convert 

from .cmn import lora_base_path, load_vae, ti_base_path, model_base_path, seed_to_generator

from .sd15_theme import theme_to_model_map, common_lora_list, commom_ti_list, common_n_prompt, theme_to_model_map, \
    ctrl_type_to_processor_id, model_name_mapper



def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

class sd15_model:
    def __init__(self, theme, control_type=None,  safety_checker=False):

        # print("Input args ", theme, prompt)
        model_map = theme_to_model_map[theme] 
        n_prompt = common_n_prompt + model_map["n_prompt"]
        lora_list = common_lora_list + model_map["lora_list"]
        ti_list = commom_ti_list + model_map["ti_list"]
        
        base_model = model_map["base_model"]
        num_inf_steps=30
        vae_path=None
        clip_skip=0, 

        if(control_type in ctrl_type_to_processor_id):
            self.processor = Processor(ctrl_type_to_processor_id[control_type])

        def gen_pipe():
            controlnet_conditioning_scale = []
            if(control_type):
                controlnet_list = []
                # Prepare inpaint model
                controlnet_list.append(
                    ControlNetModel.from_pretrained(model_name_mapper[control_type][1], torch_dtype=torch.float16))
                controlnet_conditioning_scale.append(model_name_mapper[control_type][0])

                if(os.path.isfile(os.path.join(model_base_path, base_model))):
                    model_path = os.path.join(model_base_path, base_model)
                    pipe = StableDiffusionControlNetPipeline.from_single_file(
                        model_path, controlnet=controlnet_list[0], safety_checker=safety_checker, torch_dtype=torch.float16)
                    # print("The model loaded from file ", base_model)
                else:
                    # https://discuss.huggingface.co/t/how-to-enable-safety-checker-in-stable-diffusion-2-1-pipeline/31286
                    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        base_model, controlnet=controlnet_list[0], safety_checker=safety_checker, torch_dtype=torch.float16)
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

            self.lora_base_path = lora_base_path
            alpha_wgt = 0.8
            for lora_name in lora_list:
                print("Lora ", lora_name)
                lora_path = os.path.join(self.lora_base_path, lora_name)
                pipe = convert(pipe, lora_path, LORA_PREFIX_UNET="lora_unet", 
                    LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=alpha_wgt)

            self.ti_base_path = ti_base_path
            for ti_path in ti_list:
                print("TI ", ti_path)
                pipe.load_textual_inversion(os.path.join(self.ti_base_path, ti_path))

            # Load VAE:
            if(vae_path):
                pipe.vae = load_vae(vae_path)

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
        self.guidance_scale = 10
        self.control_type = control_type

        # self.prompt = ''
        return

    # def set_prompt(self, prompt):
    #     self.prompt = prompt
    #     return
    # def append_n_prompt(self, n_prompt):
    #     self.n_prompt += n_prompt
    #     return

    def gen_img(self, prompt, ctrl_img=None, n_prompt="", height=512, width=512, seed=-1, 
        num_images=1):
        if(self.use_ctrl_net and ctrl_img is None):
            print("Error: Expected a control image")
            return -1

        seed, generator = seed_to_generator(seed)
        n_prompt = self.n_prompt + n_prompt

        if(self.use_ctrl_net):
            if(self.control_type != "inpaint"):
                ctrl_img = [self.processor((255*ctrl_img).astype(np.uint8), to_pil=True)]
            # else:
            #     mask_image = -1*np.ones((out_hgt_y, out_wth_x, 3))
            #     mask_image[inp_pos_y:inp_pos_y+inp_hgt_y, inp_pos_x:inp_pos_x+inp_wth_x, :] = ctrl_image
            #     ctrl_image = mask_image

            image = self.pipe(prompt, image=ctrl_img, 
                num_inference_steps=self.num_inf_steps,
                generator=generator, negative_prompt=self.n_prompt,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale[0],
                height=height, width=width, guidance_scale=self.guidance_scale).images
        else:
            image = self.pipe(prompt, num_inference_steps=self.num_inf_steps,
                generator=generator, negative_prompt=self.n_prompt,
                height=height, width=width, guidance_scale=self.guidance_scale).images
        return {"seed":seed, "image":image[0]}