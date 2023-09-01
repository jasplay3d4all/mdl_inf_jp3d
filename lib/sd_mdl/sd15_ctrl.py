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

from cmn import lora_base_path, load_vae, ti_base_path



def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

class sd_model:
    def __init__(self, control_type=None,  safety_checker=False):

        # print("Input args ", theme, prompt)
        model_map = theme_to_model_map[theme] 
        n_prompt = common_n_prompt + model_map["n_prompt"] + n_prompt
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
    
def create_model(theme, n_prompt, control_type, safety_checker):
    
    else:
        return sd_model, None


def gen_one_img(sd_model, processor, prompt, control_type, ctrl_img_path, ctrl_img, op_fld, height, width, seed, num_images):
    if(control_type == "inpaint"):
        if(ctrl_img_path):
            print("Error: inpaint expects img instead of img path for control")
            return -1
    elif(control_type):
        if(ctrl_img_path):
            ctrl_img = Image.open(ctrl_img_path).convert("RGB").resize((512, 512))
        ctrl_img = np.array(processor(ctrl_img, to_pil=True))[None,...]/255.0
        # print("Processed img ", ctrl_img.shape, np.max(ctrl_img))
    else:
        ctrl_img = None

    sd_model.set_prompt(prompt)
    output_info_list = sd_model.gen_img_lst(op_fld, height=height, width=width, seed=seed, num_images=num_images, 
        ctrl_img=ctrl_img)
    return output_info_list


def gen_img(theme, prompt, op_fld=None, control_type=None, ctrl_img=None, ctrl_img_path=None, n_prompt="", height=512, width=512, 
        seed=-1, num_images=1, num_inf_steps=30, safety_checker=None, collect_cache=True):

    sd_model, processor = create_model(theme, n_prompt, control_type, safety_checker)
    output_info_list = gen_one_img(sd_model, processor, prompt, control_type, ctrl_img_path, ctrl_img, op_fld, height, width, seed, num_images)

    if(collect_cache):
        del sd_model
        mem_plg.collect_cache()
    return output_info_list
