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
    StableDiffusionControlNetImg2ImgPipeline, \
    ControlNetModel, UniPCMultistepScheduler, StableDiffusionInpaintPipeline

from .data_loader import FolderLoad
from lib.lora.convert_lora_safetensors_to_diffuser import convert 

# LoRA List: 
# foodphoto.safetensors - FOODPHOTO - https://civitai.com/models/45322/food-photography
# splashes_v.1.1.safetensors - SPLASH SPLASHES SPLASHING EXPLOSION EXPLODING - https://civitai.com/models/81619/splash
# add_detail.safetensors - any weight up/down to 2/-2
# ghibli_style_offset.safetensors = ghibli style - https://civitai.com/models/6526?modelVersionId=7657
# more_detail.safetensors - between 0.5 and 1 weight - https://civitai.com/models/82098/add-more-details-detail-enhancer-tweaker-lora
# ChocolateWetStyle.safetensors - ChocolateWetStyle - https://civitai.com/models/67132/chocolate-wet-style
# FoodPorn_v2.safetensors - foodporn - https://civitai.com/models/88717/foodporn
# LyCoRIS List
# fodm4st3r.safetensors - ART BY FODM4ST3R, FODM4ST3R, FODM4ST3R STYLE https://civitai.com/models/67467/rfktrs-food-master-hot-foods-edition
# TI
# negative_hand-neg.pt - NEGATIVE_HAND NEGATIVE_HAND-NEG - https://civitai.com/models/56519/negativehand-negative-embedding
# easynegative.safetensors - easynegative - https://civitai.com/models/7808/easynegative
# ng_deepnegative_v1_75t.pt - ng_deepnegative_v1_75t - https://civitai.com/models/4629/deep-negative-v1x
# fastnegative.pt - FastNegativeV2 - https://civitai.com/models/71961/fast-negative-embedding
# UnrealisticDream.pt - UnrealisticDream - https://civitai.com/models/72437?modelVersionId=77173
# BadDream.pt - BadDream - https://civitai.com/models/72437?modelVersionId=77169

def initialize_pipe(model_name, controlnet_list, use_ctrl=True): #, use_inpaint=True):

    # if(use_inpaint):
    #     # model_name = "runwayml/stable-diffusion-inpainting"
    #     model_name = "runwayml/stable-diffusion-v1-5"
    # else:
    #     model_name = "SG161222/Realistic_Vision_V2.0"
    #     # model_name = "runwayml/stable-diffusion-v1-5"

    if(use_ctrl):
        # https://discuss.huggingface.co/t/how-to-enable-safety-checker-in-stable-diffusion-2-1-pipeline/31286
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet_list, 
            safety_checker=None, torch_dtype=torch.float16)
        # print("Model created ", len(controlnet_list))
    # elif(use_inpaint):
    #     pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #         model_name, safety_checker=None,
    #         revision="fp16", torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, safety_checker=None, torch_dtype=torch.float16) #revision="fp16", 
    
    lora_base_path = "./share_vol/models/lora/"
    lora_name_list = ["add_detail.safetensors", "more_detail.safetensors",
                      "foodphoto.safetensors", "FoodPorn_v2.safetensors",
                      "ChocolateWetStyle.safetensors", 
                      "splashes_v.1.1.safetensors", "ghibli_style_offset.safetensors",]
    alpha_wgt_list = [0.8, 0.8]#, 0.8, 0.8, 0.8, 0.8]#, 0.8]
    for lora_name, alpha_wgt in zip(lora_name_list, alpha_wgt_list):
        lora_path = os.path.join(lora_base_path, lora_name)
        print("Lora ", lora_name)
        pipe = convert(pipe, lora_path, LORA_PREFIX_UNET="lora_unet", 
            LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=alpha_wgt)

    ti_base_path = "./share_vol/models/ti/"
    ti_path_list = ["BadDream.pt", # "fastnegative.pt", 
        "negative_hand-neg.pt", "ng_deepnegative_v1_75t.pt", "UnrealisticDream.pt", "easynegative.safetensors"]
    for ti_path in ti_path_list:
        print("TI ", ti_path)
        pipe.load_textual_inversion(os.path.join(ti_base_path, ti_path))

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    num_inference_steps = 30
    return pipe, num_inference_steps


class sd_model:
    def __init__(self):

        self.n_prompt = "negative_hand, negative_hand-neg, easynegative, ng_deepnegative_v1_75t, \
            FastNegativeV2, UnrealisticDream, BadDream"
        
        self.n_prompt += "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, \
            cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, \
            ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
            mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
            disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, \
            fused fingers, too many fingers, long neck, BadDream, UnrealisticDream"
        return
    def set_model_name(self, model_name):
        # model_name = "SG161222/Realistic_Vision_V2.0"
        # model_name = "runwayml/stable-diffusion-v1-5"
        self.model_name = model_name
        return


    def set_prompt(self, fg_prompt, bg_prompt):
        self.fg_prompt = fg_prompt
        self.bg_prompt = bg_prompt
        self.prompt = fg_prompt+bg_prompt
        return

    def set_controlnet(self, control_type):
        self.plain_pipe, self.num_plain_steps = initialize_pipe(self.model_name, [], use_ctrl=False)

        model_name_mapper = {
            "softedge" : [1.0, "lllyasviel/control_v11p_sd15_softedge"],
            "inpaint" : [1.0, "lllyasviel/control_v11p_sd15_inpaint"],
            "canny" : [1.0, "lllyasviel/control_v11p_sd15_canny"],
        } 
        controlnet_list = []
        controlnet_conditioning_scale = []
        # Prepare inpaint model
        controlnet_list.append(
            ControlNetModel.from_pretrained(model_name_mapper[control_type][1], torch_dtype=torch.float16))
        controlnet_conditioning_scale.append(model_name_mapper[control_type][0])

        self.inpaint_pipe, self.num_inpaint_steps = initialize_pipe(self.model_name, controlnet_list, use_ctrl=True)
        
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        return 
        
    def gen_img_lst(self, op_fld, height=512, width=512, seed=-1, num_images=1, 
        use_inpaint=False, inpaint_img=None):

        os.makedirs(op_fld, exist_ok=True)
        output_info_list = []
        for idx in range(num_images):
            if seed == -1:
                seed = random.randint(0, 65535)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            if(use_inpaint):
                # print("Input image dimension ", bg_dpt_mask.shape, len(img_lst))
                image = self.inpaint_pipe(self.prompt, image=[inpaint_img], 
                    num_inference_steps=self.num_inpaint_steps,
                    generator=generator, negative_prompt=self.n_prompt,
                    controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                    height=height, width=width).images[0]
            else:
                image = self.plain_pipe(self.prompt, num_inference_steps=self.num_plain_steps,
                    generator=generator, negative_prompt=self.n_prompt,
                    # image=bg_img, mask_image=dpt_mask,
                    height=height, width=width).images[0]
            op_name = os.path.join(op_fld, str(idx).zfill(5)+".png")
            image.save(op_name)
            output_info_list.append({'seed':seed, 'op_name':op_name})
        return output_info_list
    
