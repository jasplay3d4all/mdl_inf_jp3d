from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import einops
import random
import imageio

from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler

from diffusers import StableDiffusionControlNetPipeline, \
    StableDiffusionControlNetImg2ImgPipeline, \
    ControlNetModel, UniPCMultistepScheduler, StableDiffusionInpaintPipeline

from .data_loader import FolderLoad

def initialize_pipe(controlnet_list, use_ctrl=True, use_inpaint=True):

    # if(use_inpaint):
    #     # model_name = "runwayml/stable-diffusion-inpainting"
    #     model_name = "runwayml/stable-diffusion-v1-5"
    # else:
    #     model_name = "SG161222/Realistic_Vision_V2.0"
    #     # model_name = "runwayml/stable-diffusion-v1-5"

    model_name = "SG161222/Realistic_Vision_V2.0"

    if(use_ctrl):
        # https://discuss.huggingface.co/t/how-to-enable-safety-checker-in-stable-diffusion-2-1-pipeline/31286
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet_list, 
            safety_checker=None, torch_dtype=torch.float16)
        # print("Model created ", len(controlnet_list))
    elif(use_inpaint):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name, safety_checker=None,
            revision="fp16", torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, safety_checker=None,
            revision="fp16", torch_dtype=torch.float16)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    num_inference_steps = 30
    return pipe, num_inference_steps


class sd_model:
    def __init__(self, model_name_list, fg_prompt, bg_prompt, n_prompt, use_ctrl=True):
        # https://huggingface.co/lllyasviel
        model_name_mapper = {
            "openpose" : [1.0, "lllyasviel/control_v11p_sd15_openpose"],
            # "openpose" : [1.0, "lllyasviel/sd-controlnet-openpose"],
            "softedge_PIDI" : [1.0, "lllyasviel/control_v11p_sd15_softedge"],
            "midas_dpth" : [1.0, "lllyasviel/control_v11f1p_sd15_depth"],
            "zoe_dpth" : [1.0, "lllyasviel/control_v11f1p_sd15_depth"],

            # "inpaint" :  [1.0, "lllyasviel/control_v11p_sd15_inpaint"]
        }
        controlnet_list = []
        controlnet_conditioning_scale = []
        for model in model_name_list:
            print("ControlNet ", model_name_mapper[model][1])
            controlnet_list.append(
                ControlNetModel.from_pretrained(model_name_mapper[model][1], torch_dtype=torch.float16))
            controlnet_conditioning_scale.append(model_name_mapper[model][0])

        self.plain_pipe, self.num_plain_steps = initialize_pipe(controlnet_list, \
            use_ctrl=use_ctrl, use_inpaint=False)

        # Prepare inpaint model
        controlnet_list.append(
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16))
        controlnet_conditioning_scale.append(0.8)

        self.inpaint_pipe, self.num_inpaint_steps = initialize_pipe(controlnet_list, \
            use_ctrl=use_ctrl, use_inpaint=True)

        self.model_name_list = model_name_list
        self.use_ctrl = use_ctrl
        
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        self.prompt = fg_prompt + bg_prompt
        self.bg_prompt = bg_prompt
        self.n_prompt = n_prompt

        return
    
    def gen_img_lst(self, main_dir, op_fld, res=512, seed=-1):

        os.makedirs(op_fld, exist_ok=True)

        dataset = FolderLoad(main_dir, self.model_name_list, res)
        if seed == -1:
            seed = random.randint(0, 65535)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Generate background image
        if(self.use_ctrl):
            img_lst = dataset.get_img_lst(0)
            bg_img = self.plain_pipe(self.bg_prompt, image=img_lst, 
                num_inference_steps=self.num_plain_steps,
                generator=generator, negative_prompt=self.n_prompt,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                ).images[0]
        else:
            bg_img = self.plain_pipe(self.bg_prompt, num_inference_steps=self.num_plain_steps,
                generator=generator, negative_prompt=self.n_prompt,
                ).images[0]
        bg_img.save(os.path.join(op_fld, "bg_img.png"))

        num_images = len(dataset)
        for idx in range(num_images):
            bg_dpt_mask, dpt_mask = dataset.get_inpaint_dpth_mask(idx, bg_img, use_zoe=True)
            # mask_img = einops.rearrange(dpt_mask, 'b c h w -> b h w c')*255
            # mask_img = Image.fromarray(mask_img.squeeze().detach().numpy().astype(np.uint8))
            # mask_img.save(os.path.join(op_fld, "op_mask_"+str(idx+1).zfill(4)+".png"))

            if(self.use_ctrl):
                img_lst = dataset.get_img_lst(idx)
                img_lst.append(bg_dpt_mask)
                # print("Input image dimension ", bg_dpt_mask.shape, len(img_lst))
                inpaint_img = self.inpaint_pipe(self.prompt, image=img_lst, 
                    num_inference_steps=self.num_inpaint_steps,
                    generator=generator, negative_prompt=self.n_prompt,
                    controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                    ).images[0]
            else:
                inpaint_img = self.inpaint_pipe(self.prompt, num_inference_steps=self.num_inpaint_steps,
                    generator=generator, negative_prompt=self.n_prompt,
                    image=bg_img, mask_image=dpt_mask,
                    ).images[0]
            
            op_file = "output"+str(idx+1).zfill(4)+".png"

            op_pth = os.path.join(op_fld, op_file)
            # inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
            inpaint_img.save(op_pth)
