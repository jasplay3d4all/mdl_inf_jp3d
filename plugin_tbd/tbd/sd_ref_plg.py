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
from .ref_pipe.sd_ctrl_ref_pipe import StableDiffusionControlNetReferencePipeline
from .ref_pipe.sd_ref_pipe import StableDiffusionReferencePipeline

from .data_loader import FolderLoad

def initialize_pipe(model_name, controlnet_list, no_ctrl=True):

    if(no_ctrl):
        pipe = StableDiffusionReferencePipeline.from_pretrained(
            model_name, safety_checker=None,
            torch_dtype=torch.float16)
    else:
        # https://discuss.huggingface.co/t/how-to-enable-safety-checker-in-stable-diffusion-2-1-pipeline/31286
        pipe = StableDiffusionControlNetReferencePipeline.from_pretrained(
            model_name, controlnet=controlnet_list, 
            safety_checker=None, torch_dtype=torch.float16)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    num_inference_steps = 20
    return pipe, num_inference_steps



class sd_model:
    def __init__(self, model_name_list, prompt, n_prompt, is_inpaint=False, no_ctrl=True):
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

        if(is_inpaint):
            controlnet_list.append(
                ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16))
            controlnet_conditioning_scale.append(0.8)

        self.model_name_list = model_name_list
        self.is_inpaint = is_inpaint

        model_name = "SG161222/Realistic_Vision_V2.0"
        # model_name = "runwayml/stable-diffusion-v1-5"

        self.pipe, self.num_inference_steps = \
            initialize_pipe(model_name, controlnet_list, no_ctrl=no_ctrl)
        self.no_ctrl = no_ctrl
        
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        self.prompt = prompt
        self.n_prompt = n_prompt

        return
        
    def gen_img(self, images_list, seed=-1, ref_img=None):
        if seed == -1:
            seed = random.randint(0, 65535)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # image = self.pipe( self.prompt,
        #     control_image=images_list, image=ref_img,
        #     generator=generator, negative_prompt=self.n_prompt,
        #     controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        #     num_inference_steps=self.num_inference_steps).images[0]
        
        if(self.no_ctrl):
            image = self.pipe(ref_image=ref_img,
                prompt=self.prompt, negative_prompt=self.n_prompt,
                num_inference_steps=self.num_inference_steps, reference_attn=True,
                reference_adain=True).images[0]
        else:
            image = self.pipe(ref_image=ref_img,
                prompt=self.prompt, image=images_list,
                negative_prompt=self.n_prompt, controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=self.num_inference_steps, reference_attn=True,
                reference_adain=True).images[0]

        return np.array(image)
    
    def gen_img_lst(self, main_dir, op_fld, res=512, seed=-1):
        os.makedirs(op_fld, exist_ok=True)

        dataset = FolderLoad(main_dir, self.model_name_list, res)

        num_images = len(dataset)
        for idx in range(num_images):
            img_lst = dataset.get_img_lst(idx)
            if(self.is_inpaint):
                masked_img, mask_pixel = dataset.get_inpaint_mask(idx)
                img_lst.append(masked_img)
            ref_img = dataset.get_ref_img(idx)

            img_np = self.gen_img(img_lst, seed=seed, ref_img=Image.fromarray(ref_img))
            
            op_file = "output"+str(idx+1).zfill(4)+".png"

            op_pth = os.path.join(op_fld, op_file)
            image = Image.fromarray(img_np.astype(np.uint8))
            image.save(op_pth)

