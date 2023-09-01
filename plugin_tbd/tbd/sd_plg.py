from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import einops
import random
from lora_diffusion import tune_lora_scale, patch_pipe
import imageio

from diffusers import StableDiffusionControlNetPipeline, \
    StableDiffusionControlNetImg2ImgPipeline, \
    ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler

from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

from .data_loader import FolderLoad


def initialize_pipe(pipe_type, model_name, lora_path, controlnet_list):
    if(pipe_type == "i2i"):
        # https://github.com/huggingface/diffusers/pull/2584
        # https://github.com/huggingface/diffusers/issues/3032
        # https://github.com/huggingface/diffusers/issues/3095
        # installed from source: https://github.com/huggingface/diffusers
        # https://colab.research.google.com/drive/1e-yLAoLFUlGBSQqcGGHaMjiaOOdTlw3L?usp=sharing
        pipeline = StableDiffusionControlNetImg2ImgPipeline
    elif(pipe_type == "t2i" or pipe_type == "t2v"):
        pipeline = StableDiffusionControlNetPipeline
    
    # https://discuss.huggingface.co/t/how-to-enable-safety-checker-in-stable-diffusion-2-1-pipeline/31286
    pipe = pipeline.from_pretrained(
        model_name, controlnet=controlnet_list, 
        safety_checker=None, torch_dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     lora_path, safety_checker=None, torch_dtype=torch.float16)

    # https://github.com/huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py
    # https://github.com/camenduru/converter-colab/blob/main/converter_colab.ipynb
    # Use a virtual environment to convert lora from safetensor to pt based (folder path)
    # source pytorch_cpu/bin/activate
    # python lora_plg/huggingface_convertor.py --base_model_path "runwayml/stable-diffusion-v1-5" --checkpoint_path ../models/LoRA/Scarlett4.safetensors --dump_path ../models/LoRA/Scarlett4/
    # deactivate
    # It merges and dumps the whole model

    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # pipe.unet.load_attn_procs(lora_path+"/unet/diffusion_pytorch_model.bin")
    # patch_pipe(pipe, lora_path, patch_text=True, patch_ti=True, patch_unet=True,)
    # tune_lora_scale(pipe.unet, 1.00)
    # tune_lora_scale(pipe.text_encoder, 1.00)

    if(pipe_type == "t2v"):
        # Set the attention processor
        pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        num_inference_steps = 20
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        num_inference_steps = 30
    return pipe, num_inference_steps



class sd_model:
    def __init__(self, model_name_list, prompt, n_prompt, is_inpaint=False):
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
            print("ControlNet ", "inpaint")
            controlnet_list.append(
                ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16))
            controlnet_conditioning_scale.append(0.8)

        self.model_name_list = model_name_list
        self.is_inpaint = is_inpaint

        # model_name = "SG161222/Realistic_Vision_V2.0"
        model_name = "runwayml/stable-diffusion-v1-5"

        # https://civitai.com/models/7468/scarlett-johanssonlora
        lora_path = "../models/LoRA/Scarlett4/"

        self.t2i_pipe, self.num_inference_steps = \
            initialize_pipe("t2i", model_name, lora_path, controlnet_list)
        self.i2i_pipe, self.num_inference_steps = \
            initialize_pipe("i2i", model_name, lora_path, controlnet_list)
        
        controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
        self.t2v_pipe, self.num_inference_steps = \
            initialize_pipe("t2v", model_name, lora_path, controlnet_model)

        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        self.prompt = prompt
        self.n_prompt = n_prompt
        # self.num_inference_steps = 30

        return
        
    def gen_img(self, images_list, seed=-1, is_t2i=True, ref_img=None):
        if seed == -1:
            seed = random.randint(0, 65535)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # https://huggingface.co/lllyasviel/sd-controlnet-canny#example
        if(is_t2i):
            image = self.t2i_pipe( self.prompt,
                image=images_list, 
                num_inference_steps=self.num_inference_steps,
                generator=generator, negative_prompt=self.n_prompt,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                ).images[0]

        else:
            image = self.i2i_pipe( self.prompt,
                control_image=images_list, image=ref_img,
                generator=generator, negative_prompt=self.n_prompt,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=self.num_inference_steps).images[0]

        return np.array(image)
    
    def gen_img_lst(self, main_dir, op_fld, res=512, seed=-1, is_t2i=True):
        os.makedirs(op_fld, exist_ok=True)

        dataset = FolderLoad(main_dir, self.model_name_list, res)

        num_images = len(dataset)
        for idx in range(num_images):
            img_lst = dataset.get_img_lst(idx)
            if(self.is_inpaint):
                masked_img, mask_pixel = dataset.get_inpaint_mask(idx)
                img_lst.append(masked_img)
                print("Input dim ", masked_img.shape)
            ref_img = dataset.get_ref_img(idx)

            img_np = self.gen_img(img_lst, seed=seed, is_t2i=is_t2i, ref_img=Image.fromarray(ref_img))
            
            op_file = "output"+str(idx+1).zfill(4)+".png"

            # if(self.is_inpaint):
            #     img_np = img_np*mask_pixel + (1 - mask_pixel)*ref_img
            #     # print(ref_img.shape, img_np.shape, mask_pixel.shape)
    
            op_pth = os.path.join(op_fld, op_file)
            image = Image.fromarray(img_np.astype(np.uint8))
            image.save(op_pth)


    def gen_vdo(self, main_dir, op_fld, res=512, seed=-1, bat_size = 6):#11):
        os.makedirs(op_fld, exist_ok=True)
        if len(self.model_name_list) > 1:
            print("Text2Vdo does not support MultiControlNet. So only one controlnet could be used")
            exit(-1)

        dataset = FolderLoad(main_dir, self.model_name_list, res)

        frame_count = len(dataset)
        # bat_size = 11 # This is based on the memory avialble for processing
        num_batch = int(frame_count/bat_size) + 1

        if seed == -1:
            seed = random.randint(0, 65535)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        latents = torch.randn((1, 4, 64, 64), device="cpu", generator=generator,
            dtype=torch.float16)        
        
        for bat_idx in range(num_batch):

            if ((bat_idx + 1)*bat_size > frame_count):
                num_frm_in_bat = frame_count - bat_idx*bat_size
            else:
                num_frm_in_bat = bat_size

            # Create batch 
            # Initialize the first image with fixed latent and pose so every frame generated would be 
            # generated using this and so will be consistent
            ctrl_img_lstlst = [dataset.get_img_lst(0)[0]]
            # ctrl_img_lstlst = []

            for i in range(num_frm_in_bat):
                idx = bat_idx*bat_size+i
                img_lst = dataset.get_img_lst(idx)
                # if(self.is_inpaint):
                #     masked_img, mask_pixel = dataset.get_inpaint_mask(idx)
                #     img_lst.append(masked_img)
                ctrl_img_lstlst.append(img_lst[0])

            generator = torch.Generator(device="cpu").manual_seed(seed)

            results = self.t2v_pipe( [self.prompt] * len(ctrl_img_lstlst),
                image=ctrl_img_lstlst, 
                num_inference_steps=self.num_inference_steps,
                latents=latents.repeat(len(ctrl_img_lstlst), 1, 1, 1), 
                negative_prompt=[self.n_prompt]* len(ctrl_img_lstlst),
                generator=generator,
                # controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                ).images

            for i in range(num_frm_in_bat):
                op_file = "output"+str(bat_idx*bat_size+i+1).zfill(4)+".png"
                op_pth = os.path.join(op_fld, op_file)
                # results[i].save(op_pth)
                results[i+1].save(op_pth)
                print("OP ", op_file, seed)
        return

