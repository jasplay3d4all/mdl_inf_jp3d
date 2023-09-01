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


# Controlnet SDXL
# SDXL Inpainiting
# SDXL Inpainting using refiner: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#inpainting 
# Img2imgpipeline
# LoRA SDXL - 
# https://github.com/huggingface/diffusers/pull/4287/#issuecomment-1655936838 
# encode_prompt(lora_scale)
# https://github.com/huggingface/diffusers/issues/4348 - MulitLora load not supported now
# SDXL speedup: [Done]
# pipe.enable_model_cpu_offload()
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# SDXL Ensemble of Expert Denoisers
# Difference between InpaintRefiner and base refiner or using other refinement ideas
# https://pypi.org/project/invisible-watermark/
# https://huggingface.co/diffusers - sdxl different controlnet like canny, depth, small, medium etc?

def config_lora(pipe, lora_path, lora_scale):
    state_dict, network_alphas = pipe.lora_state_dict(lora_path, unet_config=pipe.unet.config,)
    pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet)
    pipe.load_lora_into_text_encoder(
        state_dict,
        network_alphas=network_alphas,
        text_encoder=pipe.text_encoder,
        lora_scale=0.8 #self.lora_scale,
    )

    return pipe

def gen_sdxl_img(prompt, op_fld, lora_path=None, lora_scale = 0.8, n_prompt="", 
    height=1024, width=1024, seed=-1, use_refiner=True,
    is_eed=False, high_noise_frac = 0.7, num_images=1, num_inf_steps=75,):

    if(not use_refiner):
        is_eed=False

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
        vae=vae, torch_dtype=torch.float16, use_safetensors=True)
    if(lora_path):
        pipe = config_lora(pipe, lora_path, lora_scale)        
    pipe.enable_model_cpu_offload()

    if seed == -1:
        seed = random.randint(0, 65535)
    generator = torch.Generator(device="cpu").manual_seed(seed)

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
        refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, 
                torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        refiner.enable_model_cpu_offload()
        image = refiner(prompt=prompt, prompt_2=prompt, image=image, num_inference_steps=num_inf_steps,
            denoising_start=high_noise_frac, 
            negative_prompt=n_prompt, negative_prompt_2=n_prompt,
            generator=generator).images

    os.makedirs(op_fld, exist_ok=True)
    op_name = os.path.join(op_fld, "generated.png")
    image[0].save(op_name)
    return [{'seed':seed, 'path':op_name}]


def gen_sdxl_controlnet(prompt, ctrl_img_path, op_fld, control_type="canny", lora_path=None, lora_scale = 0.8,
    n_prompt="", height=1024, width=1024, seed=-1, num_images=1, num_inf_steps=75, use_refiner=True,
    safety_checker=None, collect_cache=True): 

    model_name_mapper = {
        "canny_small":[0.5, "diffusers/controlnet-canny-sdxl-1.0-small"],
        "canny_mid":[0.5, "diffusers/controlnet-canny-sdxl-1.0-mid"],
        "canny":[0.5, "diffusers/controlnet-canny-sdxl-1.0"],
        "depth_small":[0.5, "diffusers/controlnet-depth-sdxl-1.0-small"],
        "depth_mid":[0.5, "diffusers/controlnet-depth-sdxl-1.0-mid"],
        "depth":[0.5, "diffusers/controlnet-depth-sdxl-1.0"],
    }
    ctrl_type_to_processor_id = {
                # "pidiedge" : "softedge_pidinet", "hededge" : "softedge_hedsafe",
                # "inpaint" : [1.0, "lllyasviel/control_v11p_sd15_inpaint"],
                "canny" : "canny",
                # "openpose" : "openpose_full",
                "depth" : "depth_midas",
                # "depth" : "depth_zoe",
    }
    ctrl_img = Image.open(ctrl_img_path).convert("RGB")#.resize((512, 512))
    ctrl_img = Processor(ctrl_type_to_processor_id[control_type])(ctrl_img, to_pil=True)

    # initialize the models and pipeline
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
            controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
    if(lora_path):
        pipe = config_lora(pipe, lora_path, lora_scale)

    pipe.enable_model_cpu_offload()
    # pipe.to("cuda")
    # This speed up not working currently
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # pipe_list.append(pipe)        
    # generate image
    if seed == -1:
        seed = random.randint(0, 65535)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    num_inf_steps = 75    
    image = pipe(prompt, prompt_2=prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=ctrl_img,
        num_inference_steps=num_inf_steps, generator=generator, negative_prompt=n_prompt, negative_prompt_2=n_prompt,
        height=height, width=width, output_type="latent" if use_refiner else "pil"
    ).images
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

    # prompt = "A statue of a girl dancing"
    # negative_prompt = "low quality, bad quality, sketches"    
    # num_inference_steps = 75
    # high_noise_frac = 0.7

    if(not use_refiner):
        is_eed = False
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", 
        vae=vae, use_safetensors=True).to("cuda")

    refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

    # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    # init_image = load_image(img_url).convert("RGB")
    # mask_image = load_image(mask_url).convert("RGB")

    pipe = config_lora(pipe, lora_path, lora_scale)

    # generate image
    if seed == -1:
        seed = random.randint(0, 65535)
    generator = torch.Generator(device="cpu").manual_seed(seed)

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
        image = refiner(prompt=prompt,prompt_2=prompt, image=image,mask_image=mask_image,
            num_inference_steps=num_inf_refiner_steps, denoising_start=denoising_start,
            generator=generator, negative_prompt=n_prompt,  negative_prompt_2=n_prompt,
            height=height, width=width).images

    os.makedirs(op_fld, exist_ok=True)
    op_name = os.path.join(op_fld, "generated.png")
    image.save(op_name)
    return [{'seed':seed, 'path':op_name}]


if __name__ == "__main__":
    theme = "people"
    op_fld = "./share_vol/test/img"
    # prompt = "Strawberry icecream with chocolate sauce placed on a wall in new york"
    # logo_path = "./share_vol/data_io/inp/logo_mealo.png" # edge_inv.png" # 
    logo_path = "./share_vol/data_io/inp/3.jpeg" # edge_inv.png" # 
    lora_path = "./share_vol/models/lora/cyborg_style_xl-alpha.safetensors"

    # prompt = "cyborg style, cyborg, 3d style,3d render,cg,beautiful, goddess Kaali, fully clothed, looking at viewer, long braid, sparkling eyes, cyborg , mechanical limbs, cyberpunk, \
    #     cute gloves 3d_render_style_xl this has good facial feature holding weapons and gadget in each hand"

    # old lady with white hair,
    # prompt = "cyborg style, cyborg, 3d style,3d render,cg,beautiful, young boy, with human hands and legs who is fully clothed, looking at viewer, sparkling eyes, and smiling face,  cyborg , holding weapons and gadget in each hand"
    prompt = "cyborg style, cyborg, a group of male cyborgs dancing around a huge fireball in a circle with female cyborgs \
            group dancing in the centre and clapping hands with their male counterparts"


    # prompt = "a front facing cyborg style, man, solo, black hair, blue eyes, cable, mechanical parts, spine, android, cyborg, science fiction, simple background, black background from behind, high quality, high resolution"
    prompt = "android, 1boy, science fiction, glowing, full body, humanoid robot, blue eyes, cable, mechanical parts, spine, armor, power armor, standing, robot, cyberpunk, scifi, high quality, high resolution, dslr, 8k, 4k, ultrarealistic, realistic,, <lora:cyborg_style_xl:1>, <lora:perfect-eyes:1>,perfecteyes"

    # n_prompt = "(worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), (blush:1.2), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature,bad_hands, bad-artist-anime, easynegative, bad-image-v2-39000,"
    n_prompt = "drawing, painting, illustration, rendered, low quality, low resolution"
    # n_prompt = "text, watermark, low quality, medium quality, close up, blurry, censored, wrinkles, deformed, mutated text, watermark, low quality, medium quality, blurry, nsfw, wrinkles, deformed, mutated embedding:BadDream embedding:UnrealisticDream, drawing, painting, illustration, rendered, low quality, low resolution, ng_deepnegative_v1_75t, badhandv4, (worst quality:2), (low quality:2), (normal quality:2), blurry, lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale))"
    
    # gen_sdxl_controlnet(prompt, logo_path, op_fld, n_prompt=n_prompt, lora_path=lora_path)
    gen_sdxl_img(prompt, op_fld, lora_path=lora_path, n_prompt=n_prompt, use_refiner=True, is_eed=True)
    # sdxl_inpaint(prompt)
    # gen_img(theme, prompt, op_fld, control_type="midasdepth", ctrl_img_path=logo_path, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)
    
    logo_path = "./share_vol/data_io/inp/logo_mealo.png" # edge_inv.png" # 
    # gen_logo(logo_path, theme=theme, prompt=prompt, op_fld=op_fld, control_type="pidiedge")

    # sdxl_inpaint(prompt, neg_prompt, img_path, mask_path, vae_path=None, num_inference_steps = 75, high_noise_frac = 0.7):


    theme = "people"
    # prompt = "instagram photo, closeup face photo of 18 y.o swedish woman in dress, beautiful face, makeup, night city street, bokeh, motion blur"
    # prompt = "closeup face photo of caucasian man in black clothes, night city street, bokeh"
    prompt = "polaroid photo, night photo, photo of 24 y.o beautiful woman, pale skin, bokeh, motion blur"
    # gen_img(theme, prompt, op_fld, control_type=None, ctrl_img_path=None, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)


    # prompt = "Cyborg with multiple hands holding futuristic gadgets with beautiful face and sparkling eyes"
    # gen_img(theme, prompt, op_fld, control_type="pidiedge", 

    # prompt = "cyborg style, cyborg, 3d style,3d render,cg,beautiful, goddess Kaali, looking at viewer, long braid, sparkling eyes, cyborg , mechanical limbs, cyberpunk, \
    #     cute gloves 3d_render_style_xl this has good facial feature holding weapons and gadget in each hand"
    # gen_img(theme, prompt, op_fld, control_type="pidiedge", ctrl_img_path=logo_path, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)


#     # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md
#     lora_model_id = "./share_vol/models/lora/cyborg_style_xl-alpha.safetensors"
#     # pipe.load_lora_weights(lora_model_id)

#     state_dict, network_alphas = pipe.lora_state_dict(lora_model_id, unet_config=pipe.unet.config,)
#     pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipe.unet)
#     pipe.load_lora_into_text_encoder(
#         state_dict,
#         network_alphas=network_alphas,
#         text_encoder=pipe.text_encoder,
#         lora_scale=0.8 #self.lora_scale,
#     )


#     # Speedup and pipeline refinement
#     # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
#     pipe.enable_model_cpu_offload()
#     # This speed up not working currently
#     # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# #         pipe_list.append(pipe)



#     # get canny image
#     image = np.array(image)
#     image = cv2.Canny(image, 100, 200)
#     image = image[:, :, None]
#     image = np.concatenate([image, image, image], axis=2)
#     canny_image = Image.fromarray(image)

#     # generate image
#     image = pipe(
#         prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image,
#         # lora_scale = 0.8
#     ).images[0]

#     image.save("output.png")
#     return

# pipeline type = controlnet, inpaint, img2img, base, 
# def gen_sdxl_img(pipe_type="base", is_refiner=True, lora_path=None):

#     vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
#     model_list = ["stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-xl-refiner-1.0"]
#     pipetype2pipe = {"base":StableDiffusionXLPipeline, "inpaint":StableDiffusionXLInpaintPipeline,
#         "img2img":StableDiffusionXLImg2ImgPipeline} #"controlnet":StableDiffusionXLControlNetPipeline
#     pipe_list = []
#     for model in model_list:
#         if(pipe_type == "controlnet"): #https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl
#             model_name_mapper = { "canny" : [0.5, "diffusers/controlnet-canny-sdxl-1.0"],} # recommended for good generalization
#             controlnet_list = []
#             controlnet_conditioning_scale = []
#             controlnet_list.append(
#                 ControlNetModel.from_pretrained(model_name_mapper[control_type][1], torch_dtype=torch.float16))
#             controlnet_conditioning_scale.append(model_name_mapper[control_type][0])

#             pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
#         else:
#             pipe = pipetype2pipe[pipe_type].from_pretrained(model, vae=vae,
#                 torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
#         # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md
#         pipe.load_lora_weights(lora_model_id)
#         # Speedup and pipeline refinement
#         # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
#         pipe.enable_model_cpu_offload()
#         pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#         pipe_list.append(pipe)

#     prompt = "A majestic lion jumping from a big stone at night"
#     n_steps = 40
#     high_noise_frac = 0.8
#     image = base(
#         prompt=prompt,
#         num_inference_steps=n_steps,
#         denoising_end=high_noise_frac,
#         output_type="latent",
#     ).images
#     image = refiner(
#         prompt=prompt,
#         num_inference_steps=n_steps,
#         denoising_start=high_noise_frac,
#         image=image,
#     ).images[0]
#     return




# if(pipe_type="base"):
#     pipe = StableDiffusionXLPipeline.from_pretrained(model, vae=vae,
#         torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# elif(pipe_type="inpaint"):
#     pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, vae=vae,
#         torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# elif(pipe_type="img2img"):
#     pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, vae=vae,
#         torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# el


# 
# https://github.com/search?q=repo%3Ahuggingface%2Fdiffusers+sdxl+lora&type=code

# pipe.unet.load_attn_procs(lora_model_path)
# image = pipe(
# ...     "A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}
# ... ).images[0]




# import torch



# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt=prompt).images[0]

# import torch
# from diffusers.utils import load_image


# url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

# init_image = load_image(url).convert("RGB")
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, image=init_image).images[0]

# import torch
# from diffusers.utils import load_image

# pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

# init_image = load_image(img_url).convert("RGB")
# mask_image = load_image(mask_url).convert("RGB")

# prompt = "A majestic tiger sitting on a bench"
# high_noise_frac = 0.8
# image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80,
#     denoising_start=high_noise_frac,output_type="latent").images[0]
# image.save("output.png")

# from diffusers import DiffusionPipeline
# import torch

# base = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# base.to("cuda")

# refiner = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# refiner.to("cuda")

# n_steps = 40
# high_noise_frac = 0.8

# prompt = "A majestic lion jumping from a big stone at night"

# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
# image.save("output.png")

# def sdxl_refiner(prompt, n_prompt, base, image, num_inf_steps, high_noise_frac, 
#     generator, is_inpaint=False, is_eed=True):
#     if(is_inpaint):
#         refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=base.text_encoder_2,
#             vae=base.vae, torch_dtype=torch.float16, use_safetensors=True)
#     else:
#         refiner = DiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=base.text_encoder_2, vae=base.vae, 
#             torch_dtype=torch.float16, use_safetensors=True, variant="fp16",)
#     refiner.enable_model_cpu_offload()
#     # pipe.to("cuda")
#     # This speed up not working currently
#     # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#     # pipe_list.append(pipe)
#     if(not is_eed):
#         high_noise_frac = 0.0
#         num_inf_steps = 25

#     image = refiner(prompt=prompt, prompt_2=prompt, image=image, num_inference_steps=num_inf_steps,
#         # denoising_start=high_noise_frac, 
#         negative_prompt=n_prompt, negative_prompt_2=n_prompt,
#         generator=generator).images

#     return image


