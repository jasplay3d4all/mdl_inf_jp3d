
from diffusers import StableDiffusionUpscalePipeline, DiffusionPipeline, ControlNetModel, \
        StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image, ImageOps
import torch
import os

def sdx4upscaler(low_res_img):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()
    upscale_img = pipeline(prompt=prompt, image=low_res_img).images[0]
    return upscale_img

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def sd_tile_scaler(low_res_img_path):
    # source_image = load_image('https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png')
    low_res_img = Image.open(low_res_img_path).convert("RGB").resize((512, 512))
    # condition_image = resize_for_condition_image(low_res_img, 1024)
    image = sdx4upscaler(low_res_img)

    # controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    # # pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    # #             "runwayml/stable-diffusion-v1-5",
    # #             controlnet=controlnet,
    # #             safety_checker=None,
    # #             torch_dtype=torch.float16
    # #             )
    # pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
    #         custom_pipeline="stable_diffusion_controlnet_img2img",
    #         controlnet=controlnet, torch_dtype=torch.float16).to('cuda')
    # pipe.enable_xformers_memory_efficient_attention()

    # image = pipe(prompt="best quality", negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
    #             image=condition_image, controlnet_conditioning_image=condition_image, 
    #             width=condition_image.size[0], height=condition_image.size[1], strength=1.0,
    #             generator=torch.manual_seed(0), num_inference_steps=32).images[0]

    os.makedirs(op_fld, exist_ok=True)
    op_name = os.path.join(op_fld, "generated.png")
    image.save(op_name)
    return [{'path':op_name}]

def sd_img2img(low_res_img_path, prompt, op_fld, n_prompt="", height=1024, width=1024, 
    seed=-1, strength=0.75, guidance_scale=7.5):

    model_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_auth_token=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    init_img = Image.open(low_res_img_path).convert("RGB").resize((width, height))

    if seed == -1:
        seed = random.randint(0, 65535)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(prompt=prompt, init_image=init_img, strength=strength, 
        guidance_scale=guidance_scale, generator=generator, negative_prompt=n_prompt).images[0]

    img_scl_path = os.path.join(op_fld, "img_scl")
    os.makedirs(img_scl_path, exist_ok=True)
    op_name = os.path.join(img_scl_path, "generated.png")
    image[0].save(op_name)
    return [{'seed':seed, 'path':op_name}]
    


if __name__ == "__main__":
    theme = "food"
    op_fld = "./share_vol/test/scl"
    # prompt = "Strawberry icecream with chocolate sauce placed on a wall in new york"
    prompt = "foodphoto, splashes, Hot piping coffee and croissant <lora:more_details:0.7>"
    # gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)
    
    low_res_img_path = "./share_vol/test/img/00000.png" # edge_inv.png" # 
    # gen_logo(logo_path, theme=theme, prompt=prompt, op_fld=op_fld, control_type="pidiedge")
    sd_tile_scaler(low_res_img_path)

