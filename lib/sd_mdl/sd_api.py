
from .sd15_ctrl import sd15_model
from .sdxl_ctrl import sdxl_model
import os
import numpy as np
from PIL import Image


class sd_model:
    def __init__(self, theme, control_type=None, safety_checker=False):
        if("sdxl" in theme):
            self.model = sdxl_model(theme, control_type, safety_checker)
        else:
            self.model = sd15_model(theme, control_type, safety_checker)
        return

    def gen_img(self, prompt, ctrl_img_path=None, ctrl_img=None, op_fld=None, seed=-1, **kwargs):
        if(ctrl_img_path):
            ctrl_img = np.array(Image.open(ctrl_img_path).convert("RGB"))/255.0
        # if(mask_img_path):
        #     mask_img = np.array(Image.open(mask_img_path))/255.0
        op = self.model.gen_img(prompt=prompt, ctrl_img=ctrl_img, seed=seed, **kwargs)
        if(op_fld):
            os.makedirs(op_fld, exist_ok=True)
            op_name = os.path.join(op_fld, "image.png")
            op["image"].save(op_name)
            return [{"seed":op["seed"], "path":op_name}]
        else:
            return op



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

    # prompt = "A statue of a girl dancing"
    # negative_prompt = "low quality, bad quality, sketches"    
    # num_inference_steps = 75
    # high_noise_frac = 0.7
        
    # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    # init_image = load_image(img_url).convert("RGB")
    # mask_image = load_image(mask_url).convert("RGB")
