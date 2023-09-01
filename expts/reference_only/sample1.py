import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from sd_ref_only import StableDiffusionControlNetReferencePipeline
from sd_refwo_ctrl import StableDiffusionReferencePipeline

input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

ip_lnk = "https://static01.nyt.com/images/2023/05/24/arts/NYCB-season-highlights-03/NYCB-season-highlights-03-superJumbo.jpg?quality=75&auto=webp"
# input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

canny_img = load_image(ip_lnk)
canny_img = canny_img.resize((512, 512))
# get canny image
image = cv2.Canny(np.array(canny_img), 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetReferencePipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       controlnet=controlnet,
       safety_checker=None,
       torch_dtype=torch.float16
       ).to('cuda:0')

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

result_img = pipe(ref_image=input_image,
      prompt="single girl dancing in snow under waterfall",
      image=canny_image,
      num_inference_steps=20,
      reference_attn=True,
      reference_adain=True).images[0]
result_img.save("ref_ctrl.png")

pipe = StableDiffusionReferencePipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                torch_dtype=torch.float16
                ).to('cuda:0')
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

result_img = pipe(ref_image=input_image,
                        prompt="single girl dancing in snow under waterfall",
                        num_inference_steps=20,
                        reference_attn=True,
                        reference_adain=True).images[0]
result_img.save("ref_only.png")
input_image.save("input.png")
canny_img.save("canny.png")