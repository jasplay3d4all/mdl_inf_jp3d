from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import os 
from PIL import Image
import  numpy as np
import cv2

def read_img_np(img_loc):
    image = Image.open(img_loc).convert("RGB")
    return np.array(image)

class FolderLoad(Dataset):
    def __init__(self, main_dir, model_name_list, transform, mask_blur=1.0):
        self.model_name_list = model_name_list
        self.main_dir = main_dir
        self.transform = transform
        self.mask_blur = mask_blur
        img_loc = os.path.join(self.main_dir, "img_ext")
        img_file_lst = os.listdir(img_loc)
        img_file_lst.sort()
        self.total_imgs = [ x for x in img_file_lst if x.endswith(".png") ]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        ref_img_loc = os.path.join(self.main_dir, "img_ext", self.total_imgs[idx])
        ref_img = self.transform(Image.open(ref_img_loc).convert("RGB"))

        tensor_image_list = []
        for model_name in self.model_name_list:
            if(model_name=="inpaint"):
                # https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint
                seg_loc = os.path.join(self.main_dir, "pan_seg", self.total_imgs[idx])
                mask_pixel = self.transform(Image.open(seg_loc).convert("L")).repeat(3, 1, 1)

                # # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
                # mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), self.mask_blur)
                # print("Control image ", image.shape, mask_pixel.shape)
                # print("Min and max ", torch.max(mask_pixel), \
                #     torch.min(image), torch.max(image))
                image = ref_img.clone()
                image[mask_pixel > 0.5] = -1.0

            else:
                img_loc = os.path.join(self.main_dir, model_name, self.total_imgs[idx])
                image = Image.open(img_loc).convert("RGB")
                image = self.transform(image)
                # image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
                # image = torch.from_numpy(image)
            image = image[None, ...]
            # image = einops.rearrange(image, 'b h w c -> b c h w')
            tensor_image_list.append(image) #
        # print("INP ", self.total_imgs[idx])
        return tensor_image_list, ref_img

def create_pipe(model_name_list):
    # https://huggingface.co/lllyasviel
    model_name_mapper = {
        "openpose" : [1.0, "lllyasviel/control_v11p_sd15_openpose"],
        "inpaint" :  [1.0, "lllyasviel/control_v11p_sd15_inpaint"]
    }
    controlnet_list = []
    controlnet_conditioning_scale = []
    for model in model_name_list:
        print("ControlNet ", model_name_mapper[model][1])
        controlnet_list.append(
            ControlNetModel.from_pretrained(model_name_mapper[model][1], torch_dtype=torch.float16))
        controlnet_conditioning_scale.append(model_name_mapper[model][0])

    model_name = "SG161222/Realistic_Vision_V2.0"
    # model_name = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_name, controlnet=controlnet_list, torch_dtype=torch.float16
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    return pipe, controlnet_conditioning_scale
   

def merge_bg_fg(main_dir, transform, fg_img, op_file):
    img_loc = os.path.join(main_dir, "img_ext", op_file)
    image = Image.open(img_loc).convert("RGB")
    image = transform(image)

    # image = read_img_np(img_loc).astype(np.float32) / 255.0
    seg_loc = os.path.join(main_dir, "pan_seg", op_file)
    mask_pixel = Image.open(seg_loc).convert("L")
    mask_pixel = transform(mask_pixel).repeat(3, 1, 1)

    fg_img = transform(fg_img.convert("RGB"))

    merged_img = fg_img*mask_pixel + (1 - mask_pixel)*image

    # Match the histograms for consistent images
    # processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)

    merged_img = transforms.ToPILImage()(merged_img)

    return merged_img



def gen_img(ip_fld, op_fld, prompt, n_prompt, 
    model_name_list, res=512, batch_size=1, seed=-1):
    os.makedirs(op_fld, exist_ok=True)

    transform = transforms.Compose([transforms.Resize(res),
                                 transforms.CenterCrop(res),
                                 transforms.ToTensor()])
    dataset = FolderLoad(ip_fld, model_name_list, transform)

    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size, shuffle=False)
    
    pipe, controlnet_conditioning_scale = create_pipe(model_name_list)

    if seed == -1:
        seed = random.randint(0, 65535)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for bat_idx, cur_img_data in enumerate(dataset):
        images_list, _ = cur_img_data

        for i in range(4): # Num retries to avoid NSFW images
            image = pipe(
                prompt,
                image=images_list,
                num_inference_steps=20,
                generator=generator,
                negative_prompt=n_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images[0]

            if(np.asarray(image).sum() > 100000):
                break
            else:
                print("Its a black one. Try ", i)

        op_file = "output"+str(bat_idx+1).zfill(4)+".png"

        image = merge_bg_fg(ip_fld, transform, image, op_file)

        op_pth = os.path.join(op_fld, op_file)
        image.save(op_pth)
        # print("OP ", op_file, image)
