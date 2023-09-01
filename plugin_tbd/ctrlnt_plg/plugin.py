
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import natsort

# https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main
# https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
# https://huggingface.co/lllyasviel/Annotators/tree/main
# pip uninstall xformers

class FolderLoad(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        img_file_lst = os.listdir(main_dir)
        img_file_lst.sort()
        self.total_imgs = [ x for x in img_file_lst if x.endswith(".png") ]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image).float().cuda()
        # print("INP ", self.total_imgs[idx])
        return tensor_image

# https://www.reddit.com/r/DiscoDiffusion/comments/sxndpc/has_anyone_messed_around_with_the_eta_setting_by/
def diffuse(control, prompt, a_prompt, n_prompt, 
    num_samples=1, detect_resolution=512, ddim_steps=50, 
    guess_mode=False, strength=1, scale=10, seed=-1, 
    eta=0, save_memory=False):

    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map = input_image.copy()

        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        # control = torch.stack([control[0] for _ in range(num_samples)], dim=0)

        W, H = detect_resolution, detect_resolution
        # control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results






def diffuse_inpaint(input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur):
    # control, prompt, a_prompt, n_prompt, 
    # num_samples=1, detect_resolution=512, ddim_steps=50, 
    # guess_mode=False, strength=1, scale=10, seed=-1, 
    # eta=0, save_memory=False)
    with torch.no_grad():
        input_image = HWC3(input_image_and_mask['image'])
        input_mask = input_image_and_mask['mask']

        img_raw = resize_image(input_image, image_resolution).astype(np.float32)
        H, W, C = img_raw.shape

        mask_pixel = cv2.resize(input_mask[:, :, 0], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)

        mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)

        detected_map = img_raw.copy()
        detected_map[mask_pixel > 0.5] = - 255.0

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        mask = 1.0 - torch.from_numpy(mask_latent.copy()).float().cuda()
        mask = torch.stack([mask for _ in range(num_samples)], dim=0)
        mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

        x0 = torch.from_numpy(img_raw.copy()).float().cuda() / 127.0 - 1.0
        x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
        x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

        mask_pixel_batched = mask_pixel[None, :, :, None]
        img_pixel_batched = img_raw.copy()[None]

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, x0=x0, mask=mask)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)
        x_samples = x_samples * mask_pixel_batched + img_pixel_batched * (1.0 - mask_pixel_batched)

        results = [x_samples[i].clip(0, 255).astype(np.uint8) for i in range(num_samples)]
    return [detected_map.clip(0, 255).astype(np.uint8)] + results


def create_model(model_name):

    model = create_model(ctrlnet_path_for_plugin + f'./models/{model_name}.yaml').cpu()
    model.load_state_dict(load_state_dict(ctrlnet_path_for_plugin + './models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(ctrlnet_path_for_plugin + f'./models/{model_name}.pth', location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return model, ddim_sampler

# model_name = 'control_v11p_sd15_openpose'
# model_name = 'control_v11p_sd15_inpaint'
def gen_img(ip_fld, op_fld, prompt, a_prompt, n_prompt, 
    model_name='control_v11p_sd15_openpose', res=512, batch_size=5, seed=-1):
    os.makedirs(op_fld, exist_ok=True)

    transform = transforms.Compose([transforms.Resize(res),
                                 transforms.CenterCrop(res),
                                 transforms.ToTensor()])
    dataset = FolderLoad(ip_fld, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size, shuffle=False)
    
    model, ddim_sampler = create_model(model_type)

    for bat_idx, images in enumerate(dataloader):
        results = diffuse(images, prompt, a_prompt,
            n_prompt, detect_resolution=res, 
            num_samples=batch_size, seed=seed)
        for i in range(batch_size):
            op_file = "output"+str(bat_idx*batch_size+i+1).zfill(4)+".png"
            op_pth = os.path.join(op_fld, op_file)
            Image.fromarray(results[i]).save(op_pth)
            print("OP ",images.shape, torch.max(images), op_file, results[i].shape)

    return result
