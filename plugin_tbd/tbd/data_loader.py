from torchvision import datasets, transforms
import torch
import os 
from PIL import Image
import  numpy as np
import cv2
import einops
import random
# import imageio

def read_img_np(img_loc):
    image = Image.open(img_loc).convert("RGB")
    return np.array(image)

class FolderLoad:
    def __init__(self, main_dir, model_name_list, res, mask_blur=1.0):
        self.model_name_list = model_name_list
        self.main_dir = main_dir
        self.mask_blur = mask_blur
        img_loc = os.path.join(self.main_dir, "img_ext")
        img_file_lst = os.listdir(img_loc)
        img_file_lst.sort()
        self.total_imgs = [ x for x in img_file_lst if x.endswith(".png") ]

        self.transform = transforms.Compose([transforms.Resize(res),
                                transforms.CenterCrop(res),
                                transforms.ToTensor()])


    def __len__(self):
        return len(self.total_imgs)

    def get_ref_img(self, idx, is_np=True):
        ref_img_loc = os.path.join(self.main_dir, "img_ext", self.total_imgs[idx])
        ref_img = self.transform(Image.open(ref_img_loc).convert("RGB"))
        if(is_np):
            ref_img = einops.rearrange(ref_img, 'c h w -> h w c').numpy()
            ref_img = (ref_img*255).astype(np.uint8)
        return ref_img

    def get_inpaint_mask(self, idx, img_mask=None):
        # https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint
        seg_loc = os.path.join(self.main_dir, "per_seg", self.total_imgs[idx])
        mask_pixel = self.transform(Image.open(seg_loc).convert("L")).repeat(3, 1, 1)

        # # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        # mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), self.mask_blur)
        # print("Control image ", image.shape, mask_pixel.shape)
        # print("Min and max ", torch.max(mask_pixel), \
        #     torch.min(image), torch.max(image))
        # image = ref_img.copy()
        image = self.get_ref_img(idx, is_np=False)        
        image[mask_pixel > 0.5] = -1.0
        # image[mask_pixel > -1.0] = -1.0
        mask_pixel = einops.rearrange(mask_pixel, 'c h w -> h w c').numpy()
        return image[None, ...], mask_pixel

    def get_inpaint_dpth_mask(self, idx, bg_img, use_zoe=True):
        # https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint
        seg_loc = os.path.join(self.main_dir, "per_seg", self.total_imgs[idx])
        mask_pixel = self.transform(Image.open(seg_loc).convert("L")).repeat(3, 1, 1)

        if(use_zoe):
            dpt_loc = os.path.join(self.main_dir, "zoe_dpth", self.total_imgs[idx])
        else:
            dpt_loc = os.path.join(self.main_dir, "midas_dpth", self.total_imgs[idx])
        dpt_pixel = self.transform(Image.open(dpt_loc).convert("L")).repeat(3, 1, 1)

        fg_avg_dpt = torch.mean(dpt_pixel[mask_pixel > 0.5])
        bg_avg_dpt = torch.mean(dpt_pixel[mask_pixel <= 0.5])
        dpt_thresh = (fg_avg_dpt + bg_avg_dpt) / 2

        dpt_msk_pxl = torch.ones_like(mask_pixel)
        dpt_msk_pxl[dpt_pixel > dpt_thresh] = 1.0
        dpt_msk_pxl[dpt_pixel <= dpt_thresh] = 0.0


        # # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        # mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), self.mask_blur)
        # print("Control image ", image.shape, mask_pixel.shape)
        
        bg_img = self.transform(bg_img.copy())
        bg_img[dpt_msk_pxl > 0.5] = -1.0
        # print("Img mask Min and max ", torch.max(bg_img), torch.min(bg_img), torch.max(dpt_msk_pxl))
        return bg_img[None, ...], dpt_msk_pxl[None, ...]

    def get_img_lst(self, idx):
        tensor_image_list = []
        for model_name in self.model_name_list:
            img_loc = os.path.join(self.main_dir, model_name, self.total_imgs[idx])
            image = Image.open(img_loc).convert("RGB")
            image = self.transform(image)
            # image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            # image = torch.from_numpy(image)
            image = image[None, ...]
            # image = einops.rearrange(image, 'b h w c -> b c h w')
            tensor_image_list.append(image) #
        # print("INP ", self.total_imgs[idx])
        return tensor_image_list