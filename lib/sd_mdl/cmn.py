import os
import torch
import random

from diffusers import AutoencoderKL
model_base_path = "./share_vol/models/base_mdl/"
lora_base_path = "./share_vol/models/lora/"
ti_base_path = "./share_vol/models/ti/"

def load_vae(vae_path):
    # Load VAE:
    if(vae_path and os.path.isfile(vae_path)):
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16).to("cuda")
    elif(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to("cuda")
    print("VAE Path ", vae_path)#vae
    return vae

def seed_to_generator(seed):
    if seed == -1:
        seed = random.randint(0, 65535)
    return seed, torch.Generator(device="cpu").manual_seed(seed)


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
