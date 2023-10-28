Download StableDiffusion Model:https://huggingface.co/runwayml/stable-diffusion-v1-5

mkdir stable-diffusion-v1-5
cd stable-diffusion-v1-5
mkdir vae tokenizer unet text_encoder feature_extractor scheduler
cd vae
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/config.json
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin
cd ../tokenizer
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/special_tokens_map.json
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_config.json
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json
cd ../text_encoder
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json
cd ../unet
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json
cd ../feature_extractor
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/feature_extractor/preprocessor_config.json
cd ../scheduler
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/scheduler/scheduler_config.json
cd ../

# Make pipe.safety_checker = None (This way it does not require safety_checker)

Installation:
pip install diffusers==0.11.1 transformers==4.25.1 imageio==2.27.0 decord==0.6.0 einops omegaconf safetensors
pip install xformers==0.0.16 # This installs torch again. Try directly installing xformers

ControlNet based:
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install accelerate==0.20.3 gdown xformers==0.0.21 diffusers==0.20.2 transformers==4.32.1 controlnet-aux==0.0.6 imageio==2.27.0 imageio[ffmpeg] decord==0.6.0 einops omegaconf safetensors
pip install 'mediapipe'
    - gradio
    - wandb
Models for controlNet: https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main
ControlNet based support seems to have memory problems. So try increasing memory??


stable diffusion: https://huggingface.co/runwayml/stable-diffusion-v1-5/

TemporalNet:
https://huggingface.co/CiaraRowles/controlnet-temporalnet-sdxl-1.0/blob/main/runtemporalnetxl.py
https://huggingface.co/CiaraRowles/TemporalNet/tree/main 

https://stable-diffusion-art.com/video-to-video/


python lib/vdo_temporalnet.py --video_path ../vdo_diff/data/ctrl_vdo/video.mp4 --frames_dir ../vdo_diff/data/temporalnet/extracted/ --output_frames_dir ../vdo_diff/data/temporalnet/output/ --init_image_path ../vdo_diff/data/temporalnet/frame0000.png

Three major libraries:
- ControledAnimateDiff, AnimateDiff with input image and just plain AnimateDiff.
- https://github.com/Logeswaran123/Stable-Diffusion-Playground - This has features for video generation like pan left/pan right, warping etc
- The script to pull the models is above in this file