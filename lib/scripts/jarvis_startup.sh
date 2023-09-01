#!/bin/sh
# export PIP_CONFIG_FILE=/workspace/source_code/pip.conf 

wget https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -O /usr/local/bin/yt-dlp
chmod a+rx /usr/local/bin/yt-dlp
# System level installation
apt-get update && apt-get install ffmpeg libsm6 libxext6 zip  -y
apt install python3-pip python-is-python3 -y
apt-get install iputils-ping net-tools -y
pip install eventlet python-socketio einops scipy moviepy ffmpegio imageio[ffmpeg] timm==0.6.13 basicsr annotator mediapipe omegaconf jupyter-archive
# Node Installation
cd ./test/pipeline/
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs
npm install supervisor socket.io socket.io-client 
npm install supervisor concurrently -g 
cd ../../

# GPU installation
pip install xformers safetensors insightface==0.7.3 onnxruntime gdown controlnet-aux
pip install accelerate transformers==v4.31.0 diffusers accelerate==0.20.3
# cd ~/source_code/mdl_inf_jp3d/

## audio
pip install git+https://github.com/suno-ai/bark.git
pip install -U audiocraft  # stable release

cd ext_lib/SadTalker
pip install -r requirements.txt
cd ../../

export HF_HOME="./share_vol/models/hf_cache/"
python sio_api.py