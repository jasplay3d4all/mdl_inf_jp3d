Folder Structure:
----------------
ext_lib - List of all external modules which need to be imported by all the pipeline modules for inference pipelines 
models - Large files to be used by inference modules
pipeline - connecting different modules to achieve required outputs
plugin - interface of interacting with external libraries
data_io - Input and output for input and output images
Folders to be ignored: ./library, ./models, ./data_io

checkout the below modules
--------------------------
git clone https://github.com/lllyasviel/ControlNet-v1-1-nightly
git clone https://github.com/huggingface/diffusers
git clone https://github.com/s0md3v/roop 
git clone https://github.com/Winfredy/SadTalker.git

Not used but experimented:
-------------------------
git clone https://github.com/IDEA-Research/OSX 
git clone https://github.com/Stability-AI/lora_accelerated 
git clone https://github.com/cloneofsimo/lora 
git clone https://github.com/zyddnys/sd_animation_optical_flow
git clone https://github.com/PruneTruong/DenseMatching 
git clone https://github.com/haofeixu/gmflow 
git clone https://github.com/camenduru/Text-To-Video-Finetuning 

make the following folders:
--------------------------
articulated_motion
optical_flow
pose_est_3d

models used:
-----------
- musicgen: /home/.cache/torch/hub/checkpoints/f79af192-61305ffc49.th
- bark: 13G ~/.cache/suno/bark_v0/* -> ../../huggingface/hub/models--suno--bark/blobs/

Installation:
-------------

# System level installation
wget https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -O /usr/local/bin/yt-dlp
chmod a+rx /usr/local/bin/yt-dlp
apt-get update && apt-get install ffmpeg libsm6 libxext6 zip  python3-pip python-is-python3 -y
apt-get install iputils-ping net-tools -y
pip install eventlet python-socketio einops scipy moviepy ffmpegio controlnet-aux timm basicsr requests mediapipe omegaconf jupyter-archive
<!-- pip install git+https://github.com/jarvislabsai/jlclient.git -->

# Node Installation
<!-- https://github.com/nodesource/distributions#debinstall -->
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs
npm install supervisor socket.io socket.io-client concurrently -g

# GPU installation
pip install xformers safetensors insightface==0.7.3 onnxruntime gdown
<!-- pip install mmcv mmdet # mmcv takes a long time? -->
pip install  ./ext_lib/diffusers/ 
## audio
pip install git+https://github.com/suno-ai/bark.git
pip install -U audiocraft  # stable release
pip install git+https://github.com/huggingface/transformers.git accelerate==0.20.3
## SadTalker
cd ext_lib/articulated_motion/SadTalker
pip install -r requirements.txt
cd ../../../
## inswapper
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
<!-- pip install onnxruntime basicsr face_recognition
pip install git+https://github.com/sajjjadayobi/FaceLib.git -->
cd ext_lib/articulated_motion/inswapper
pip install -r requirements.txt
cd ../../../

## roop
cd ext_lib/articulated_motion/roop
pip install -r requirement.txt
apt-get install git-lfs
cd ../../../

# TTS
pip install TTS mecab-python3 unidic-lite

export HF_HOME="./share_vol/models/hf_cache/huggingface/"

# Install using scripts
sh ./lib/scripts/jarvis_startup.sh
export HF_HOME="./share_vol/models/hf_cache/"
<!-- https://stackoverflow.com/questions/41650158/how-to-change-pip-installation-path -->
export PIP_CONFIG_FILE=/workspace/source_code/pip.conf 

# Autostart runpod instance
bash -c 'cd /workspace/source_code/mdl_inf_jp3d/ && sh ./lib/scripts/jarvis_startup.sh & && cd / && ./start.sh'
bash -c 'cd /workspace/source_code/mdl_inf_jp3d/; sh ./lib/scripts/jarvis_startup.sh & cd / && ./start.sh'
<!-- https://stackoverflow.com/questions/49649082/how-to-run-background-process-with-bash-c -->
probably have separate scripts for nodejs and gpu
Docker command:
<!-- https://docs.runpod.io/docs/customize-a-template
https://github.com/runpod/runpodctl/blob/main/doc/runpodctl_start.md Not working fully -->
bash -c 'cd /workspace/source_code/mdl_inf_jp3d && sh ./lib/scripts/jarvis_startup.sh && /start.sh'


# First time/ One Time scripts
sh ./lib/scripts/checkout.sh
sh ./lib/scripts/model_download.sh


# Run command
concurrently "supervisor --exec node ./test/pipeline/product_paradise_test.js" "supervisor --exec python sio_api.py"

<!-- concurrently "node ./test/pipeline/product_paradise_test.js" "python sio_api.py" -->
concurrently "node ./test/pipeline/ad_test.js" "python sio_api.py"
concurrently "python sio_api.py" "cd ./test/pipeline/; node single_state/single_state_test.js" 

## Run using backend and frontend
cd .. or cd ~/source_code/
concurrently "cd website_jp3d/;supervisor --exec node node_lib/main.js" "cd mdl_inf_jp3d/; supervisor --exec python sio_api.py"
concurrently "cd website_jp3d/;node node_lib/main.js" "cd mdl_inf_jp3d/; python sio_api.py"

pip install chumpy numpy==1.23 ultralytics
cd pose_est_3d/OSX/
bash install.sh # mmpose installaion takes a long time. But is cached.
    - Does not install trimeshopen so commented out
    - Install pytorch required version and then install the mmcv: 

## Github checkin
In VS code bottom left there is a account icon above setting account and it is logged into the github.


install models from civitai:
----------------------------
go to civitai / instert curl https://chrome.google.com/webstore/detail/curlwget/dgcfkhmmpcmkikfmonjcalnjcmjcjjdn  - plugin
using that get the download command prompt 
go to source folder cd source_code/mdl_inf_jp3d/share_vol/models/


./ext_lib/faceswap/lora/.
Code Backup:
------------
cd /mdl_inf_jp3d/../
zip -r mdl_inf_jp3d_v0.0.1.zip ./mdl_inf_jp3d -x mdl_inf_jp3d/ext_lib/**\* mdl_inf_jp3d/share_vol/**\* '*__pycache__*/*' '*.DS_Store*'
zip -r mdl_inf_jp3d_v0.0.9.zip ./mdl_inf_jp3d -x mdl_inf_jp3d/ext_lib/**\* mdl_inf_jp3d/share_vol/**\* '*node_modules*/*' '*__pycache__*/*' '*.DS_Store*' '*.ipynb_checkpoints*/*' '*.jpeg*' '*.png*' '*.jpg*'


zip -r website_jp3d_v0.0.1.zip ./website_jp3d -x website_jp3d/.github/**\*  '*__pycache__*/*' '*.DS_Store*'
zip -r website_jp3d_v0.0.1.zip ./website_jp3d -x website_jp3d/.github/**\* website_jp3d/node_modules/**\*  '*__pycache__*/*' '*.DS_Store*'

Deployment:
----------

Features:
---------
- ability to publish to different streams like youtube using api?
- Using QR code for the images and ads?
- Just single frame face swapping for professional and imaginary photo graphs
- generate different pipelines using chatgpt
- deploy on runpod its cheap
- sdxl for upsampling and denoiser
- sdxl for nerf and thin spline motion retargetting
- train sdxl for lora
- Roop based video face swapping
- build color consistency and audio consistency across video
    - create a consistent music by using longterm music across scenes
- using 3D smplx pose estimation to generate images - mimic dance
- Connect cfg from upper module to lower module using wrkflow_lnk
- upload file to tmporg and share path link
- using https://github.com/coqui-ai/TTS - for text to speech and text to music
- 2D warping features
- 3D warping features - using zoe depth
- Development - plugins, 
    - core block - plugin interact to create a core block [one GPU execution]
    - pipeline - core blocks linked to form a pipeline []
- Textual Inversion and lora based celebrity search and adding it to model - use civitai api
- Use brisk and fast based sound to make the video to give a fast playing like in tiktok
- adding transition music and transition videos
- [x]using moviepy for editor like transitions and audio combination etc
- [x] generate a ad image set for a given input img and logo - logo to img - img with inpaint for all output formats
- [x] controlnet based logo edge or depth inpainting or just images
- [x] pass a lora list instead of hardcoding
- [x] Ability to test individual sections instead of the entire pipeline
- [x] pipeline for music and speech from text

Probable Errors:
- ping_timeout=60000, ping_interval=60000. Sometimes it can cause issues due to long delays
- Make sure you use the right LoRa like remove the chocolaty lora for food
- Explore alpha weight and TI and proper LorA words to add to input
- Optimize the filedump and read for video creation. Try using numpy array instead? or streaming video?
- Optimize for model load and gpu switching on and off
- How to handle the progress and intermediate progress
- add zoom in and zoom out
- seems like if you run for a long time the system seems to slow down and the loading takes a long time
- test anno_plg for controlnet
