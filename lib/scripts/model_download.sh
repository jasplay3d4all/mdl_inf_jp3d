mkdir ./share_vol/models
mkdir ./share_vol/models/base_mdl
mkdir ./share_vol/models/lora
mkdir ./share_vol/models/lycoris
mkdir ./share_vol/models/roop
mkdir ./share_vol/models/ti
mkdir ./share_vol/models/hf_cache
mkdir ./share_vol/models/vae


# "BadDream.pt", "UnrealisticDream.pt", "CyberRealistic_Negative-neg.pt"
# "easynegative.safetensors","ng_deepnegative_v1_75t.pt",  # "fastnegative.pt", "negative_hand-neg.pt", 
cd ./share_vol/models/ti
wget https://civitai.com/api/download/models/77169 --content-disposition
wget https://civitai.com/api/download/models/77173 --content-disposition
wget https://civitai.com/api/download/models/82745 --content-disposition
wget https://civitai.com/api/download/models/9208 --content-disposition
wget https://civitai.com/api/download/models/5637 --content-disposition
wget https://civitai.com/api/download/models/60938 --content-disposition
cd ../../../

# "add_detail.safetensors", "more_detail.safetensors"
# "foodphoto.safetensors", "FoodPorn_v2.safetensors", "splashes_v.1.1.safetensors",
# "ghibli_style_offset.safetensors",],#"ChocolateWetStyle.safetensors"
cd ./share_vol/models/lora
wget https://civitai.com/api/download/models/49946 --content-disposition
wget https://civitai.com/api/download/models/91874 --content-disposition
wget https://civitai.com/api/download/models/87153 --content-disposition
wget https://civitai.com/api/download/models/7657 --content-disposition
wget https://civitai.com/api/download/models/71765 --content-disposition
wget https://civitai.com/api/download/models/94406 --content-disposition
wget https://civitai.com/api/download/models/72109 --content-disposition
cd ../../../

cd ext_lib/SadTalker
bash scripts/download_models.sh
cd ../../

cd ./share_vol/models/base_mdl
# Cyborgdiffusion, Cyborgdiffusion style  - Cyborg Diffusion
wget https://civitai.com/api/download/models/1449 --content-disposition
# Life Like Diffusion (Ethnicities supported - Native American, Desi Indian, Arab, Hispanic, Latino, South Asian, Black / African, Turkish, Korean & Chinese )
wget https://civitai.com/api/download/models/136151 --content-disposition
cd ../../../

# Download all the loras required for cyborg related and SD1.5
cd ./share_vol/models/lora
wget https://civitai.com/api/download/models/103778 --content-disposition
wget https://civitai.com/api/download/models/8093 --content-disposition
wget https://civitai.com/api/download/models/73329 --content-disposition
wget https://civitai.com/api/download/models/92614 --content-disposition
wget https://civitai.com/api/download/models/128027 --content-disposition
wget https://civitai.com/api/download/models/112577 --content-disposition
wget https://civitai.com/api/download/models/108471 --content-disposition
wget https://civitai.com/api/download/models/98413 --content-disposition
wget https://civitai.com/api/download/models/108428 --content-disposition
wget https://civitai.com/api/download/models/93666 --content-disposition
wget https://civitai.com/api/download/models/122982 --content-disposition
wget https://civitai.com/api/download/models/85513 --content-disposition
wget https://civitai.com/api/download/models/127815 --content-disposition
wget https://civitai.com/api/download/models/97835 --content-disposition
cd ../../../

# Download all the loras required for cyborg related and SDXL
cd ./share_vol/models/lora
wget https://civitai.com/api/download/models/129723 --content-disposition
wget https://civitai.com/api/download/models/138640 --content-disposition
wget https://civitai.com/api/download/models/128068 --content-disposition
wget https://civitai.com/api/download/models/139281 --content-disposition
wget https://civitai.com/api/download/models/139401 --content-disposition
wget https://civitai.com/api/download/models/137876 --content-disposition
wget https://civitai.com/api/download/models/138654 --content-disposition
wget https://civitai.com/api/download/models/128403 --content-disposition
wget https://civitai.com/api/download/models/130743 --content-disposition
wget https://civitai.com/api/download/models/136677 --content-disposition
cd ../../../

# Download VAE
cd ./share_vol/models/vae
wget https://civitai.com/api/download/models/136151?type=VAE --content-disposition
cd ../../../
