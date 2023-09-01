
# Controlnet SDXL
# SDXL Inpainiting
# SDXL Inpainting using refiner: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#inpainting 
# Img2imgpipeline
# LoRA SDXL - 
# https://github.com/huggingface/diffusers/pull/4287/#issuecomment-1655936838 
# encode_prompt(lora_scale)
# https://github.com/huggingface/diffusers/issues/4348 - MulitLora load not supported now
# SDXL speedup: [Done]
# pipe.enable_model_cpu_offload()
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# SDXL Ensemble of Expert Denoisers
# Difference between InpaintRefiner and base refiner or using other refinement ideas
# https://pypi.org/project/invisible-watermark/
# https://huggingface.co/diffusers - sdxl different controlnet like canny, depth, small, medium etc?

model_name_mapper = {
    "canny_small":[0.5, "diffusers/controlnet-canny-sdxl-1.0-small"],
    "canny_mid":[0.5, "diffusers/controlnet-canny-sdxl-1.0-mid"],
    "canny":[0.5, "diffusers/controlnet-canny-sdxl-1.0"],
    "depth_small":[0.5, "diffusers/controlnet-depth-sdxl-1.0-small"],
    "depth_mid":[0.5, "diffusers/controlnet-depth-sdxl-1.0-mid"],
        "depth":[0.5, "diffusers/controlnet-depth-sdxl-1.0-small"], #"diffusers/controlnet-depth-sdxl-1.0"],
}
ctrl_type_to_processor_id = {
            # "pidiedge" : "softedge_pidinet", "hededge" : "softedge_hedsafe",
            "canny" : "canny", "canny_small" : "canny", "canny_mid" : "canny",
            # "openpose" : "openpose_full",
            "depth" : "depth_midas", "depth_small" : "depth_midas", "depth_mid" : "depth_midas",
            # "depth" : "depth_zoe",
}

# Lora SDXL
# https://civitai.com/api/download/models/129723 - cyborg style, cyborg, android - Cyborg Style SDXL | Goofy Ai
# https://civitai.com/api/download/models/138640 - holding sword, armor, holding weapon, hailoknight - CLIP SKIP: 1 - XL Fantasy warriors - by HailoKnight
# https://civitai.com/api/download/models/128068 - edgFut_clothing, wearing edgFut_clothing, neons, electric circuits - Futuristic XL - by EDG
# https://civitai.com/api/download/models/139281 - Eagle - Eagle
# https://civitai.com/api/download/models/139401 - Pakistani dress - CLIP SKIP: 2 - Pakistani Studio - Female Edition - SDXL
# https://civitai.com/api/download/models/137876 - desert, rock, weed - desert_SDXL
# https://civitai.com/api/download/models/138654 - Saree, Indian Saree - Desi Style Sarees [SDXL]
# https://civitai.com/api/download/models/128403 - mir - MIR效果图风格 MIRStyle SDXL
# https://civitai.com/api/download/models/130743 - no words - YQXL_CrvArchitecture
# https://civitai.com/api/download/models/136677 - smxl - Snow mountain XL

theme_to_model_map = {
    "sdxl_base": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0" , 
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "lora_list":[],
        "use_refiner":True,
        "n_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, \
            cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, \
            ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
            mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
            disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, \
            fused fingers, too many fingers, long neck, BadDream, UnrealisticDream",
        "prompt":"RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    }
}