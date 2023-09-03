# LoRA List: 
# foodphoto.safetensors - FOODPHOTO - https://civitai.com/models/45322/food-photography
# splashes_v.1.1.safetensors - SPLASH SPLASHES SPLASHING EXPLOSION EXPLODING - https://civitai.com/models/81619/splash
# add_detail.safetensors - any weight up/down to 2/-2
# ghibli_style_offset.safetensors = ghibli style - https://civitai.com/models/6526?modelVersionId=7657
# more_detail.safetensors - between 0.5 and 1 weight - https://civitai.com/models/82098/add-more-details-detail-enhancer-tweaker-lora
# ChocolateWetStyle.safetensors - ChocolateWetStyle - https://civitai.com/models/67132/chocolate-wet-style
# FoodPorn_v2.safetensors - foodporn - https://civitai.com/models/88717/foodporn
# LyCoRIS List
# fodm4st3r.safetensors - ART BY FODM4ST3R, FODM4ST3R, FODM4ST3R STYLE https://civitai.com/models/67467/rfktrs-food-master-hot-foods-edition
# TI
# negative_hand-neg.pt - NEGATIVE_HAND NEGATIVE_HAND-NEG - https://civitai.com/models/56519/negativehand-negative-embedding
# easynegative.safetensors - easynegative - https://civitai.com/models/7808/easynegative
# ng_deepnegative_v1_75t.pt - ng_deepnegative_v1_75t - https://civitai.com/models/4629/deep-negative-v1x
# fastnegative.pt - FastNegativeV2 - https://civitai.com/models/71961/fast-negative-embedding
# UnrealisticDream.pt - UnrealisticDream - https://civitai.com/models/72437?modelVersionId=77173
# BadDream.pt - BadDream - https://civitai.com/models/72437?modelVersionId=77169

# hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
# midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
# mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
# open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
# pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
# normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
# lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
# lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
# zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
# sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
# leres = LeresDetector.from_pretrained("lllyasviel/Annotators")

# # instantiate
# canny = CannyDetector()
# content = ContentShuffleDetector()
# face_detector = MediapipeFaceDetector()

# CKPT SD1.5:
# https://civitai.com/api/download/models/1449 - Cyborgdiffusion, Cyborgdiffusion style  - Cyborg Diffusion
# https://civitai.com/api/download/models/50 - nousr robot - Robo-Diffusion
# https://civitai.com/api/download/models/134065 - CLIP SKIP: 1 - epiCRealism


# LoRA SD1.5
# https://civitai.com/api/download/models/103778 - cyborg woman - EdobCyborgFemale
# https://civitai.com/api/download/models/8093 - robot - [LuisaP] 🤖 Humanoid Robots [1MB]
# https://civitai.com/api/download/models/73329 - robort - Robort
# https://civitai.com/api/download/models/92614 - weird future fashion, headress, plastic, veil, mask, helmet - Weird Future Fashion [LoRA]
# https://civitai.com/api/download/models/128027 - edgFut_clothing, wearing edgFut_clothing,  [futuristic headpiece|visor], - Futuristic - by EDG
# https://civitai.com/api/download/models/112577 - robot, cyborg, android - Future Diffusion Robot Youtuber
# https://civitai.com/api/download/models/108471 - candi buddha - Candi Buddha - Ancient Javanese Worship Place
# https://civitai.com/api/download/models/98413 - arabarmor - Arabian Warrior - Traditional Dress Series
# https://civitai.com/api/download/models/108428 - mosque - Great Mosque
# https://civitai.com/api/download/models/93666 - ashuralora - Ashura - The Magic Kingdom of India
# https://civitai.com/api/download/models/122982 - dwarapula - CLIP SKIP: 2 - Candi Hindu - Ancient Javanese Worship Place
# https://civitai.com/api/download/models/85513 - sandsculpturecd - LYCORIS - Realistic sand sculpture art style
# https://civitai.com/api/download/models/127815 - futubot - CLIP SKIP: 1 - Futuristicbot4
# https://civitai.com/api/download/models/97835 - neotech, sleek - NeoFuturistic Tech - World Morph

# common_lora_list = ["add_detail.safetensors", "more_detail.safetensors"]
common_lora_list = ["more_details.safetensors"] # "add_detail.safetensors",
commom_ti_list = ["BadDream.pt", "easynegative.safetensors","ng_deepnegative_v1_75t.pt",  # "fastnegative.pt", 
        "negative_hand-neg.pt", "UnrealisticDream.pt", "CyberRealistic_Negative-neg.pt"]
common_n_prompt = "BadDream, easynegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.3, \
        negative_hand, negative_hand-neg, UnrealisticDream, CyberRealistic_Negative, CyberRealistic_Negative-neg"
theme_to_model_map = {
    "people": {
        "base_model": "nextphoto_v30.safetensors", # "lifeLikeDiffusionEthnicitiesSupportedNative_lifeLikeDiffusionV30.safetensors", # "SG161222/Realistic_Vision_V4.0", # "nextphoto_v30.safetensors", # "cyberdelia/CyberRealistic" , #
        "vae": None,
        "lora_list":[],
        "ti_list":[],
        "n_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, \
            cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, \
            ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
            mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
            disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, \
            fused fingers, too many fingers, long neck, BadDream, UnrealisticDream",
        "prompt":"RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    },
    "food":{
        "base_model": "runwayml/stable-diffusion-v1-5",
        "vae": None,
        "lora_list":["foodphoto.safetensors", "FoodPorn_v2.safetensors", "splashes_v.1.1.safetensors",],
                    # "ghibli_style_offset.safetensors",],#"ChocolateWetStyle.safetensors", "foodStyle.safetensors"
        "ti_list":[],
        "n_prompt": "",
        "lora_prompts":["foodphoto"], 
        "trigger_word":["splash", " splashes", "splashing", "explosion", "exploding", "chocolatewetstyle", "foodstyle", "foodstyle"]
    },
     "foodpeople":{
        "base_model": "SG161222/Realistic_Vision_V5.1_noVAE", #"SG161222/Realistic_Vision_V4.0",
         # # "./share_vol/models/base_mdl/nextphoto_v30.safetensors", "SG161222/Realistic_Vision_V4.0", runwayml/stable-diffusion-v1-5","SG161222/Realistic_Vision_V5.1_noVAE", "./share_vol/models/base_mdl/nextphoto_v30.safetensors",
         "vae": None,
        "lora_list":["foodphoto.safetensors", "FoodPorn_v2.safetensors", "splashes_v.1.1.safetensors","foodStyle.safetensors"],
                    # "ghibli_style_offset.safetensors",],#"ChocolateWetStyle.safetensors", 
        "ti_list":[],
        # "n_prompt": "(worst quality:0.8), cartoon, halftone print, burlap, (verybadimagenegative_v1.3:0.3), (cinematic:1.2), \
        # (surreal:0.8),(modernism:0.8), (art deco:0.8), (art nouveau:0.8)",
         "n_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, \
             cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, \
             ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
             mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
             disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, \
             fused fingers, too many fingers, long neck, BadDream, UnrealisticDream",
        "trigger_word":["foodphoto", "foorporn", "photo", "photograph"]
    },
    "people_lifelike": {
        "base_model": "./share_vol/models/base_mdl/lifeLikeDiffusionEthnicitiesSupportedNative_lifeLikeDiffusionV30.safetensors", #"cyberdelia/CyberRealistic", 
        # https://huggingface.co/stabilityai/sd-vae-ft-mse, https://huggingface.co/stabilityai
        "vae": "stabilityai/sd-vae-ft-mse",# "stabilityai/sd-vae-ft-ema", #"./share_vol/models/vae/vae-ft-mse-840000-ema-pruned.safetensors", # "stabilityai/sd-vae-ft-ema",
        "clip_skip":1,
        "lora_list":["Futuristicbotv.2.safetensors"],
        "ti_list":[],
        "n_prompt": "small-pupils, small-eyes, closed-eyes, ugly-eyes, multiple-navel, multiple belly-button, ((lazy-eyes)) (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, closed-eyes, ugly-eyes, ((lazy-eyes))",
        "prompt":"RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    },
    "cyborg": {
        "base_model": "./share_vol/models/base_mdl/cyborgDiffusion_cyborgDiffusionV1.ckpt", #
        "vae" : None,
        "lora_list":[],
        "ti_list":[],
        "n_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, \
            cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, \
            ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
            mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, \
            disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, \
            fused fingers, too many fingers, long neck, BadDream, UnrealisticDream",
        "prompt":"((instagram photo)), ((Full Shot)) full body photo of 28 year old beautiful ((woman superhero costume)) strong girlboss in (( avatar the last airbender movie)) (high detailed skin:1.2) 8k uhd, dslr, soft lighting, high quality, film grain ((amateur photo))"
    },
}

# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe"]
ctrl_type_to_processor_id = {
    "pidiedge" : "softedge_pidinet", "hededge" : "softedge_hedsafe",
    # "inpaint" : [1.0, "lllyasviel/control_v11p_sd15_inpaint"],
    "canny" : "canny",
    "openpose" : "openpose_full",
    "midasdepth" : "depth_midas",
    "zoedepth" : "depth_zoe",
}
model_name_mapper = {
    "pidiedge" : [1.0, "lllyasviel/control_v11p_sd15_softedge"],
    "hededge" : [1.0, "lllyasviel/control_v11p_sd15_softedge"],
    "inpaint" : [1.0, "lllyasviel/control_v11p_sd15_inpaint"],
    "canny" : [1.0, "lllyasviel/control_v11p_sd15_canny"],
    "openpose" : [1.0, "lllyasviel/control_v11p_sd15_openpose"],
    "midasdepth" : [1.0, "lllyasviel/control_v11f1p_sd15_depth"],
    "zoedepth" : [1.0, "lllyasviel/control_v11f1p_sd15_depth"],
} 


