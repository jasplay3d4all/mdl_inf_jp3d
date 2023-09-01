
import os
import sys
import utils_plg.linux_cmd as lc 

# import ctrlnt_plg.plugin as cnplg Will most probably remove this
# import ctrlnt_plg.hf_ctrl_plg as hfcnplg
# import ctrlnt_plg.anno_plg as annoplg # Controlnet plugin

# import mmdet_plg.pan_seg_plg as psplg
# from hf_plg.sd_plg import sd_model
# from hf_plg.sd_ref_plg import sd_model as sd_mdl_ref
# from opt_flw_plg.den_mat_plg.pdcnet_plus import PDCNetPlus
# from hf_plg.sd_inpaint_plg import sd_model as sd_mdl_inpaint
from pose_est_plg.osx_plg import osx_pose_est as pose_est



def mimic_dance(url, work_folder, fg_prompt, bg_prompt, n_prompt, seed=-1):
    os.makedirs(work_folder, exist_ok=True)
    # Download video from youtube
    video_folder = "yt_vdo"
    vdo_fld_pth = os.path.join(work_folder, video_folder)
    # lc.dld_vdo(url, vdo_fld_pth)
    # Split video into smaller clips
    vdo_splt_fld = "splt_vdo"
    time = "00:00:10"
    vdo_splt_fld_pth = os.path.join(work_folder, vdo_splt_fld)
    input_file = os.path.join(vdo_fld_pth, "video.mp4") 
    lc.split_vdo(input_file, vdo_splt_fld_pth, time)
    # Convert video to image clips
    img_ext_fld = "img_ext"
    fps = "30"
    img_ext_fld_pth = os.path.join(work_folder, img_ext_fld)
    # input_file = os.path.join(vdo_splt_fld_pth, "output0016.mp4")
    # input_file = os.path.join(vdo_splt_fld_pth, "output0015.mp4")
    input_file = os.path.join(vdo_splt_fld_pth, "output0000.mp4")

    lc.ext_imgs(input_file, img_ext_fld_pth, fps)

    # Extract pose from each image
    img_pose_fld = "openpose"
    img_pose_fld_pth = os.path.join(work_folder, img_pose_fld)
    # annoplg.gen_pose(img_ext_fld_pth, img_pose_fld_pth)

    # extract 3D pose for each image
    img_3dpose_fld = "smplxpose"
    img_3dpose_fld_pth = os.path.join(work_folder, img_3dpose_fld)
    pose_smpl_est = pose_est()
    # pose_smpl_est.gen_pose_lst(img_ext_fld_pth, img_3dpose_fld_pth)
    img_pose_fld = "openpose"
    img_pose_fld_pth = os.path.join(work_folder, img_pose_fld)
    pose_smpl_est.draw_pose_lst(work_folder, img_pose_fld_pth)

    # Extract person from background
    # pan_seg_fld = "pan_seg"
    # pan_seg_fld_pth = os.path.join(work_folder, pan_seg_fld)
    # psplg.gen_pan_seg_img(img_ext_fld_pth, pan_seg_fld_pth)

    # Extract person from background
    per_seg_fld = "per_seg"
    per_seg_fld_pth = os.path.join(work_folder, per_seg_fld)
    # lc.bg_remover(img_ext_fld_pth, per_seg_fld_pth)

    # Extract depth from each image
    img_midas_fld = "midas_dpth"
    img_midas_fld_pth = os.path.join(work_folder, img_midas_fld)
    img_zoe_fld = "zoe_dpth"
    img_zoe_fld_pth = os.path.join(work_folder, img_zoe_fld)
    # annoplg.gen_depth(img_ext_fld_pth, img_midas_fld_pth)
    # annoplg.gen_depth(img_ext_fld_pth, img_zoe_fld_pth, is_midas=False)

    img_softedge_fld = "softedge_PIDI"
    img_softedge_fld_pth = os.path.join(work_folder, img_softedge_fld)
    # annoplg.gen_softedge(img_ext_fld_pth, img_softedge_fld_pth, det='PIDI')
    # annoplg.gen_softedge(img_ext_fld_pth, img_softedge_fld_pth, det='HED')

    # img_of_fld = "opt_flw"
    # img_of_fld_pth = os.path.join(work_folder, img_of_fld)
    # of_net = PDCNetPlus("../models/DenseMatching/PDCNet_plus_megadepth.pth.tar")
    # of_net.gen_msk_all_imgs(img_ext_fld_pth, img_of_fld_pth, conf_thres=0.8)


    # Generate images using pose
    img_gen_fld = "img_gen"
    img_gen_fld_pth = os.path.join(work_folder, img_gen_fld)
    # cnplg.gen_img(img_pose_fld_pth, img_gen_fld_pth, 
    #     prompt, a_prompt, n_prompt, res=512, seed=seed)

    model_name_list = [  "softedge_PIDI", ] # "zoe_dpth", ] #"openpose", ]#, ] # , "midas_dpth"] # 
    # hfcnplg.gen_img(work_folder, img_gen_fld_pth, 
    #     prompt, n_prompt, model_name_list, 
    #     res=512, seed=seed)
    
    # prompt = fg_prompt + bg_prompt
    # sd_plg = sd_model(model_name_list, prompt, n_prompt, is_inpaint=True)
    # sd_plg.gen_img_lst(work_folder, img_gen_fld_pth, res=512, seed=-1, is_t2i=False)
    # # sd_plg.gen_vdo(work_folder, img_gen_fld_pth, res=512, seed=seed)
    
    # sd_plg = sd_mdl_ref(model_name_list, prompt, n_prompt, is_inpaint=False, no_ctrl=False)
    # sd_plg.gen_img_lst(work_folder, img_gen_fld_pth, res=512, seed=seed)


    # sd_plg = sd_mdl_inpaint(model_name_list, fg_prompt, bg_prompt, n_prompt, use_ctrl=True)
    # sd_plg.gen_img_lst(work_folder, img_gen_fld_pth, res=512, seed=seed)

    # Generate video from images generated
    gen_vid_fld = "gen_vid"
    gen_vid_fld_path = os.path.join(work_folder, gen_vid_fld)
    framerate = "15"
    lc.make_vdo(img_gen_fld_pth, gen_vid_fld_path, framerate)

if __name__ == "__main__":
    # url = "https://www.youtube.com/watch?v=E8zMZKRdxsM&ab_channel=TikTokVibe"
    url = "https://www.youtube.com/watch?v=F3_t-CVYP3Q"
    work_folder = "../output/ttv/"

    # prompt = "Tom and jerry, fantasy, intricate, elegant, highly detailed, digital painting, \
    #     artstation, concept art, matte, sharp focus, illustration, hearthstone, art by artgerm \
    #     and greg rutkowski and alphonse mucha, 8k"
    # prompt = "Tom and jerry cartoon" #"Anime cartoons"
    # prompt = "BTS boys, fantasy, fantasy magic, intricate, sharp focus, illustration, \
    #     highly detailed, digital painting, concept art, matte, Artgerm and Paul lewin \
    #     and kehinde wiley, masterpiece"
    # prompt = "scarlett, highly detailed, artstation, sharp focus, 8K, " #art by art by artgerm and greg rutkowski and edgar maxence."
    # prompt = "space girl| standing alone on hill| centered| detailed gorgeous face| anime style| key visual| intricate detail| highly detailed| breathtaking| vibrant| panoramic| cinematic| Carne Griffiths| Conrad Roset| ghibli"
    # prompt = "photo of scarlett as a beautiful female model, georgia fowler, beautiful face, with short dark brown hair, in cyberpunk city at night. She is wearing a leather jacket, black jeans, dramatic lighting, (police badge:1.2)"
    # prompt = "photo of scarlett,  8k, detailed face, photorealistic, beautiful hands, masterpiece, best quality, sharp focus, highres, highly detailed, ((full body)), (RAW photo, best quality), (realistic, photo-realistic:1.2), (high detailed skin:1.2)"
    # prompt = "(photograph of scarlett as a blonde woman in a crowded street, highly detailed face, no makeup), skin blemishes, visible pores, goosebumps, 16k, high resolution, 8k uhd, dslr, high quality, film grain, Fujifilm XT3"

    fg_prompt = "BTS boys"
    bg_prompt = "fantasy, fantasy magic, intricate, sharp focus, illustration, \
        highly detailed, digital painting, concept art, matte, Artgerm and Paul lewin \
        and kehinde wiley, masterpiece"

    # https://nerdschalk.com/best-negative-prompts-in-stable-diffusion/
    # n_prompt = "disfigured, disproportionate , bad anatomy, bad proportions, \
    #     ugly, out of frame, mangled, asymmetric, cross-eyed, depressed, \
    #     immature, stuffed animal, out of focus, high depth of field, cloned face, \
    #     cloned head, age spot, skin blemishes, collapsed eyeshadow, asymmetric ears, \
    #     imperfect eyes, unnatural, conjoined, missing limb, missing arm, \
    #     missing leg, poorly drawn face, poorly drawn feet, poorly drawn hands, \
    #     floating limb, disconnected limb, extra limb, malformed limbs, malformed hands, \
    #     poorly rendered face, poor facial details, poorly rendered hands, double face, \
    #     unbalanced body, unnatural body, lacking body, long body, cripple, old , fat, \
    #     cartoon, weird colors, unnatural skin tone, unnatural skin, stiff face, fused hand, \
    #     skewed eyes, surreal, cropped head, group of people"

    # prompt = "(masterpiece:1.0), (best quality:1.4), (ultra highres:1.2), \
    #     (photorealistic:1.4), (8k, RAW photo:1.2), (soft focus:1.4), 1 woman, \
    #     (sharp focus:1.4), (japanese:1.2), (american:1.1), detailed beautiful \
    #     face, black hair, (detailed leather jacket and leather pants:1.4), beautiful white shiny humid skin, smiling"
    
    # n_prompt = "illustration, 3d, sepia, painting, cartoons, sketch, \
    #     (worst quality:2), (low quality:2), (normal quality:2), lowres, \
    #     bad anatomy, bad hands, normal quality, ((monochrome)), \
    #     ((grayscale:1.2)),newhalf, collapsed eyeshadow, multiple eyebrows, \
    #     pink hair, analog, analogphoto"


    # Using Realistic model: https://huggingface.co/SG161222/Realistic_Vision_V2.0
    # prompt = "RAW photo, Beautiful BTS boys, fully clothed, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    # prompt = "Pixar style little girl, 4k, 8k, unreal engine, octane render photorealistic by cosmicwonder, hdr, photography by cosmicwonder, high definition, symmetrical face, volumetric lighting, dusty haze, photo, octane render, 24mm, 4k, 24mm, DSLR, high quality, 60 fps, ultra realistic"
    n_prompt = "(deformed iris, deformed pupils, bad drawn nose, fused nose, poorly drawn nose, extra nose,  bad mouth, \
        fused mouth, poorly drawn mouth, big mouth, mouth cake, cracked mouth, crooked lips, \
        semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, \
        worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, \
        poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, \
        extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, \
        extra legs, fused fingers, too many fingers, long neck, "
    # n_prompt = "done by bad-artist, ugly, dazed, light blue eyes, 3D render, 3D game, 3D game scene, 3D character, mosaic, \
    #     painting, illustration, digital art, cartoon, anime, doll, toy, photoshop, video game, surreal, sign, 3dcg, decorating, \
    #     decoration, crayon, clipart, cgi, rendered, 3d, cartoon face, drawing, cgstation, airbrushed, sketch, render, unreal \
    #     engine, blender, digital painting, airbrush, pointillism, painting, image compression, distorted, JPEG artifacts, noisy, \
    #     shaky, pixelated, unclear, artifacts, low detail, low quality, low resolution, distortion, amateur, low res, low-res, \
    #     cropped body, cut off, basic, boring, botched, unprofessional, draft, failure, fake, image corruption, irregular, uneven, \
    #     unnatural, contorted, twisted, unappealing, blurry, haze, worst quality, normal quality, bad shadow, poor quality, amateur \
    #     photography, tasteless, tacky, lacklustre, simple, plain, grainy, out of focus, fuzzy, cropped, uncentered, out of frame, \
    #     body out of frame, split image, truncated, disjointed, incoherent, disorganized, jumbled, floating, objects, unreal, \
    #     deformations, kitsch, unattractive, opaque, boring pose, plain background, boring, plain, standard, average, uninventive, \
    #     derivative, homogenous, uncreative, ineffective, drab, amateur, censor, censored, censor_bar, text, font, ui, error, \
    #     watermark, username, signature, QR code, bar code, copyright, logos, HUD, tiling, label, watermarks, calligraphy, kanji, \
    #     hanzi, hangul, hanza, chu, nom, latin, arabic, cyrillic, symbols, alphanumeric, unicode, script, artist name, logo, censor, \
    #     high contrast, low contrast, High pass filter, watermarked, monotone, smooth, blur, vignette, filter, writing, oversaturation, \
    #     over saturation, over shadow, gaussian, blurred, weird colors, blurred, grain, bad art, black-white, posterization, \
    #     colour banding, grayscale, monochrome, b&w, oversaturated, black and white, no color, greyscale, poorly drawn, messy \
    #     drawing, bad proportions, gross proportions, imperfection, dehydrated, misshappen, duplicate, double, clones, twins, \
    #     brothers, same face, repeated person, bad anatomy, anatomical nonsense, malformed, misshaped body, uncoordinated body, \
    #     unnatural body, long body, liquid body, deformed, mutilated, mutation, mutated, tumor, deformed body, lopsided, mangled, \
    #     skin defect, disfigured, conjoined, connected, intertwined, hooked, bad body, amputation, siamese, cropped head, bad framing, \
    #     out of shot, awkward poses, unusual poses, smooth skin, misshapen, gross proportions, poorly drawn face, bad face, fused face, \
    #     cloned face, big face, long face, dirty face, long neck, warped face, loose face, crooked face, asymetric jaw, \
    #     asymmetric chin, fake face, deformed face, extra heads, big forehead, head cut off, ugly hair, bald, poorly drawn hair, \
    #     bad drawn eyes, asymmetric eyes, unaligned eyes, crooked eyes, closed eyes, looking_away, fused eyes, poorly drawn eyes, \
    #     extra eye, cross eyed, imperfect eyes, cataracts, glaucoma, strabismus, heterochromia, woobly iris, square iris, weird eyes, \
    #     distorted eyes, deformed glasses, extra eyes, bright blue eyes, cross-eyed, blurry eyes, poorly drawn eyes, fused eyes, \
    #     blind, red eyes, bad eyes, ugly eyes, dead eyes, bad drawn nose, fused nose, poorly drawn nose, extra nose, bad mouth, \
    #     fused mouth, poorly drawn mouth, big mouth, mouth cake, cracked mouth, crooked lips, dirty teeth, yellow teeth, ugly teeth, \
    #     liquid tongue, colorful tongue, black tongue, bad tongue, tongue within mouth, too long tongue, crooked teeth, yellow teeth, \
    #     long teeth, bad teeth, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, \
    #     asymmetric ears, big ears, ugly ears, bad collarbone, fused collarbone, missing collarbone, liquid collarbone, missing limb, \
    #     malformed limbs, extra limb, floating limbs, disconnected limbs, extra limb, amputee, extra limbs, different limbs \
    #     proportions, decapitated limbs, mutated hands, poorly drawn hands, malformed hands, bad hands, fused hands, missing hand, \
    #     extra hand, mangled hands, more than 1 left hand, more than 1 right hand, less than two hands appearing in the image, \
    #     cropped hands, out of frame hands, thousand hands, mutated hands and fingers, missing hands, distorted hands, \
    #     deformed hands, imperfect hands, undetailed hands, fused fingers, mutated fingers, (tentacle finger), missing fingers, \
    #     one hand with more than 5 fingers, disfigured hands, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, extra fingers, too many fingers, bad gloves, poorly drawn gloves, fused gloves, disappearing arms, short arm, missing arms, extra arms, less than two arms appearing in the image, cropped arms, out of frame arms, long arms, deformed arms, short arm, different arms proportions, multiple belly buttons, missing belly button, broken legs, disappearing legs, missing legs, extra legs, more than 2 legs, huge thighs, disappearing thigh, missing thighs, extra thighs, more than 2 thighs, deformed legs, bad thigh gap, missing thigh gap, fused thigh gap, liquid thigh gap, poorly drawn thigh gap, huge calf, disappearing calf, missing calf, extra calf, fused calf, bad knee, extra knee, broken legs, different legs proportions, mutated feet, poorly drawn feet, malformed feet, bad feet, fused feet, missing feet, mangled feet, more than 1 left foot, more than 1 right foot, less than two foot appearing in the image, cropped feet, thousand feet, mutated feet and fingers, missing feet, distorted feet, deformed feet, imperfect feet, undetailed feet, ugly feet, extra foot, long toes, extra shoes, bad shoes, fused shoes, more than two shoes, poorly drawn shoes, fused cloth, poorly drawn cloth, multiple breasts, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, missing breasts, more than 2 nipples, missing nipples, different nipples, fused nipples, bad nipples, poorly drawn nipples, black nipples, colorful nipples, unnatural nipples, without form nipples, withered nipples, unerect nipples, extra nipples, more than two nipples, imperfect nipples"
    # n_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"

    # https://civitai.com/models/7468/scarlett-johanssonlora
    # n_prompt = "anime, cartoon, penis, fake, drawing, illustration, boring, 3d render, long neck, out of frame, extra fingers, mutated hands, ((monochrome)), ((poorly drawn hands)), 3DCG, cgstation, ((flat chested)), red eyes, multiple subjects, extra heads, close up, man asian, text ,watermarks, logo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, 3d, extra fingers, (((mutated hands and fingers))), large breasts, weapons, underwear, panties, cleavage"
    # n_prompt = "wires, ear rings, dirty face, (deformed iris, deformed pupils, bad eyes, semi-realistic:1.4) (bad-image-v2-39000, bad_prompt_version2, bad-hands-5, EasyNegative, NG_DeepNegative_V1_4T, bad-artist:0.7),(worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, (tan, muscular, loli, petite, child, infant, toddlers, young, chibi, sd character:1.1), multiple view, Reference sheet, long neck"

    seed=1294216578
    # seed=-1

    mimic_dance(url, work_folder, fg_prompt, bg_prompt, n_prompt, seed=seed)