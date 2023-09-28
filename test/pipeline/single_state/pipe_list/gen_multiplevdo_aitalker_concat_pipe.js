module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"gen_multiplevdo_aitalker_concat_pipe", id:"gen_multiplevdo_aitalker_concat"}, // pipe to create multiple videos and concat them in one shot
        user_state:{
            info:{
                id:"JAYKANIDAN",
                // attachments: [{url:"./share_vol/data_io/inp/logo_mealo.png", 
                //     filename: 'logo_mealo.png', width: 512, height: 512, content_type: 'image/png'}]
               // attachments: [{url:"http://127.0.0.1:80/static/inp/logo_mealo.png", 
                //    filename: 'logo_mealo.png', width: 512, height: 512, content_type: 'image/png'}],
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // STAGE 0: Generate the IMAGE 1 using PROMPT 1.
            // gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
            // seed=-1, num_images=1, safety_checker=None, collect_cache=True)
            
              config:{ 
                theme:{val : "food", label: "Theme", show: true, type:"default"},
                prompt:{val : "RAW photo, Gourmet Strawberry Milkshake, <lora:foodphoto:1> foodphoto, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
                       label: "Prompt", show: true, type:"default"},
               //seed: {val:3141637636, label: "seed", show: true, type: "default"},
               // n_prompt: {val: "illustration, 3d, 2d, painting, cartoons, sketch, (worst quality:1.9), (low quality:1.9), (normal quality:1.9), lowres, ((monochrome)), ((grayscale)), (cropped), oversaturated, imperfect, (bad hands), signature, watermark, username, artist name, error, bad image, bad photo, (worst quality, no repeat image, 2, 3 ,4 images, low quality, imperfect utensils",label: "n_prompt", show: true, type: "default"},
                op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_one_img", // Name of the function to call
              stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
              stage_progress: 0, // Percentage of task completion
              output:['https://tmpfiles.org/1900371/generated123.mp4',
                      'https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // STAGE 1: Generate the VIDEO 1 using IMAGE 1
            // gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, bg_music_prompt=None, voice_text=None, **gen_vdo_args)
            config:{ 
                img_path:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                theme:{val : {stg_idx:0, type:"cfg2idx", idx:'theme'}, show: false, type:"wrkflw_link"},
                prompt:{val : {stg_idx:0, type:"cfg2idx", idx:'prompt'}, show: false, type:"wrkflw_link"},
                motion_template:{val:"pan_top_left", label: "Motion Template", show: true, type:"default"},
                bg_music_prompt:{val:"pop music.", show: true, type:"default"},
                //voice_text:{val:"Feeling hungry ? want to have a milkshake? ",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:6, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                //fps:{val:8, label: "fps",type: "default", show: false},
                history_prompt:{val: "v2/en_speaker_6", label:"history_prompt", type: "default", show: false},
                //is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // STAGE 2: Generate the IMAGE 2 using PROMPT 2.
            // gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
            // seed=-1, num_images=1, safety_checker=None, collect_cache=True)
            
              config:{ 
                theme:{val : "food", label: "Theme", show: true, type:"default"},
                prompt:{val : "RAW photo, pizza in a cafe, <lora:foodphoto:0.8> foodphoto, dslr, soft lighting, high quality, film grain, Fujifilm XT",
                       label: "Prompt", show: true, type:"default"},
                n_prompt: {val: "illustration, 3d, 2d, painting, cartoons, sketch, (worst quality:1.9), (low quality:1.9), (normal quality:1.9), lowres, ((monochrome)), ((grayscale)), (cropped), oversaturated, imperfect, (bad hands), signature, watermark, username, artist name, error, bad image, bad photo, (worst quality, no repeat image, 2, 3 ,4 images, low quality, imperfect utensils",label: "n_prompt", show: true, type: "default"},
                //seed: {val:2697012101, label: "seed", show: true, type: "default"},
                op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_one_img", // Name of the function to call
              stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
              stage_progress: 0, // Percentage of task completion
              output:['https://tmpfiles.org/1900371/generated123.mp4',
                      'https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // STAGE 3: Generate the VIDEO 2 using IMAGE 2 2
            // gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, bg_music_prompt=None, voice_text=None, **gen_vdo_args)
            config:{ 
                img_path:{val : {stg_idx:2, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                theme:{val : {stg_idx:2, type:"cfg2idx", idx:'theme'}, show: false, type:"wrkflw_link"},
                prompt:{val : {stg_idx:2, type:"cfg2idx", idx:'prompt'}, show: false, type:"wrkflw_link"},
                motion_template:{val:"pan_bottom_right", label: "Motion Template", show: true, type:"default"},
                bg_music_prompt:{val:"Create an ambient electronic melody in the key of A minor, set at a slow tempo of 80 BPM. The rhythm should be ethereal with long sustains and gradual transitions, evoking a sense of tranquility and introspection.",label: "BG Music", show: true, type:"default"},
                voice_text:{val:" [pause] Or do you want to have a burger ? ",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:6, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                //fps:{val:8, label: "fps",type: "default", show: false},
                history_prompt:{val: "v2/en_speaker_6", label:"history_prompt", type: "default", show: false},
                //is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // STAGE 4: Generate the FACE OF THE AI MODEL
            // gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512,seed=-1, num_images=1, safety_checker=None,collect_cache=True):
            config:{ 
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val :"professional photo, closeup portrait photo of 35 y.o man, wearing black sweater, smiling face, looking at me, dramatic lighting, nature, gloomy, cloudy weather, bokeh",
                        label: "Prompt", show: true, type:"default"},
                n_prompt: {val: "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, BadDream, UnrealisticDream",label: "n_prompt", show: true, type: "default"},
                //control_type:{val: "openpose", show: false, type:"default"},
                //seed:{val :908525188, label: "seed_val", show: true, type:"default"},
                op_fld:{val : "./img", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_one_img", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},{
            // Stage 5: Generate the SPEECH of the AI MODEL
            // gen_speech(text, music_path, history_prompt="v2/en_speaker_1")
            config:{ 
                history_prompt:{val : "v2/en_speaker_6", label: "Speaker Type", show: true, type:"default"},
                text:{val : "Be it desserts or main course we are here to serve you always!" ,
                        label: "Text to talk", show: true, type:"default"},
                music_path:{val : "./speech", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},{
            // STAGE 6: Generate the COMPLETE AI MODEL WITH Speech generated IN STAGE 5
            // def talking_head(driven_audio, source_image, result_dir, enhancer=None, still=False, preprocess=None, expression_scale=0.0, ref_eyeblink=None,ref_pose=None, path_to_sadtalker='./ext_lib/articulated_motion/SadTalker/')
            
            // enhancer = Using 'gfpgan' or 'RestoreFormer' to enhance the generated face via face restoration network
            // expression_scale : default=1, a larger value will make the expression motion stronger.
            // preprocess : Run and produce the results in the croped input image. Other choices: resize, where the images will be resized to the specific resolution. full Run the full image animation, use with --still to get better results.
            // ref_eyeblink : 	A video path, where we borrow the eyeblink from this reference video to provide more natural eyebrow movemen
            // ref_pose : A video path, where we borrow the pose from the head reference video.
            config:{ 
                driven_audio:{val : {stg_idx:5, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                source_image:{val : {stg_idx:4, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
            //  enhancer: {val:2, show:true, type: "default"},
            //  preprocess: {val : "full", show: false, type:"default"},
            //  expression_scale: {val : 1.0, show: true, type:"default"},
                result_dir:{val : "./talking_vdo", show: true, type:"path"}
                
            },
            gpu_state:{ 
                function: "talking_head", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4']
            }},{
            // STAGE 7: Concat the different videos [ FROM STAGE 1 & 3 ] into a single video
            // concat_vdo(vdo_file_lst, op_fld)
            config:{ 
                vdo_file_lst:{val : {stg_idx:[1,3,6], type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                op_fld:{val : "./concat/", show: false, type:"path"},
            },
            gpu_state:{ 
                function: "concat_vdo", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},
        ],
    }
    return module;
};