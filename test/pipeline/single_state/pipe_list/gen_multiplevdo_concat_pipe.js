module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"gen_multiplevdo_concat_pipe", id:"gen_multiplevdo_concat"}, // pipe to create multiple videos and concat them in one shot
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
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val : "RAW photo, A single silhouette of a women in full yoga wear performing the classic Tree pose, a gradient sky in hues of sunrise or sunset, transitioning from a deep warm orange near the horizon to a soothing sky blue as you go upwards, raw,8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
                       label: "Prompt", show: true, type:"default"},
               //seed: {val:3141637636, label: "seed", show: true, type: "default"},
               n_prompt: {val: "illustration, 3d, 2d, painting, cartoons, sketch, (worst quality:1.9), (low quality:1.9), (normal quality:1.9), lowres, ((monochrome)), ((grayscale)), (cropped), oversaturated, imperfect, (bad hands), signature, watermark, username, artist name, error, bad image, bad photo, (worst quality, low quality, imperfect utensils, deformed body parts",label: "n_prompt", show: true, type: "default"},
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
                motion_template:{val:"zoom_in", label: "Motion Template", show: true, type:"default"},
                bg_music_prompt:{val:"soothing synth pads with gentle flutes", show: true, type:"default"},
                //voice_text:{val:"parrots chirping.",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:4, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                //fps:{val:8, label: "fps",type: "default", show: false},
                //history_prompt:{val: "v2/en_speaker_9", label:"history_prompt", type: "default", show: false},
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
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val : "RAW photo,A single silhouette of an individual performing the classic Dhanurasana,the silhouette could be either black or white, contrasting the background.A gradient sky in hues of sunrise or sunset, transitioning from a deep warm orange near the horizon to a soothing sky blue as you go upwards. This depicts the transition from tension to tranquility through yoga , canon r5, lens 24 - 70mm, mavic pro 3, style raw,8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3,8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
                       label: "Prompt", show: true, type:"default"},
                //seed: {val:36795, label: "seed", show: true, type: "default"},
                n_prompt: {val: "illustration, 3d, 2d, painting, cartoons, sketch, (worst quality:1.9), (low quality:1.9), (normal quality:1.9), lowres, ((monochrome)), ((grayscale)), (cropped), oversaturated, imperfect, (bad hands), signature, watermark, username, artist name, error, bad image, bad photo, (worst quality, no repeat image, 2, 3 ,4 images, low quality, imperfect utensils, deformed body parts",label: "n_prompt", show: true, type: "default"},
                op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_one_img", // Name of the function to call
              stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
              stage_progress: 0, // Percentage of task completion
              output:['https://tmpfiles.org/1900371/generated123.mp4',
                      'https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // STAGE 3: Generate the VIDEO 2 using IMAGE 2
            // gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, bg_music_prompt=None, voice_text=None, **gen_vdo_args)
            config:{ 
                img_path:{val : {stg_idx:2, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                theme:{val : {stg_idx:2, type:"cfg2idx", idx:'theme'}, show: false, type:"wrkflw_link"},
                prompt:{val : {stg_idx:2, type:"cfg2idx", idx:'prompt'}, show: false, type:"wrkflw_link"},
                motion_template:{val:"zoom_out", label: "Motion Template", show: true, type:"default"},
                bg_music_prompt:{val:"soothing synth pads with gentle flutes",label: "BG Music", show: true, type:"default"},
                //voice_text:{val:"In a world where moments race by, hold onto every tender wag and soothing purr; they're the whispers of love that linger forever",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:4, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                //fps:{val:, label: "fps",type: "default", show: false},
                //history_prompt:{val: "v2/en_speaker_6", label:"history_prompt", type: "default", show: false},
                //is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // STAGE 4: Concat the different videos [ FROM STAGE 1 & 3 ] into a single video
            // concat_vdo(vdo_file_lst, op_fld)
            config:{ 
                vdo_file_lst:{val : {stg_idx:[1,3], type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
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