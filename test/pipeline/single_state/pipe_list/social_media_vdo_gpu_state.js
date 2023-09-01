module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"Social Media Post", id:"social_media_post"}, //Not sure whether it is required
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
            // Stage 0: Generate the image
            //gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
            // seed=-1, num_images=1, safety_checker=None, collect_cache=True)
              config:{ 
                   theme:{val : "people", label: "Theme", show: true, type:"default"},
                   prompt:{val : " A modern art masterpiece, RAW photo, landscape, old fashion, nature, high detailed, masterpiece, best quality, 8k uhd, high quality, film grain, Fujifilm XT3",
                       label: "Prompt", show: true, type:"default"},
                   seed: {val:1914929369, label: "seed", show: true, type: "default"},
                   n_prompt: {val: "illustration, 3d, 2d, painting, cartoons, sketch, (worst quality:1.9), (low quality:1.9), (normal quality:1.9), lowres, ((monochrome)), ((grayscale)), (cropped), oversaturated, imperfect, (bad hands), signature, watermark, username, artist name, error, bad image, bad photo, (worst quality, low quality",
                              label: "n_prompt", show: true, type: "default"},
                   op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_img", // Name of the function to call
              stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
              stage_progress: 0, // Percentage of task completion
              output:['https://tmpfiles.org/1900371/generated123.mp4',
                      'https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // Stage 1: Generate a video using the above image
            // gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, bg_music_prompt=None, voice_text=None, **gen_vdo_args)
            config:{ 
                img_path:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                theme:{val : {stg_idx:0, type:"cfg2idx", idx:'theme'}, show: false, type:"wrkflw_link"},
                prompt:{val : {stg_idx:0, type:"cfg2idx", idx:'prompt'}, show: false, type:"wrkflw_link"},
                motion_template:{val:"zoom_out", label: "Motion Template", show: true, type:"default"},
                //bg_music_prompt:{val:"a piano and cello duet playing a sad chambers music",label: "BG Music", show: true, type:"default"},
                //voice_text:{val:"parrots chirping.",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:6, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                fps:{val:8, label: "fps",type: "default", show: false},
                //history_prompt:{val: "v2/en_speaker_9", label:"history_prompt", type: "default", show: false},
                //is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // Stage 2: Generate a video using the above image
            // gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, bg_music_prompt=None, voice_text=None, **gen_vdo_args)
            config:{ 
                img_path:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                theme:{val : {stg_idx:0, type:"cfg2idx", idx:'theme'}, show: false, type:"wrkflw_link"},
                prompt:{val : {stg_idx:0, type:"cfg2idx", idx:'prompt'}, show: false, type:"wrkflw_link"},
                motion_template:{val:"zoom_in", label: "Motion Template", show: true, type:"default"},
                //bg_music_prompt:{val:"a piano and cello duet playing a sad chambers music",label: "BG Music", show: true, type:"default"},
                //voice_text:{val:"parrots chirping.",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:6, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                fps:{val:8, label: "fps",type: "default", show: false},
                //history_prompt:{val: "v2/en_speaker_9", label:"history_prompt", type: "default", show: false},
                //is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // Stage 3: Concat the different videos into a single video
            // concat_vdo(vdo_file_lst, op_fld)
            config:{ 
                vdo_file_lst:{val : {stg_idx:[1,2], type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
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