module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"Social Media Post", id:"social_media_post"}, //Not sure whether it is required
        user_state:{
            info:{
                id:"JAYKANIDAN",
                attachments: [{url:"./share_vol/data_io/inp/logo_mealo.png", filename:'00000.png', width: 512, height: 512, content_type: 'image/png'}],
               // attachments: [{url:"http://127.0.0.1:80/static/inp/logo_mealo.png", filename: 'logo_mealo.png', width: 512, height: 512, content_type:'image/png'}],
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 0: Generate the image
            //gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
            // seed=-1, num_images=1, safety_checker=None, collect_cache=True)
              config:{ 
                   theme:{val : "An", label: "Theme", show: true, type:"default"},
                   prompt:{val : "RAW photo, fresh fruit salad, food photography, full view, 45 degree view, full view, ultra detailed, 8k, clean + cinematic shot + photos taken by ARRI, photos taken by sony, photos taken by canon, photos taken by nikon, photos taken by sony, photos taken by hasselblad + incredibly detailed, sharpen, details + professional lighting, photography lighting + 50mm, 80mm, 100m + lightroom gallery + behance photographys + unsplash, food photography, studio lighting, 35mm lens,<lora:foodphoto:1> foodphoto, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
                       label: "Prompt", show: true, type:"default"},
                   //seed: { val:3141637636 , label: "seed_val", show: false, type: "default"},
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
                motion_template:{val:"pan_top_left", label: "Motion Template", show: true, type:"default"},
                //bg_music_prompt:{val:"Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach",
                //                 label: "BG Music", show: true, type:"default"},
                // voice_text:{val:"Sip, Savor, Repeat – Awaken Your Senses with Every Pour.",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:12, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                //fps:{val:8, label: "fps",type: "default", show: false},
                // history_prompt:{val: "v2/en_speaker_9", label:"history_prompt", type: "default", show: false},
                // is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            // Stage 2: Generate a video with logo using the above vdo
            // add_img_to_vdo(vdo_path, img_path, op_fld, pos_x="right", pos_y="bottom", scale=0.25)
            //Writing video ./share_vol/data_io/JAYKANIDAN/1/./vdo/mrg/generated.mp4
            config:{ 
                vdo_path:{val : {stg_idx:1, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                img_path:{val : "", show: false, type:"attachments"},
                pos_y: {val: "top", show: true, label:"logo_position_pos_y", type: "default"},
                scale: {val:0.10, show: true, label: "scale", type: "default"},
                op_fld:{val : "./vdo/vdowithlogo/", show: false, type:"path"},
            },
            gpu_state:{ 
                function: "add_img_to_vdo", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},
        ],
    }
    return module;
};