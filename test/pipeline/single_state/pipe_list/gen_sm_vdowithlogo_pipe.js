module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"gen_sm_vdowithlogo_pipe", id:"gen_sm_vdowithlogo_pipe"}, //Not sure whether it is required
        user_state:{
            info:{
                id:"JAYKANIDAN",
                attachments: [{url:"./share_vol/data_io/logo1_jasPlay3d_white.png", filename:'logo1_jasPlay3d_white.png', width: 512, height: 512, content_type: 'image/png'}],
                //attachments: [{url:"http://127.0.0.1:80/static/inp/logo1_jasPlay3d_white.png", filename: 'logo1_jasPlay3d_white.png', width: 512, height: 512, content_type:'image/png'}],
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 0: Generate the image
            //gen_one_img(theme, prompt, control_type=None, safety_checker=None, **kwargs)
            //:sd_mdl = sd_model(theme=theme, control_type=control_type, safety_checker=safety_checker)
              config:{ 
                    theme:{val : "people", label: "Theme", show: true, type:"default"},
                    prompt:{val :"RAW photo, close up of girl twins in summer wear in a lawn, looking at me, (best quality, highest quality),\
                                (ultra detailed),(8k, 4k, intricate),(highly detailed:1.2), (detailed face:1.2),(gradients),\
                                (ambient light:1.3)",
                       label: "Prompt", show: true, type:"default"},
                    n_prompt: {val: "(worst quality:0.8),nudity, cartoon, halftone print, burlap, candle,\
                       (verybadimagenegative_v1.3:0.3), (cinematic:1.2), (surreal:0.8), (modernism:0.8), \
                       (art deco:0.8),bad anatomy, bad hands, three hands, three legs, bad arms, missing legs,\
                       missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, \
                       extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh,\
                       extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers,long fingers, horn,\
                       extra eyes, huge eyes,amputation, disconnected limbs, error, sketch ,duplicate, ugly, \
                       monochrome, geometry, mutation, disgustingcartoon, cg, 3d, unreal, deformed legs, deformed hands,\
                       improper teeth alignment", 
                       label: "n_prompt", show: true, type: "default"},       
                   //seed: { val:3141637636 , label: "seed_val", show: false, type: "default"},
                   op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_one_img", // Name of the function to call
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
                motion_template:{val:"zoom_in", label: "Motion Template", show: true, type:"default"},
                bg_music_prompt:{val:"Create a playful theme with a BPM of 80, in the key of C major, and a time signature of 4/4. Use the piano, violin, and cello as the main instruments. The song should have a light and airy feel, with a sense of adventure and excitement",
                label: "BG Music", show: true, type:"default"},
                //voice_text:{val:"We have tour packages for Couples and family",label: "Voice Text", show: true, type:"default"},
                num_sec:{val:12, type: "default", label:"num_sec",show: false},
                op_fld:{val : "./vdo/", show: false, type:"path"},
                fps:{val:8, label: "fps",type: "default", show: false},
                history_prompt:{val: "v2/en_speaker_3", label:"history_prompt", type: "default", show: false},
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
                pos_y: {val: "bottom", show: true, label:"logo_position_pos_y", type: "default"},
                pos_x: {val: "right", show: true, label:"logo_position_pos_x", type: "default"},
                scale: {val:0.20, show: true, label: "scale", type: "default"},
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