module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"Single Talking Head", id:"talking_head"}, //Not sure whether it is required
        user_state:{
            info:{
                id:"JAYKANIDAN",
                attachments: [{url:"./share_vol/data_io/inp/logo_mealo.png", 
                    filename: '00000.png', width: 512, height: 512, content_type: 'image/png'}]
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 0: Generate the image
            // gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
            // seed=-1, num_images=1, safety_checker=None, collect_cache=True):
            config:{ 
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val : "a perfect well-lit (closeup:1.15) (medium shot portrait:0.6) photograph of a beautiful Indian woman standing\
                             on the hiking trail, wearing an intriguing outfit,looking at me, slight smile, with face painted in Orange,\
                             White and Green like Indian flag",
                        label: "Prompt", show: true, type:"default"},
                // control_type:{val: "openpose", show: false, type:"default"},
                op_fld:{val : "./img", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_img", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},{
            // Stage 1: Generate the speech of the person
            // gen_speech(text, music_path, history_prompt="v2/en_speaker_1")
            config:{ 
                history_prompt:{val : "v2/en_speaker_9", label: "Speaker Type", show: true, type:"default"},
                text:{val : "Proverbs are traditional sayings that are particular to a certain country. \
                            They are short, wise sayings that usually offer some kind of advice, or capture an idea found in life.",
                        label: "Text to talk", show: true, type:"default"},
                music_path:{val : "./speech", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},{
            // Stage 2: Generate the talking head for the speech generated
            // def talking_head(driven_audio, source_image, result_dir, enhancer=None, still=False,
            //     preprocess=None, expression_scale=0.0, ref_eyeblink=None, ref_pose=None,
            //     path_to_sadtalker='./ext_lib/articulated_motion/SadTalker/')
            config:{ 
                driven_audio:{val : {stg_idx:1, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                source_image:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                result_dir:{val : "./talking_vdo", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "talking_head", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},
        ],
    }

    return module;
};