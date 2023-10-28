
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
            // gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512,seed=-1, num_images=1, safety_checker=None,collect_cache=True):
            config:{ 
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val :"RAW photo, a photograph perfect well-lit (closeup:1.15) (medium shot portrait:0.6) photograph of a handsome Indian 35 y.o man with, charming smile standing, white shirt, pleasent background, looking at me, slight smile,8k uhd, dslr, soft lighting, high quality,                                 film grain, Fujifilm XT3",
                        label: "Prompt", show: true, type:"default"},
                n_prompt: {val: "illustration, 3d, 2d, painting, cartoons, sketch, (worst quality:1.9), (low quality:1.9), (normal quality:1.9), lowres,((monochrome)), ((grayscale)), (cropped), oversaturated, imperfect, (bad hands), signature, watermark, username, artist name, error, bad image, bad photo, (worst quality, low quality, bad anatomy,bad proportions, blurry, boring, cloned face, cropped,deformed, dehydrated, dull, error, extra arms, extra fingers",label: "n_prompt", show: true, type: "default"},
                //control_type:{val: "openpose", show: false, type:"default"},
                //seed:{val :50037, label: "seed_val", show: true, type:"default"},
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
                history_prompt:{val : "v2/en_speaker_6", label: "Speaker Type", show: true, type:"default"},
                text:{val : "if you need modern art Our tool created outcome  Lets say A portrait of a young woman in a white dress with long flowing hair and piercing blue eyes. Genearated image  Finally Image of A beautiful landscape, like a mountain range or a waterfall  Our tool output",
                        label: "Text to talk", show: true, type:"default"},
                music_path:{val : "./speech", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},{
            // Stage 2: Generate the talking head for the speech generated
            // def talking_head(driven_audio, source_image, result_dir, enhancer=None, still=False, preprocess=None, expression_scale=0.0, ref_eyeblink=None,ref_pose=None, path_to_sadtalker='./ext_lib/articulated_motion/SadTalker/')
            
            // enhancer = Using 'gfpgan' or 'RestoreFormer' to enhance the generated face via face restoration network
            // expression_scale : default=1, a larger value will make the expression motion stronger.
            // preprocess : Run and produce the results in the croped input image. Other choices: resize, where the images will be resized to the specific resolution. full Run the full image animation, use with --still to get better results.
            // ref_eyeblink : 	A video path, where we borrow the eyeblink from this reference video to provide more natural eyebrow movemen
            // ref_pose : A video path, where we borrow the pose from the head reference video.
            config:{ 
                driven_audio:{val : {stg_idx:1, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                source_image:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
           //     enhancer: {val:2, show:true, type: "default"},
           //     preprocess: {val : "full", show: , type:"default"},
           //     expression_scale: {val : 1.0, show: true, type:"default"},
                result_dir:{val : "./talking_vdo", show: true, type:"path"}
                
            },
            gpu_state:{ 
                function: "talking_head", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                output:['https://tmpfiles.org/1900371/generated123.mp4']
            }},
        ],
    }

    return module;
};
