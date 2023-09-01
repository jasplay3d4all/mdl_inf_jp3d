module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        pipe_info:{label:"Social Media Post", id:"social_media_post"}, //Not sure whether it is required
        user_state:{
            info:{
                id:"JAYKANIDAN",
                attachments: [{url:"./share_vol/data_io/inp/infinity_logo.png", 
                    filename: '00000.png', width: 512, height: 512, content_type: 'image/png'}]
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 0: Generate the logo image
            // gen_img(theme, prompt, op_fld, control_type=None, ctrl_img_path=None, n_prompt="", height=512, width=512, 
            // seed=-1, num_images=1, safety_checker=None, collect_cache=True)
            config:{ 
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val : "A Image of A beautiful landscape, like a mountain range or a waterfall ,high detailed, masterpiece, best quality, 8k uhd, high quality, dslr, soft lighting, film grain," ,label: "Prompt", show: true, type:"default"},
                n_prompt: {val: "(worst quality:0.8), cartoon, halftone print, burlap, candle, (verybadimagenegative_v1.3:0.3), (cinematic:1.2), (surreal:0.8), (modernism:0.8), (art deco:0.8), (art nouveau:0.8)", label: "n_prompt", show: true, type: "default"},
                op_fld:{val : "./img/", show: false, type:"path"},
                //ctrl_img_path:{val : "", show: false, type:"attachments"},
                //control_type:{val : "pidiedge", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_img", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }}, {
            // Stage 1: Generate the different dimensions for different social media format
            // gen_inpaint_filler(theme, prompt, img_path, imgfmt_list, op_fld)
            config:{ 
                img_path:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                theme:{val : "food", label: "Theme", show: true, type:"default"},
                prompt:{val : "coloring effect",
                        label: "Prompt", show: true, type:"default"},
                imgfmt_list:{val:"insta_square, insta_potrait, twit_instream",
                        label: "Image Format", show: true, type:"array"},
                op_fld:{val : "./logo_list/", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_inpaint_filler", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},
        ],
    }

    return module;
};