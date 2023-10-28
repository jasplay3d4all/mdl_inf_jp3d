module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line - to generate 1-5 images using only text prompt and also generate images for various social media sites
        pipe_info:{label:"gen_img_for_sm_pipe", id:"gen_img_for_sm"}, 
        user_state:{
            info:{
                id:"JAYKANIDAN",
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 0: Generate the image - setting the parameters to generate image.
            //gen_one_img(theme, prompt, control_type=None, safety_checker=None, **kwargs)
            //:sd_mdl = sd_model(theme=theme, control_type=control_type, safety_checker=safety_checker)
            // theme - choose required theme to be used for img gen
            // prompt - give the text prompt for img genenation
            // n_prompt: if we have particular details to be set in negative prompt we can give it prompt value here.
            // op_fld - path of the output where the image is stored.
              config:{ 
                 
                theme:{val : "people", label: "Theme", show: true, type:"default"}, 
                prompt:{val :"RAW photo, long shot of Hitler in Gandhian dress,\
                            looking at me, calm, peaceful, dslr, soft lighting, high quality, uhd, 8k, film grain, Fujifilm XT ", 
                           label: "prompt", show: true, type: "default"}, 
                n_prompt: {val: "(worst quality:0.8),nudity, cartoon, halftone print, burlap, candle,\
                           (verybadimagenegative_v1.3:0.3), (cinematic:1.2), (surreal:0.8), (modernism:0.8), \
                           (art deco:0.8),bad anatomy, bad hands, three hands, three legs, bad arms, missing legs,\
                           missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, \
                           extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh,\
                           extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers,long fingers, horn,\
                           extra eyes, huge eyes, 2girl, amputation, disconnected limbs, error, sketch ,duplicate, ugly, \
                           monochrome, geometry, mutation, disgustingcartoon, cg, 3d, unreal,  ", 
                           label: "n_prompt", show: true, type: "default"},
                op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_one_img", // Name of the function to call
              stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
              stage_progress: 0, // Percentage of task completion
              output:['https://tmpfiles.org/1900371/generated123.mp4',
                      'https://tmpfiles.org/1900371/generated123.mp4'],
            }},{
            //  Stage 1: Generate the different dimensions for different social media format - 
            //  we are calling inpaint to fill in the extra space outside the imagegenerated.
            //  gen_inpaint_filler(theme, prompt, img_path, imgfmt_list, op_fld)
            config:{ 
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val : "Waterfall background in the wild", label: "Prompt", show: true, type:"default"},
                img_path:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"},
                imgfmt_list:{val:"insta_square, insta_potrait, twit_instream, whatsapp_status, fb_post",
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