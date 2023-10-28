module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line - to generate only images using given text prompt 1-5 images using only text prompt.
        pipe_info:{label:"i2i", id:"gen_img_only_pipe"}, 
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
            // prompt - give the text prompt for img gen
            // seed-can be set only when you want a particular (same) type of image to be generated, else better comment the line to get various outputs 
            // n_prompt: if we have particular details to be set in negative prompt we can give it prompt value here.
            // op_fld - path of the output where the image is stored.
              config:{ 
                    theme:{val : "people", label: "Theme", show: true, type:"default"}, 
                    prompt:{val :"RAW photo,photo of national leader Jawahlal Nehru in a park,in day sunlight,\
                                looking at me, soft lighting, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3", 
                                label: "prompt", show: true, type: "default"}, 
                    //num_images: {val:5 , label: "num_images", show: false, type: "defalut"}, 
                    //seed: { val:18011 , label: "seed_val", show: false, type: "default"},
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
              output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }}
        ],
    }
    return module;
};