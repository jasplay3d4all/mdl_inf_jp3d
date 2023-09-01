module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line: pipe_ctrl_multiple_img_gen: mainly used to generate multiple images using single text prompt.
        pipe_info:{label:"pipe_gen_ctrl_multiple_imgs", id:"pipe_gen_ctrl_multiple_imgs"}, 
        user_state:{
            info:{
                id:"JAYKANIDAN",
                attachments: [{url:"./share_vol/data_io/inp/infinity_white.png", 
                               filename:'00000.png', width: 512, height: 512, content_type: 'image/png'}], // path of the file with which we are going to ctrl the image generation 
               // attachments: [{url:"http://127.0.0.1:80/static/inp/logo_mealo.png", filename: 'logo_mealo.png', width: 512, height: 512, content_type:'image/png'}],
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 0: Generate the image - setting the parameters to generate image.
            //gen_img(theme, prompt, op_fld, control_type=None, ctrl_img_path=None, n_prompt="", height=512, width=512,seed=-1, num_images=1, safety_checker=None, collect_cache=True)
              config:{ 
                  // theme - choose required theme to be used for img gen
                  theme:{val : "people", label: "Theme", show: true, type:"default"}, 
                  // prompt - give the text prompt for img gen
                  prompt:{val : "RAW photo, close up portrait of 15 y.o twins babies in a cradle in a garden looking at me with a cute smile, wearing a white hat with rainbow colored summer dresss, on a bright sunny day, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 ", 
                           label: "prompt", show: true, type: "default"}, 
                  // num_images: number of images to be generated. max uptil 5 we can generate, avoid while using with seed value.
                  num_images: {val:5 , lable: "num_images", show: false, type: "defalut"}, 
                  // control_type - choose the ctrl types like pidiedge,hededge,canny,openpose,midasdepth,zoedepth
                  control_type: {val: "midasdepth", label: "contol_type", show: true, type: "default"},
                   //ctrl_img_path - path of the image which should be used for reference. type attachments.
                  ctrl_img_path:{val:"", label:"ctrl_img_path", show: false, type:"attachments"},
                  // seed - can be set only when you want a particular (same) type of image to be generated, else better to be in comment state to get variou outputs 
                  seed: { val:50018 , label: "seed_val", show: false, type: "default"},
                  // n_prompt: if we have particular details to be set in negative prompt we can give it prompt value here.
                  n_prompt: {val: " ",label: "n_prompt", show: true, type: "default"},
                  //op_fld - path of the output where the image is stored.
                   op_fld:{val : "./img/", show: false, type:"path"},
            },
            gpu_state:{ 
              function: "gen_img", // Name of the function to call
              stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
              stage_progress: 0, // Percentage of task completion
              output:['https://tmpfiles.org/1900371/generated123.mp4',
                      'https://tmpfiles.org/1900371/generated123.mp4'],
            }},
        ],
    }
    return module;
};