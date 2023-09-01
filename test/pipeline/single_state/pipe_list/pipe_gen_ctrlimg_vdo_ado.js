module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line: pipe_ctrl_img_vdo_gen: used to generate 1 image and vdo corresponding to that the text prompt and reference image given [Not sure whether it is required]
        pipe_info:{label:"pipe_gen_ctrlimg_vdo_ado", id:"pipe_gen_ctrlimg_vdo_ado"}, //pipe_line - module name - Not sure whether it is required
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
            }},{
            // Stage 1: Generate a video using the above image - helps to generate vdo with ado and speech
            // gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, bg_music_prompt=None, voice_text=None, **gen_vdo_args)
            config:{ 
                img_path:{val : {stg_idx:0, type:"array2idx", idx:0}, show: false, type:"wrkflw_link"}, // takes the input from above path from array.
                theme:{val : {stg_idx:0, type:"cfg2idx", idx:'theme'}, show: false, type:"wrkflw_link"}, // takes the input from above path from array.
                prompt:{val : {stg_idx:0, type:"cfg2idx", idx:'prompt'}, show: false, type:"wrkflw_link"}, // takes the input from above path from array.
                // motion_template - option to give in various motions like pan_right, pan_left, pan_top_left, pan_bottom_righ, zoom_in, zoom_out.
                motion_template:{val:"pan_bottom_right", label: "Motion Template", show: true, type:"default"},
                // bg_music_prompt - text prompt for the background music to be added to the vdo.
                bg_music_prompt:{val:"Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach",
                                label: "BG Music", show: true, type:"default"},
                // voice_text - text prompt for the backgound voice prompt to the vdo. if music is there better to avoid this.
                voice_text:{val:"Sip, Savor, Repeat – Awaken Your Senses with Every Pour.",label: "Voice Text", show: true, type:"default"},
                //num_sec - this helps is setting the seconds for length of the vdo.
                num_sec:{val:12, type: "default", label:"num_sec",show: false},
                //op_fld - path of the output vdo.
                op_fld:{val : "./vdo/", show: false, type:"path"},
                //fps - decides quality of the vdo. preferably should be in multiples of 8. it can be maintined as commented.
                //fps:{val:8, label: "fps",type: "default", show: false},
                //history_prompt - this helps in choosing the voice to speak. default speaker is set to 1. we have options from 1 - 9 [ 6 is preferable voice & 9 is lady voice ]
                history_prompt:{val: "v2/en_speaker_9", label:"history_prompt", type: "default", show: false},
                // is_gif: helps to give a giff image for the image generated. for now the resolution is more. size of the image is very huge.
                // is_gif:{val:true, label: "gif_Image", show: false, type:"default"},
            },
            gpu_state:{ 
                function: "gen_vdo_ado_speech", // Name of the function to call 
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
                //output - temp link from where the vdo can be downloaded. Its visible one the gpu job is 2 complete.
                output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},
        ],
    }
    return module;
};