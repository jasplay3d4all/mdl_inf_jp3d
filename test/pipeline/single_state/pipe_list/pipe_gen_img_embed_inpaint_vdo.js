module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line - to generate 1-5 images using only text prompt
        pipe_info:{label:"pipe_gen_img_embed_inpaint_vdo", id:"pipe_gen_img_embed_inpaint_vdo"}, 
        user_state:{
            info:{
                id:"JAYKANIDAN",
                // path of the file with which we are going to ctrl the image generation 
                attachments: [{url:"./share_vol/data_io/inp/logo_mealo.png", 
                               filename:'00000.png', width: 512, height: 512, content_type: 'image/png'}],
              
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            {
            // Stage 0: Generate the different dimensions for different social media format
            // gen_inpaint_filler(theme, prompt, img_path, imgfmt_list, op_fld)
            config:{ 
                theme:{val : "people", label: "Theme", show: true, type:"default"},
                prompt:{val : "picture depicting artificial intelligence", label: "Prompt", show: true, type:"default"},
                img_path:{val : " ", show: false, type:"attachments"},
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