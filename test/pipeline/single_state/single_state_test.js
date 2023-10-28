const io = require("socket.io-client")
// const url = 'https://notebooksb.jarvislabs.ai:80';
// const url = "https://notebooksb.jarvislabs.ai/V2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HUS/proxy/6006/"
const url = "http://127.0.0.1:8000/gpu"
// https://docs.runpod.io/docs/expose-ports
// const url = "https://q00z52fjcxtmx8-80.proxy.runpod.net/gpu" // https://{POD_ID}-{INTERNAL_PORT}.proxy.runpod.net
// const url = "http://172.17.0.4:8000/gpu"
const sio = io(url,{
    // path:'/2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HU/gpu/socket.io/', 
    reconnect: true}) // rejectUnauthorized:false})

// Socket listeners for invokeai backend api
//socket.on("connect", (socket) => {log(socket)})
sio.on("gpu_job_progress", (pipe_state) => {
    console.log("Received gpu_job_progress ",)
})
sio.on("gpu_job_complete", (pipe_state) => {
    pipe_state.cur_stage_idx += 1;
    console.log("Received gpu_job_complete ",pipe_state.cur_stage_idx)
    var num_stages = pipe_state.stage_state.length;
    if(pipe_state.cur_stage_idx < num_stages){
        sio.emit('gpu_job_start', pipe_state);
    } 
})
sio.on("gpu_job_error", (pipe_state) => {
    console.log("Received gpu_job_error ",)
    pipe_list[pipe_state.name].job_error(pipe_state) 
    // remove_job_from_queue(pipe_state)
})

const pipe_state_list = {
    // pipe_line: gen_img_only - to generate only images using given text prompt 1-5 images using only text prompt.
    // pipe_line: gen_music_only - to generate only bg music using given text prompt.
    // gen_img_for_social_media: to generate mutliple images using only text prompt and get output for various social media sites
    // gen_ctrl_multiple_imgs: to generate mutliple images using ctrlimg functions
    // gen_ctrlimg_vdo_ado: to generate vdo ado with logo using ctrlimg functions
    // gen_vdo_with_ado: to generate vdo with ado or music with motion template.
    // gen_multiplevdo_concat: to concat multiple vdos generated from different text prompts ( now its done for 2 vdos ).
    

    'gen_img_only': require("./pipe_list/gen_img_only_pipe.js")(),
    'gen_music_only': require("./pipe_list/gen_music_only_pipe.js")(),
    'gen_speech_only': require("./pipe_list/gen_speech_only_pipe.js")(),
    'gen_sm_vdowithlogo': require("./pipe_list/gen_sm_vdowithlogo_pipe.js")(), // works for pan right,left,bottom_right,top_left
    
    'gen_img_for_sm':require("./pipe_list/gen_img_for_sm_pipe.js")(), // Not working
    'gen_vdo_with_ado': require("./pipe_list/gen_vdo_with_ado_pipe.js")(), // not working

    // 'gen_txt_2_two_vdos': require("./pipe_list/gen_txt_2_two_vdos_pipe.js")(),
    'social_media_vdo': require("./pipe_list/social_media_vdo_gpu_state.js")(),
    
    
    
    'gen_ctrl_multiple_imgs': require("./pipe_list/pipe_gen_ctrl_multiple_imgs.js")(), 
    'gen_ctrlimg_vdo_ado': require("./pipe_list/pipe_gen_ctrlimg_vdo_ado.js")(),
    'gen_multiplevdo_concat': require("./pipe_list/gen_multiplevdo_concat_pipe.js")(),
    'gen_multiplevdo_aitalker_concat': require("./pipe_list/gen_multiplevdo_aitalker_concat_pipe.js")(),
    'talking_head': require("./pipe_list/talking_head_gpu_state.js")(),
}
var test_pipe =  'gen_vdo_with_ado';
// Initialize pipe_state
var pipe_state = pipe_state_list[test_pipe].pipeline_state;
// set the current stage and pipeline progress
pipe_state.cur_stage_idx = 0; // In Default we are executing only one module at a time

function trigger_first_event() {
    sio.emit('gpu_job_start', pipe_state); 
    console.log("Emitted first event");
 }

setTimeout(trigger_first_event, 10000);
