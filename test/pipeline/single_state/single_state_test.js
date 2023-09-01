const io = require("socket.io-client")
// const url = 'https://notebooksb.jarvislabs.ai:80';
// const url = "https://notebooksb.jarvislabs.ai/V2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HUS/proxy/6006/"
const url = "http://127.0.0.1:80/gpu"
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
    'social_media_post': require("./pipe_list/social_media_post_gpu_state.js")(),
    'social_media_vdo': require("./pipe_list/social_media_vdo_gpu_state.js")(),
    'social_media_vdowithlogo': require("./pipe_list/social_media_vdowithlogo_gpu_state.js")(),
    // to generate mutliple images using only text prompt and get output for various social media sites
    // 'gen_img_with_txt_prompt_for_sm':require("./pipe_list/pipe_gen_img_with_txt_prompt_for_sm.js")(),
    // to generate mutliple images using ctrlimg functions
    'gen_ctrl_multiple_imgs': require("./pipe_list/pipe_gen_ctrl_multiple_imgs.js")(), 
    // to generate vdo ado with logo using ctrlimg functions
    'gen_ctrlimg_vdo_ado': require("./pipe_list/pipe_gen_ctrlimg_vdo_ado.js")(),
    'talking_head': require("./pipe_list/talking_head_gpu_state.js")(),
}

var test_pipe = 'social_media_vdo';
// Initialize pipe_state
var pipe_state = pipe_state_list[test_pipe].pipeline_state;
// set the current stage and pipeline progress
pipe_state.cur_stage_idx = 0; // In Default we are executing only one module at a time

function trigger_firsy_event() {
    sio.emit('gpu_job_start', pipe_state); 
    console.log("Emitted first event");
 }

setTimeout(trigger_firsy_event, 1000);
