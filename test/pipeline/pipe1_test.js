const io = require("socket.io-client")
const sio = io("http://127.0.0.1:8000/gpu",{reconnect: true})

pipe_state_list = {
    'pipe1': require("../../pipeline/pipe1/pipe1_gpu_state.js")(),
}
var cur_pipe = 'pipe1';
// Initialize pipe_state
var pipe_state = pipe_state_list['pipe1'].pipeline_state;
var stage1_state = pipe_state_list['pipe1'].stage1_state;
var stage2_state = pipe_state_list['pipe1'].stage2_state;
var stage3_state = pipe_state_list['pipe1'].stage3_state;

// Update the config and user id data
pipe_state.user_state.info.id = "0QWERTY123"
pipe_state.stage_state[0].config.theme.val = "food"
pipe_state.stage_state[0].config.num_scene.val = 1


// Update pipe_state based on config and update stage config
s1s = JSON.parse(JSON.stringify(stage1_state))
s1s.config.fg_prompt.val = "Hot piping pizza"
s1s.config.bg_prompt.val = "In front of a beach"
pipe_state.stage_state.push(s1s)

s2s = JSON.parse(JSON.stringify(stage2_state))
s2s.config.motion_template.val = "pan_right"
pipe_state.stage_state.push(s2s)

s3s = JSON.parse(JSON.stringify(stage3_state))
s3s.config.video_ordering.val = "default"
pipe_state.stage_state.push(s3s) 

// Socket listeners for invokeai backend api
//socket.on("connect", (socket) => {log(socket)})
sio.on("gpu_job_progress", (pipe_state) => {
    console.log("Received gpu_job_progress ",)
})
sio.on("gpu_job_complete", (pipe_state) => {
    pipe_state.cur_stage_idx += 1;
    console.log("Received gpu_job_complete ",pipe_state.cur_stage_idx)
    if(pipe_state.cur_stage_idx < 4){
        sio.emit('gpu_job_start', pipe_state);
    } 
})
sio.on("gpu_job_error", (pipe_state) => {
    console.log("Received gpu_job_error ",)
    pipe_list[pipe_state.name].job_error(pipe_state) 
    // remove_job_from_queue(pipe_state)
})

// set the current stage and pipeline progress
pipe_state.cur_stage_idx = 3; //1; // 0 - stage 0 is not processed in gpu. 
sio.emit('gpu_job_start', pipe_state); 
