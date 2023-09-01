const io = require("socket.io-client")
// const url = 'https://notebooksb.jarvislabs.ai:80';
// const url = "https://notebooksb.jarvislabs.ai/V2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HUS/proxy/6006/"
// const url = "http://127.0.0.1:8000/gpu"
const url = "http://172.17.0.3:6006/gpu"
const sio = io(url,{
    // path:'/2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HU/gpu/socket.io/', 
    reconnect: true}) // rejectUnauthorized:false})


pipe_state_list = {
    'ad': require("../../pipeline/ad/ad_gpu_state.js")(),
}
var cur_pipe = 'ad';
// Initialize pipe_state
var pipe_state = pipe_state_list[cur_pipe].pipeline_state;
var stage1_state = pipe_state_list[cur_pipe].stage1_state;
var stage2_state = pipe_state_list[cur_pipe].stage2_state;
var stage3_state = pipe_state_list[cur_pipe].stage3_state;
var stage4_state = pipe_state_list[cur_pipe].stage4_state;

// 0 Update the config and user id data
pipe_state.user_state.info.id = "JAYKANIDAN"
pipe_state.stage_state[0].config.theme.val = "food"
pipe_state.stage_state[0].config.num_scene.val = 2

// Lora: foodphoto, SPLASH, SPLASHES, SPLASHING, EXPLOSION, EXPLODING
// 1 Update pipe_state based on config and update stage config
s1s = JSON.parse(JSON.stringify(stage1_state))
s1s.config.prompt.val = "foodphoto, splashes, Strawberry icecream with chocolate sauce <lora:more_details:0.7>" + "Desert with camels and sand dunes"
s1s.config.n_prompt.val = ""
pipe_state.stage_state.push(s1s)

// 2
s1s = JSON.parse(JSON.stringify(stage1_state))
// s1s.config.fg_prompt.val = "foodporn, EXPLODING, Chocolate milk shake with vanilla chocochip toppings , <lora:add_details:0.7>"
s1s.config.prompt.val = "Two women in casual dress on a summer beach"+"Amazon forest with lush green bushes and waterfall"
s1s.config.n_prompt.val = ""
pipe_state.stage_state.push(s1s)

// 3
s2s = JSON.parse(JSON.stringify(stage2_state))
s2s.config.motion_template.val = "pan_right" // pan_right, pan_left, pan_top_left, pan_bottom_right
s2s.config.bg_music_prompt.val = "flute with guitar"
pipe_state.stage_state.push(s2s)

// 4
s2s = JSON.parse(JSON.stringify(stage2_state))
s2s.config.motion_template.val = "pan_left"
s2s.config.bg_music_prompt.val = "birds chirping. Owls hooting"
pipe_state.stage_state.push(s2s)

// 5
s3s = JSON.parse(JSON.stringify(stage3_state))
s3s.config.logo_img_path.val = "./share_vol/data_io/inp/logo_mealo.png"
s3s.config.motion_template.val = "pan_top_left"
s3s.config.voice_text.val = "Mealo: Melt your heart away. Your food is a click away."
s3s.config.logo_prompt.val = "Two women in casual dress on a summer beach 24 y.o woman, wearing black dress,\
    beautiful face, smile, blonde, cinematic shot, dark shot, dramatic lighting"
pipe_state.stage_state.push(s3s)


// 6
s4s = JSON.parse(JSON.stringify(stage4_state))
s4s.config.video_ordering.val = "default"
pipe_state.stage_state.push(s4s) 

// Socket listeners for invokeai backend api
//socket.on("connect", (socket) => {log(socket)})
sio.on("gpu_job_progress", (pipe_state) => {
    console.log("Received gpu_job_progress ",)
})
sio.on("gpu_job_complete", (pipe_state) => {
    pipe_state.cur_stage_idx += 1;
    console.log("Received gpu_job_complete ",pipe_state.cur_stage_idx)
    if(pipe_state.cur_stage_idx < 7){
        sio.emit('gpu_job_start', pipe_state);
    } 
})
sio.on("gpu_job_error", (pipe_state) => {
    console.log("Received gpu_job_error ",)
    pipe_list[pipe_state.name].job_error(pipe_state) 
    // remove_job_from_queue(pipe_state)
})

// set the current stage and pipeline progress
pipe_state.cur_stage_idx = 1; // 0 - stage 0 is not processed in gpu. 
sio.emit('gpu_job_start', pipe_state); 
console.log("Emitted first event")
