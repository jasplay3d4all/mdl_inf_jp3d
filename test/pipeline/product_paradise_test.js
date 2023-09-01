const io = require("socket.io-client")
// const url = 'https://notebooksb.jarvislabs.ai:80';
// const url = "https://notebooksb.jarvislabs.ai/V2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HUS/proxy/6006/"
const url = "http://127.0.0.1:8000/gpu"
// const url = "http://172.17.0.4:8000/gpu"
const sio = io(url,{
    // path:'/2rvGqSfVl_MPiwtezXDWWHd-Pl9euIfF4Wi-5m7tJkx7k78E4pdK6xZFVxhJh2HU/gpu/socket.io/', 
    reconnect: true}) // rejectUnauthorized:false})


pipe_state_list = {
    'product_paradise': require("../../pipeline/product_paradise/product_paradise_gpu_state.js")(),
}
var cur_pipe = 'product_paradise';
// Initialize pipe_state
var pipe_state = pipe_state_list[cur_pipe].pipeline_state;
var stage1_state = pipe_state_list[cur_pipe].stage1_state;
var stage2_state = pipe_state_list[cur_pipe].stage2_state;
var stage3_state = pipe_state_list[cur_pipe].stage3_state;
var stage4_state = pipe_state_list[cur_pipe].stage4_state;

// 0 Update the config and user id data
pipe_state.user_state.info.id = "0QWERTY123"
pipe_state.stage_state[0].config.theme.val = "food"
pipe_state.stage_state[0].config.num_scene.val = 2
pipe_state.stage_state[0].gpu_state.logo_path = "./share_vol/data_io/inp/edge_inv.png"

// Lora: foodphoto, SPLASH, SPLASHES, SPLASHING, EXPLOSION, EXPLODING
// 1 Update pipe_state based on config and update stage config
s1s = JSON.parse(JSON.stringify(stage1_state))
s1s.config.fg_prompt.val = "A close-up of a scoop of chocolate ice cream melting in a bowl,The ice cream is smooth and creamy, with a rich chocolate flavor,The lighting is soft and natural, with a warm glow.The photography effect is a shallow,depth of field, which blurs the background and focuses on the ice cream, The resolution is 4K, and the frame rate is 60fps"
s1s.config.bg_prompt.val = "The background is a rustic wooden table, with a white tablecloth and a vase of flowers"
pipe_state.stage_state.push(s1s)

// 2
s1s = JSON.parse(JSON.stringify(stage1_state))
// s1s.config.fg_prompt.val = "foodporn, EXPLODING, Chocolate milk shake with vanilla chocochip toppings , <lora:add_details:0.7>"
s1s.config.fg_prompt.val = "A couple  sitting on a table, enjoying milkshakes, The milkshakes are different flavors, and they are topped with whipped cream,chocolate sauce,sprinkles,The resolution is 4K, and the frame rate is 60fps"
s1s.config.bg_prompt.val = "The background is a modern, brightly lit cafe"
pipe_state.stage_state.push(s1s)

// 3
s2s = JSON.parse(JSON.stringify(stage2_state))
s2s.config.motion_template.val = "pan_botttom_right" // pan_right, pan_left, pan_top_left, pan_bottom_right
s2s.config.bg_music_prompt.val = "pop music with 96 BPM"
pipe_state.stage_state.push(s2s)

// 4
s2s = JSON.parse(JSON.stringify(stage2_state))
s2s.config.motion_template.val = "pan_top_left"
s2s.config.bg_music_prompt.val = "hopeful, laid back, ponderous, romantic , smooth ,broadcasting in Beats in 96 BPM"
pipe_state.stage_state.push(s2s)

// 5
s3s = JSON.parse(JSON.stringify(stage3_state))
s3s.config.motion_template.val = "pan_right"
s3s.config.voice_text.val = "Desserts! The taste of happiness"
s3s.config.logo_prompt.val = "dessert, churro, chocolate, cream, hot chocolate, fruit, soft lighting, high quality, film grain,on the white table.The resolution is 4K, and the frame rate is 60fps in The background of modern cafe"
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
