module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line: gen_music_only - only to generate bg music
        pipe_info:{label:"gen_music_only_pipe", id:"gen_music_only"}, 
        user_state:{
            info:{
                id:"JAYKANIDAN",
                
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 1: Generate only bg music from text prompt - below is the function we call for that 
            
            // gen_music(bg_music_prompt, num_sec, music_path, melody_path=None, musicgen_mdl='melody')
            
            // bg_music_prompt - text prompt for the background music to be generated.
            // num_sec - this helps is setting the seconds for length of the vdo.
            // music_path - path were the bg music gets stored.
            config:{ 
                                
                bg_music_prompt:{val:"Create a serene mantra-inspired music track at 80 BPM, incorporating traditional sitar and tabla instruments to evoke a peaceful and meditative mood, perfect for yoga and relaxation-focused YouTube Shorts content.",
                                label: "BG Music", show: true, type:"default"},
                num_sec:{val:15, type: "default", label:"num_sec",show: true},
                music_path:{val : "./bgmusic", show: false, type:"path"}
                               
            },
            gpu_state:{ 
                function: "gen_music", // Name of the function to call 
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
               // output:['https://tmpfiles.org/1900371/generated123.mp4'],
            }},
        ],
    }
    return module;
};