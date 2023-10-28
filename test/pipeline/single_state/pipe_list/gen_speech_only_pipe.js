module.exports = function () {
    var module = {};

    module.pipeline_state = {
        name: "single_state",
        cur_stage_idx: 0,
        //pipe_line: gen_speech_only - only to generate bg voice from text
        pipe_info:{label:"gen_speech_only_pipe", id:"gen_speech_only"}, 
        user_state:{
            info:{
                id:"JAYKANIDAN",
            },
            gpu_state:{} //Not sure whether it is required. Probably future use
        },
        stage_state:[{
            // Stage 1: Generate the speech of the person
            // gen_speech(text, music_path, history_prompt="v2/en_speaker_1")
            config:{ 
                history_prompt:{val : "v2/en_speaker_6", label: "Speaker Type", show: true, type:"default"},
                text:{val : "Jasplay3D Features are, one ideas to image, number two Deisgn In Motion,\
                three..Visual Depth effect, four...visual alchemy...five...whispers of life, six...Visual Symphony,\
                six...Picture in Picture, seven...Clips to cinematice magic...then we have few Add Ons too...." ,
                        label: "Text to talk", show: true, type:"default"},
                music_path:{val : "./speech", show: false, type:"path"}
            },
            gpu_state:{ 
                function: "gen_speech", // Name of the function to call
                stage_status:"in_queue", // "pre_stage" "in_queue", "gpu_triggered", "error", "complete"
                stage_progress: 0, // Percentage of task completion
            }},
        ],
    }
    return module;
};