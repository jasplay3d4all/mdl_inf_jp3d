import os
import sys
import shutil
import numpy as np
import urllib.request

from lib.utils_plg.dict_obj import dict2obj, obj2dict
from lib.utils_plg.linux_cmd import upload_tmpfiles, download_link

from lib.ctrl_img_gen import gen_img, gen_inpaint_filler # gen_logo, 
# from core_blks.vdo_gen import gen_vdo, gen_logo_vdo "gen_logo_vdo":gen_logo_vdo, "gen_vdo":gen_vdo,
from lib.audio_plg import gen_music, gen_speech
# from lib.annotated_motion import talking_head
from lib.vdo_plg import gen_vdo_ado_speech, gen_zoom_video, gen_pan_video, concat_vdo, add_img_to_vdo
from lib.talking_head_api import talking_head
name_to_fn_mapper = { "gen_img":gen_img, "gen_inpaint_filler":gen_inpaint_filler, # "gen_logo":gen_logo, 
    "gen_music":gen_music, "gen_speech":gen_speech, "talking_head":talking_head,
    "gen_vdo_ado_speech": gen_vdo_ado_speech, "gen_zoom_video":gen_zoom_video, "gen_pan_video":gen_pan_video,
    "concat_vdo":concat_vdo, "add_img_to_vdo":add_img_to_vdo
}

# def dummy_test(**stg_cfg):
#     print("Input args ", stg_cfg)
#     return
# name_to_fn_mapper = { "gen_img":dummy_test, "gen_logo":dummy_test, "gen_inpaint_filler":dummy_test,
#     "gen_logo_vdo":dummy_test, "gen_vdo":dummy_test, "gen_music":dummy_test, "gen_speech":dummy_test, "talking_head":dummy_test
# }

class single_state:
    name="single_state"
    def __init__(self, config):
        self.fn_map = name_to_fn_mapper
        self.config = config
        self.num_sec = 4; self.fps = 24; self.vdo_wth=1280; self.vdo_hgt=720;
        return

    def unload(self, stage_state_0):
        return
    def load(self, stage_state_0, sio_api):
        # Use the stage_state 0 to configure the pipe
        self.sio_api = sio_api; # Store the sio api class to call progress and error call back
        # memory_stats(3)
        return self
    def create_workspace(self, user_id): 
        # Each user gets a single folder
        self.user_basefolder = os.path.join(self.config.data_io_path, user_id)
        # shutil.rmtree(path, ignore_errors=False, onerror=None, *, dir_fd=None)
        if(os.path.isdir(self.user_basefolder)):
            shutil.rmtree(self.user_basefolder)
        os.makedirs(self.user_basefolder, exist_ok=True)
        self.inp_folder = os.path.join(self.user_basefolder, "inp")
        os.makedirs(self.inp_folder, exist_ok=True)
        return

    def process(self, pipe_state):
        def get_args_dict(stg_cfg, cur_stage_idx):
            stg_cfg = obj2dict(stg_cfg)
            stg_cfg_args = {}
            for key in stg_cfg.keys():
                if(stg_cfg[key]["type"] == "attachments"):
                    img_url = pipe_state.user_state.info.attachments[0].url
                    if('http' in img_url):
                        img_name = pipe_state.user_state.info.attachments[0].filename
                        attach_img_path = os.path.join(self.inp_folder, img_name)
                        download_link(img_url, attach_img_path)
                        stg_cfg_args[key] = attach_img_path
                        # print("http cfg ",'http' in img_url, img_url, attach_img_path)
                    else:
                        stg_cfg_args[key] = img_url
                elif(stg_cfg[key]["type"] == "wrkflw_link"):
                    # Go to the previous stage output and use that as the input for this stage
                    if(stg_cfg[key]["val"]["type"] == "array2idx"):
                        if(hasattr(stg_cfg[key]["val"]["stg_idx"], "__len__")):
                            stg_cfg_args[key] = []
                            for upper_stg_idx in stg_cfg[key]["val"]["stg_idx"]:
                                stg_cfg_args[key].append( pipe_state.stage_state[upper_stg_idx].gpu_state.output[stg_cfg[key]["val"]["idx"]].path)
                        else:
                            upper_stg_idx = stg_cfg[key]["val"]["stg_idx"]
                            stg_cfg_args[key] = pipe_state.stage_state[upper_stg_idx].gpu_state.output[stg_cfg[key]["val"]["idx"]].path
                    elif(stg_cfg[key]["val"]["type"] == "cfg2idx"):
                        upper_stg_idx = stg_cfg[key]["val"]["stg_idx"]
                        upper_stg_cfg = obj2dict(pipe_state.stage_state[upper_stg_idx].config)
                        # print("cfg2idx ", upper_stg_cfg)
                        stg_cfg_args[key] = upper_stg_cfg[stg_cfg[key]["val"]["idx"]]["val"]
                elif(stg_cfg[key]["type"] == "array"):
                    # comma separated list of strings that are to be used
                    stg_cfg_args[key] = [x.strip() for x in stg_cfg[key]["val"].split(',')]
                elif(stg_cfg[key]["type"] == "path"):
                    # Append basefolder to localize the changes
                    stg_cfg_args[key] = os.path.join(self.user_basefolder, str(cur_stage_idx), stg_cfg[key]["val"])
                else:
                    # All other types just append the param
                    stg_cfg_args[key] = stg_cfg[key]["val"]
            return stg_cfg_args

        def post_process(output):
            # Convert output from single/array to a file list and upload it to tmpfiles.org
            url_list = []
            if(hasattr(output, "__len__")):# Check whether the output is a array
                for entry in output:
                    url_list.append(upload_tmpfiles(entry["path"]))
            else:
                url_list.append(upload_tmpfiles(output["path"]))
                
            return url_list
        if(pipe_state.cur_stage_idx == 0):
            self.create_workspace(pipe_state.user_state.info.id)
        stage_state = pipe_state.stage_state[pipe_state.cur_stage_idx];
        stg_cfg = get_args_dict(stage_state.config, pipe_state.cur_stage_idx)

        output = self.fn_map[stage_state.gpu_state.function](**stg_cfg);
        stage_state.gpu_state.output = output
        stage_state.gpu_state.tmpfile_output = post_process(output)
        print("Current stage ", pipe_state.cur_stage_idx)
        print("Output ", stage_state.gpu_state.tmpfile_output, output)

        return stage_state.gpu_state.output
