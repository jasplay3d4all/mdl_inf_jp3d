import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
# from IPython.display import HTML
from base64 import b64encode
import datetime

pipe = DiffusionPipeline.from_pretrained("camenduru/potat1", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.to('cuda:0')

prompt = 'Darth Vader surfing a wave' #@param {type:"string"}
negative_prompt = "low quality" #@param {type:"string"}
num_frames = 30 #@param {type:"raw"}
video_frames = pipe(prompt, negative_prompt=negative_prompt, width=1024, height=576, num_inference_steps=25, num_frames=num_frames).frames
output_video_path = export_to_video(video_frames)
