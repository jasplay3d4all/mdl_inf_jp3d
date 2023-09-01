


import os
import subprocess



def audio_animate_face():
    return

def swap_face_in_photo():
    return

def animate_photo():
    return

def gen_single_face_talking_head():
    gen_img() # Generate a potrait of a person
    gen_speech() # Generate speech for the given text
    
    return


def add_args(args, in_path, tag, original_directory, sep='--'):
    if(in_path):
        in_path = os.path.join(original_directory, in_path)
        args += [sep+tag, in_path]
    return args
def talking_head(driven_audio, source_image, result_dir, enhancer=None, still=False,
    preprocess=None, expression_scale=0.0, ref_eyeblink=None, ref_pose=None,
    path_to_sadtalker='./ext_lib/SadTalker/'):

    # Save the current working directory
    original_directory = os.getcwd()
    # Change to the new directory
    os.chdir(path_to_sadtalker)
    args = []
    args = add_args(args, driven_audio, 'driven_audio', original_directory)
    args = add_args(args, source_image, 'source_image', original_directory)
    args = add_args(args, result_dir, 'result_dir', original_directory)
    args = add_args(args, ref_pose, 'ref_pose', original_directory)
    args = add_args(args, ref_eyeblink, 'ref_eyeblink', original_directory)
    args += ['--enhancer'] if enhancer else []
    args += ['--still'] if still else []
    # Use --still with full
    args += ['--preprocess', preprocess] if preprocess in ['crop', 'resize', 'full'] else []
    args += ['--expression_scale', expression_scale] if expression_scale > 0.0 else []

    completed_process = subprocess.run(['python', './inference.py'] + args, capture_output=True, text=True)
    # completed_process = subprocess.run(["ls" ], capture_output=True, text=True)
    # Print return code (should be 0 if command executed successfully)
    print('Return Code:', completed_process.returncode)
    print('Output:', completed_process.stdout)
    print('Error:', completed_process.stderr)

    # Change back to the original directory
    os.chdir(original_directory)
    return

# https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md
# Animate without voice using diffanimate for establishing cntext
# generate voice using text to speech
# then animate using voice image
# Using this lenght add music
# combine all the three to get a video - moviepy editor
# 

# python inference.py --driven_audio <audio.wav> \
#                     --source_image <video.mp4 or picture.png> \
#                     --enhancer gfpgan 

# python inference.py --driven_audio <audio.wav> \
#                     --source_image <video.mp4 or picture.png> \
#                     --result_dir <a file to store results> \
#                     --still \
#                     --preprocess full \
#                     --enhancer gfpgan 

# --expression_scale - controls the emotion
# --ref_eyeblink - use external video to control the generted video
# --ref_pose - use external pose to control the generated video
# --input_yaw,
# --input_pitch,
# --input_roll

# use resize with real portraits to generate images
# use full mode with still and enhancer to produce better images with full length?
# --enhancer <gfpgan or RestoreFormer> - only face
# --background_enhancer <realesrgan> - full image

# python run.py --execution-provider cuda -s ../SadTalker/examples/source_image/art_17.png 
#     -t ../SadTalker/results/2023_07_25_05.24.26.mp4 -o ../../../share_vol/test/

def swap_head_vdo(source_image, target_video, result_dir, 
    path_to_roop='./ext_lib/articulated_motion/roop/'):

    # Save the current working directory
    original_directory = os.getcwd()
    # Change to the new directory
    os.chdir(path_to_roop)
    args = []
    args = add_args(args, source_image, 's', original_directory, sep ='-')
    args = add_args(args, target_video, 't', original_directory, sep = '-')
    args = add_args(args, result_dir, 'o', original_directory, sep = '-')

    completed_process = subprocess.run(['python', 'run.py'] + args, capture_output=True, text=True)
    # completed_process = subprocess.run(["ls" ], capture_output=True, text=True)
    # Print return code (should be 0 if command executed successfully)
    print('Return Code:', completed_process.returncode)
    print('Output:', completed_process.stdout)
    print('Error:', completed_process.stderr)

    # Change back to the original directory
    os.chdir(original_directory)
    return


# python swapper.py \
# --source_img="./data/man1.jpeg;./data/man2.jpeg" \
# --target_img "./data/mans1.jpeg" \
# --face_restore \
# --background_enhance \
# --face_upsample \
# --upscale=2 \
# --codeformer_fidelity=0.5

def image_swapper(source_img, target_img, face_restore=False, background_enhance=False, face_upsample=False, 
    upscale=None, codeformer_fidelity=None, path_to_swapper='./ext_lib/articulated_motion/inswapper/'):

    # Save the current working directory
    original_directory = os.getcwd()
    # Change to the new directory
    os.chdir(path_to_swapper)

    args = []
    args = add_args(args, source_img, 'source_img', original_directory, sep='--')
    args = add_args(args, target_img, 'target_img', original_directory, sep='--')

    if not isinstance(face_restore, bool):
        raise ValueError("face_restore must be a boolean value.")
    if face_restore:
        args += ['--face_restore']

    if not isinstance(background_enhance, bool):
        raise ValueError("background_enhance must be a boolean value.")
    if background_enhance:
        args += ['--background_enhance']

    if not isinstance(face_upsample, bool):
        raise ValueError("face_upsample must be a boolean value.")
    if face_upsample:
        args += ['--face_upsample']

    if upscale is not None:
        if not isinstance(upscale, int) or upscale < 0:
            raise ValueError("upscale must be a positive integer.")
        args += ['--upscale', str(upscale)]

    if codeformer_fidelity is not None:
        if not isinstance(codeformer_fidelity, float) or codeformer_fidelity < 0 or codeformer_fidelity > 1:
            raise ValueError("codeformer_fidelity must be a float between 0 and 1.")
        args += ['--codeformer_fidelity', str(codeformer_fidelity)]

    completed_process = subprocess.run(['python', './swapper.py'] + args, capture_output=True, text=True)

    # Change back to the original directory
    os.chdir(original_directory)

    return completed_process.returncode



if __name__ == "__main__":
    driven_audio = "./share_vol/0QWERTY123/stg_3/ado/output00000.wav"
    driven_audio = "./ext_lib/articulated_motion/SadTalker/examples/driven_audio/RD_Radio40_000.wav"

    source_image = "./ext_lib/articulated_motion/SadTalker/examples/source_image/art_0.png"
    ref_vdo = "./ext_lib/articulated_motion/SadTalker/examples/ref_video/WDA_AlexandriaOcasioCortez_000.mp4"
    result_dir = "./share_vol/test/"
    talking_head(driven_audio, source_image, result_dir)