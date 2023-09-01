import os
import subprocess
import json
from pathlib import Path

# Placeholder fill-in function
def add_args(args, in_path, tag, original_directory, sep='--'):
    if in_path:
        in_path = os.path.join(original_directory, in_path)
        args += [sep+tag, in_path]
    return args

# New function
def talking_head(driven_audio, source_image, result_dir, enhancer=None, still=False,
    preprocess=None, expression_scale=0.0, ref_eyeblink=None, ref_pose=None,
    path_to_sadtalker='./ext_lib/SadTalker/'):

    if not os.path.exists(path_to_sadtalker):
        raise ValueError("Path to sadtalker does not exist.")
    if not os.path.exists(driven_audio):
        raise ValueError("Driven audio does not exist.")
    if not os.path.exists(source_image):
        raise ValueError("Source image does not exist.")
    if ref_eyeblink and not os.path.exists(ref_eyeblink):
        raise ValueError("Reference eyeblink does not exist.")
    if ref_pose and not os.path.exists(ref_pose):
        raise ValueError("Reference pose does not exist.")
    if not isinstance(expression_scale, (int, float)) or expression_scale < 0.0:
        raise ValueError("Expression scale must be a non-negative number.")
    if preprocess not in [None, 'crop', 'resize', 'full']:
        raise ValueError("Preprocess must be one of None, 'crop', 'resize', 'full'.")

    # Create result directory if it does not exist
    Path(result_dir).mkdir(parents=True, exist_ok=True)

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
    args += ['--preprocess', preprocess] if preprocess else []
    args += ['--expression_scale', str(expression_scale)] if expression_scale > 0.0 else []

    completed_process = subprocess.run(['python', './inference.py'] + args, capture_output=True, text=True)

    # Change back to the original directory
    os.chdir(original_directory)

    result = [{"path": os.path.join(result_dir, f)} for f in os.listdir(result_dir) if os.path.isfile(os.path.join(result_dir, f))]

    return result #json.dumps(result)
