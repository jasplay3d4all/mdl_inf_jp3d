import os
import subprocess

def add_args(args, in_path, tag, original_directory, sep='--'):
    if in_path:
        in_path = os.path.join(original_directory, in_path)
        args += [sep + tag, in_path]
    return args

def image_swapper(source_img, target_img, face_restore=False, background_enhance=False, 
                  face_upsample=False, upscale=None, codeformer_fidelity=None, 
                  path_to_swapper='./path_to_swapper_directory/'):
    
    # Input validation
    if not isinstance(face_restore, bool):
        raise ValueError("face_restore must be a boolean.")
    if not isinstance(background_enhance, bool):
        raise ValueError("background_enhance must be a boolean.")
    if not isinstance(face_upsample, bool):
        raise ValueError("face_upsample must be a boolean.")
    if upscale and (not isinstance(upscale, int) or upscale < 1):
        raise ValueError("upscale must be an integer greater than 0.")
    if codeformer_fidelity and (not isinstance(codeformer_fidelity, (int, float)) or codeformer_fidelity < 0 or codeformer_fidelity > 1):
        raise ValueError("codeformer_fidelity must be a number between 0 and 1.")
    
    # Check if path_to_swapper exists and is a directory
    if not os.path.isdir(path_to_swapper):
        raise ValueError(f"Invalid path_to_swapper: {path_to_swapper}")

    # Save the current working directory
    original_directory = os.getcwd()

    try:
        # Change to the new directory
        os.chdir(path_to_swapper)
        args = []
        args = add_args(args, source_img, 'source_img', original_directory)
        args = add_args(args, target_img, 'target_img', original_directory)
        args += ['--face_restore'] if face_restore else []
        args += ['--background_enhance'] if background_enhance else []
        args += ['--face_upsample'] if face_upsample else []
        args += ['--upscale', str(upscale)] if upscale else []
        args += ['--codeformer_fidelity', str(codeformer_fidelity)] if codeformer_fidelity else []

        completed_process = subprocess.run(['python', './swapper.py'] + args, capture_output=True, text=True)

        # Print return code (should be 0 if command executed successfully)
        print('Return Code:', completed_process.returncode)
        print('Output:', completed_process.stdout)
        print('Error:', completed_process.stderr)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Change back to the original directory
        os.chdir(original_directory)
    return


if __name__ == '__main__':
    

    img = ins_get_image('t1')
    print("Input image ", img.shape, np.min(img), np.max(img))
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    assert len(faces)==6
    source_face = faces[2]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("./t1_swapped.jpg", res)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("./t1_swapped2.jpg", res)

