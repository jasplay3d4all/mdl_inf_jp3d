import os
# https://github.com/python-ffmpegio/python-ffmpegio#transcoding
import ffmpegio
# from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_merge_video_audio
import glob
from PIL import Image

# Path for inputs have to be full system path and does not work with relative paths

def concat_vdo(vdo_file_lst, output_folder, verbose=True):
    os.makedirs(output_folder, exist_ok=True)
    outfile = os.path.join(output_folder, "generated.mp4")

    ffconcat = ffmpegio.FFConcat()
    ffconcat.add_files(vdo_file_lst)
    with ffconcat: # generates temporary ffconcat file
        ffmpegio.transcode(ffconcat, outfile, f_in='concat', show_log=verbose, safe_in=0, codec='copy')#
    # files = ['/video/video1.mkv','/video/video2.mkv']

def merge_vdo_ado(vdo_path, ado_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    outfile = os.path.join(output_folder, "generated.mp4")

    # rates, data = ffmpegio.media.read(vdo_path, ado_path)
    # print(rates['v:0'], rates['a:0'], data['v:0'].shape, data['a:0'].shape)
    # ffmpegio.video.write(outfile, rates, data)

    # video_clip = VideoFileClip(vdo_path)
    # audio_clip = AudioFileClip(ado_path)
    # # Concatenate the video clip with the audio clip
    # final_clip = video_clip.set_audio(audio_clip)
    # # Export the final video with audio
    # final_clip.write_videofile(outfile)

    ffmpeg_merge_video_audio(vdo_path, ado_path, outfile, vcodec='h264', 
        acodec='mp3', ffmpeg_output=False, logger='bar')
    return



def make_gif(frame_folder):
    os.makedirs(output_folder, exist_ok=True)
    outfile = os.path.join(output_folder, "generated.mp4")

    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    basepath = "/home/source_code/mdl_inf_jp3d/"
    ado_path = os.path.join(basepath, "expts/music_speech/musicgen_out.wav")
    ado_path1 = os.path.join(basepath, "expts/music_speech/bark_generation.wav")
    vdo_path = os.path.join(basepath, "share_vol/data_io/0QWERTY123/vdo/stg_3/vdo/generated.mp4")
    merge_vdo_ado(vdo_path, ado_path1, "./output")
    
