
import os
import subprocess
import json
import requests


def executecmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    return std_out, std_err

def read_txt_file(filepath):
    text_file = open(filepath, "r")
    lines = text_file.readlines()
    text_file.close()
    return lines

def dld_imgs(url_list, output_folder, verbose=False):
    os.makedirs(output_folder, exist_ok=True)

    for i, link in enumerate(url_list):
        extension = link.strip().split('.')[-1]
        outfile = os.path.join(output_folder, str(i)+"."+extension)
        
        runcmd = "wget "
        runcmd += "--output-document='" + outfile + "' "
        runcmd += link
        # print(runcmd)
        executecmd(runcmd, verbose=verbose)

# PATH_YT_DLP="../expt/yt-dlp_linux "
PATH_YT_DLP="yt-dlp "

def dld_vdo(url, output_folder, verbose=True):
    os.makedirs(output_folder, exist_ok=True)

    outfile = os.path.join(output_folder, "video.webm")
    
    runcmd = PATH_YT_DLP
    runcmd += "-o "+ outfile
    runcmd += " " + url
    # print(runcmd)
    executecmd(runcmd, verbose=verbose)
    # Convert to mp4 ffmpeg -i video.webm video.mp4
    outfile_mp4 = os.path.join(output_folder, "video.mp4")
    runcmd = "ffmpeg "
    runcmd += " -i "+ outfile
    runcmd += " "+ outfile_mp4
    executecmd(runcmd, verbose=verbose)
    # return outfile +"%(ext)s"

def split_vdo(input_file, output_folder, time, verbose=False):
    os.makedirs(output_folder, exist_ok=True)
    # "ffmpeg -i input.mp4 -c copy -map 0 -segment_time 00:20:00 -f segment -reset_timestamps 1 output%03d.mp4"

    outfile = os.path.join(output_folder, "output%05d.mp4")
    
    runcmd = "ffmpeg -i "+input_file
    runcmd += " -c copy -map 0 -segment_time " + time
    runcmd += " -f segment -reset_timestamps 1 " + outfile
    # print(runcmd)
    executecmd(runcmd, verbose=verbose)

def ext_imgs(input_file, output_folder, fps, verbose=False):
    os.makedirs(output_folder, exist_ok=True)
    # "ffmpeg -i input.mp4 -vf fps=1 %04d.png"
    outfile = os.path.join(output_folder, "output%05d.png")

    runcmd = "ffmpeg -i "+input_file
    runcmd += " -vf fps="+fps+" "+outfile
    executecmd(runcmd, verbose=verbose)

def make_vdo(input_folder, output_folder, framerate, width, height, verbose=True):
    # mogrify -path black -thumbnail 640x480 -background black -gravity center -extent 640x480 png/*.png
    os.makedirs(output_folder, exist_ok=True)
    # ffmpeg -framerate 10 -i filename-%03d.jpg output.mp4
    # -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"

    input_file = os.path.join(input_folder, "output%05d.png")
    outfile = os.path.join(output_folder, "generated.mp4")
    wth_hgt = str(width)+':'+str(height)

    runcmd = "ffmpeg "+" -framerate "+framerate+" -y -i "+input_file
    runcmd += ' -vf "scale='+wth_hgt+':force_original_aspect_ratio=decrease,pad='+wth_hgt+':(ow-iw)/2:(oh-ih)/2,setsar=1" '
    runcmd += " -pix_fmt yuv420p "
    runcmd += "  "+outfile
    print(runcmd)
    executecmd(runcmd, verbose=verbose)

def bg_remover(input_folder, output_folder, verbose=True):
    # rembg p path/to/input path/to/output
    os.makedirs(output_folder, exist_ok=True)

    runcmd = "rembg p -om -m  u2net_human_seg "+input_folder+" "+output_folder
    print(runcmd)
    executecmd(runcmd, verbose=verbose) 

# Audio related solutions
# https://stackoverflow.com/questions/14498539/how-to-overlay-downmix-two-audio-files-using-ffmpeg
# ffmpeg -i input0.mp3 -i input1.mp3 -filter_complex amix=inputs=2:duration=longest output.mp3
# https://json2video.com/how-to/ffmpeg-course/ffmpeg-add-audio-to-video.html

def upload_tmpfiles(file_path, verbose=False):
    if(not os.path.isfile(file_path)):
        return -1
    runcmd = 'curl -F "file=@'+file_path+'" https://tmpfiles.org/api/v1/upload'
    if(verbose):
        print(runcmd)
    std_out, std_err = executecmd(runcmd, verbose=verbose)
    std_out = json.loads(std_out)
    if(std_out["status"]== "success"):
        return std_out["data"]["url"]
    else:
        return std_out["status"]
    
def download_link(http_link, file_path, verbose=False):

    r = requests.get(http_link, stream = True)
    if(r.status_code != 200):
        print("Download failed ", r.status_code)
        return 0

    with open(file_path,"wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            # writing one chunk at a time to pdf file
            if chunk:
                f.write(chunk)
                # print("chunk ", chunk)
    
    return 1

if __name__ == "__main__":

    # link_path = "links.txt"
    # output_folder = "./images/"
    # links = get_links(link_path)
    # get_images(links, output_folder)

    # file_path = "../../share_vol/data_io/JAYKANIDAN/0/img/00000.png"
    # for i in range(100):
    #     upload_tmpfiles(file_path, verbose=True)

    url = 'http://127.0.0.1/static/inp/logo_mealo.png'
    download_link(url, "./out.png")
