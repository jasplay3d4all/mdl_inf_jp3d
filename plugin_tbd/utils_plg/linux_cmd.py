
import os
import subprocess

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
    pass

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

    outfile = os.path.join(output_folder, "output%04d.mp4")
    
    runcmd = "ffmpeg -i "+input_file
    runcmd += " -c copy -map 0 -segment_time " + time
    runcmd += " -f segment -reset_timestamps 1 " + outfile
    # print(runcmd)
    executecmd(runcmd, verbose=verbose)

def ext_imgs(input_file, output_folder, fps, verbose=False):
    os.makedirs(output_folder, exist_ok=True)
    # "ffmpeg -i input.mp4 -vf fps=1 %04d.png"
    outfile = os.path.join(output_folder, "output%04d.png")

    runcmd = "ffmpeg -i "+input_file
    runcmd += " -vf fps="+fps+" "+outfile
    executecmd(runcmd, verbose=verbose)

def make_vdo(input_folder, output_folder, framerate, verbose=True):
    # mogrify -path black -thumbnail 640x480 -background black -gravity center -extent 640x480 png/*.png
    os.makedirs(output_folder, exist_ok=True)
    # ffmpeg -framerate 10 -i filename-%03d.jpg output.mp4
    # -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"

    input_file = os.path.join(input_folder, "output%04d.png")
    outfile = os.path.join(output_folder, "generated.mp4")
    wth_hgt = '720:1280'

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


if __name__ == "__main__":

    link_path = "links.txt"
    output_folder = "./images/"
    links = get_links(link_path)
    get_images(links, output_folder)
