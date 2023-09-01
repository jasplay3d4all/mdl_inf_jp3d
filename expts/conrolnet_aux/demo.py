from PIL import Image
import requests
from io import BytesIO
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector
import time
import numpy as np

# load image
url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

img_np = np.array(img, dtype=np.uint8)


# load checkpoints
# hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
# midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
# mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
# pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
# normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
# lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
# lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
# zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")

# instantiate
# canny = CannyDetector()
# content = ContentShuffleDetector()
# face_detector = MediapipeFaceDetector()


# process
t1 = time.time()
processed_image_open_pose = open_pose(img, 
    hand_and_face=False, return_pil=False)
# processed_image_open_pose.save("pose.png")
print("Openpose time ", time.time() - t1, processed_image_open_pose.shape)
print(np.sum(processed_image_open_pose), np.max(img_np))
# processed_image_hed = hed(img)
# processed_image_midas = midas(img)
# processed_image_mlsd = mlsd(img)
# processed_image_pidi = pidi(img, safe=True)
# processed_image_normal_bae = normal_bae(img)
# processed_image_lineart = lineart(img, coarse=True)
# processed_image_lineart_anime = lineart_anime(img)
# processed_image_zoe = zoe(img)

# processed_image_canny = canny(img)
# processed_image_content = content(img)
# processed_image_mediapipe_face = face_detector(img)