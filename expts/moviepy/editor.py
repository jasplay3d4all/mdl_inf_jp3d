from moviepy.editor import concatenate_videoclips
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.fx import resize, rotate, crop
# from moviepy.video.fx.all import crossfadein, crossfadeout

# Step 2: Load the images and set duration
image_paths = [
    "../../share_vol/data_io/JAYKANIDAN/stg_3/00000.png",
    "../../share_vol/data_io/JAYKANIDAN/stg_4/00000.png",
    "../../share_vol/data_io/JAYKANIDAN/stg_5/00000.png"
]  # Add paths to your images here
duration_per_image = 2  # Set the duration for each image in seconds

# Create ImageSequenceClip from the images with transformations
image_clips = []
for img_path in image_paths:
    clip = ImageSequenceClip([img_path], durations=[duration_per_image])

    # Apply transformations (pan, zoom, and rotation) to each image
    pan_x, pan_y = 0.2, 0.1  # Adjust pan values to pan by 20% and 10% of the frame width and height, respectively
    zoom_factor = 1.2  # Zoom in by 20%
    rotation_angle = 30  # Rotate the image by 30 degrees counter-clockwise

    clip = (clip.fx(resize, newsize=(int(clip.w * zoom_factor), int(clip.h * zoom_factor)))  # Zoom the image
            .fx(rotate, rotation_angle)  # Rotate the image
            .fx(crop, x_center = clip.w*(1-pan_x), y_center = clip.h*(1-pan_y))  # Pan the image
            .set_duration(duration_per_image)  # Set the duration after transformations
           )
    image_clips.append(clip)

# Apply crossfade transition
transition_duration = 1
# image_clips = [clip.fx(crossfadein, transition_duration).fx(crossfadeout, transition_duration) for clip in image_clips]

# Step 3: Create a video clip with transitions
video_clip = concatenate_videoclips(image_clips, method="compose")

# Step 4: Save the final transition video
video_clip.write_videofile("transition_video_with_effects.mp4", codec="libx264")
