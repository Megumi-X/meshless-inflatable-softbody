import os
import imageio
import shutil
from pathlib import Path
from log import create_folder, delete_folder

def export_gif(folder_name, gif_name, fps, name_prefix, name_suffix):
    frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f)) and f.startswith(name_prefix) and f.endswith(name_suffix)]
    frame_names = sorted(frame_names)

    # Read images.
    images = [imageio.v2.imread(f) for f in frame_names]
    if fps > 0:
        imageio.mimsave(gif_name, images, fps=fps)
    else:
        imageio.mimsave(gif_name, images)

def export_mp4(folder_name, mp4_name, fps, name_prefix, name_suffix):
    frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f)) and f.startswith(name_prefix) and f.endswith(name_suffix)]
    frame_names = sorted(frame_names)

    # Create a temporary folder.
    tmp_folder = Path("_export_mp4")
    create_folder(tmp_folder, exist_ok=False)
    for i, f in enumerate(frame_names):
        shutil.copyfile(f, tmp_folder / '{:08d}.png'.format(i))

    os.system("ffmpeg -r " + str(fps) + " -i " + str(tmp_folder / "%08d.png") + " -vcodec libx264 -y " + str(mp4_name))

    # Delete temporary folder.
    delete_folder(tmp_folder)