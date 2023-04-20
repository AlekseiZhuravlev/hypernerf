import time
from pathlib import Path

import mediapy
import numpy as np

from hypernerf import image_utils

def load_rendered_images(experiment_path):
    """Load rendered images from a directory.
    Args:
        experiment_path: Path to the experiment directory.
    Returns:
        A list of images.
    """
    img_dir = experiment_path / 'renders'
    img_paths = sorted(img_dir.glob('*.png'))
    images = [image_utils.load_image(path) for path in img_paths]
    return images

def load_ground_truth_images(img_dir):
    """Load rendered images from a directory.
    Args:
        img_dir: Path to the directory with ground truth images.
    Returns:
        A list of images.
    """
    img_paths = sorted(img_dir.glob('*.png'))
    images = [image_utils.load_image(path) for path in img_paths]
    return images

def concatenate_images(rendered_images, gt_images):
    """Concatenate rendered images and ground truth images.
    Args:
        rendered_images: A list of rendered images.
        gt_images: A list of ground truth images.

    Returns:
        A list of concatenated images.
    """
    images = []
    for rendered_image, gt_image in zip(rendered_images, gt_images[:len(rendered_images)]):
        frame = np.concatenate([rendered_image, gt_image], axis=1)
        images.append(frame)
    return images

def save_video(experiment_path, images):
    """Save a video from a list of images.
    Args:
        experiment_path: Path to the experiment directory.
        images: A list of images.
    """
    save_folder = experiment_path / 'videos'
    save_folder.mkdir(exist_ok=True)

    mediapy.write_video(save_folder / f'video_{int(time.time())}.mp4', images, fps=15)



if __name__ == '__main__':
    experiment_path = Path('/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/Exp_03/')

    rendered_images = load_rendered_images(experiment_path)
    gt_images = load_ground_truth_images(Path('/itet-stor/azhuavlev/net_scratch/Projects/Data/HyperNerf/hand1-dense-v2/rgb/4x/'))

    concatenated_images = concatenate_images(rendered_images, gt_images)
    save_video(experiment_path, concatenated_images)
