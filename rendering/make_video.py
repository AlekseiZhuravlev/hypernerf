import os
import sys
import time
from pathlib import Path

import cv2
import mediapy
import numpy as np

sys.path.append('..')
from hypernerf import image_utils

def load_all_images_from_dir(img_dir, extention='png'):
    """Load rendered images from a directory.
    Args:
        experiment_path: Path to the experiment directory.
    Returns:
        A list of images.
    """
    img_paths = sorted(img_dir.glob(f'*.{extention}'))
    images = [image_utils.load_image(path) for path in img_paths]
    return images


def load_depth_rgb_from_eval_dir(experiment_path, sub_dir='renders/00250000/train/train'):
    """Load rendered images from a directory.
    Args:
        experiment_path: Path to the experiment directory.
        sub_dir: Subdirectory with rendered images.
    Returns:
        A list of rgb, depth images, and a list of image ids.
    """
    img_dir = experiment_path / sub_dir
    rgb_paths = sorted(img_dir.glob('rgb_*.png'))
    depth_paths = sorted(img_dir.glob('depth_median_viz_*.png'))

    rgb_images = [image_utils.load_image(path) for path in rgb_paths]
    depth_images = [image_utils.load_image(path) for path in depth_paths]
    img_ids = [path.stem.split('_')[-1] for path in rgb_paths]

    return rgb_images, depth_images, img_ids


def load_ground_truth_images(img_dir, img_ids=None, img_ext='png'):
    """Load rendered images from a directory.
    Args:
        img_dir: Path to the directory with ground truth images.
        img_ids: A list of image ids to load. If None, will load all images.
    Returns:
        A list of images.
    """
    # load images with ids from img_ids
    if img_ids is not None:
        img_paths = [img_dir / f'{img_id}.{img_ext}' for img_id in img_ids]
    else:
        img_paths = sorted(img_dir.glob(f'*.{img_ext}'))

    images = [image_utils.load_image(path) for path in img_paths]
    return images


def concatenate_images(images_1, images_2, num_images=None):
    """Concatenate rendered images and ground truth images.
    Args:
        images_1: A list of images to be concatenated on the left.
        images_2: A list of images to be concatenated on the right.
        num_images: Number of images to concatenate. If None, will be set to the length of images_1.
    Returns:
        A list of concatenated images.
    """
    if num_images is None:
        num_images = len(images_1)

    images = []
    for image_1, image_2 in zip(images_1[:num_images], images_2[:num_images]):
        frame = np.concatenate([image_1, image_2], axis=1)
        images.append(frame)
    return images


def save_video(experiment_path, images, fps):
    """Save a video from a list of images.
    Args:
        experiment_path: Path to the experiment directory.
        images: A list of images.
    """
    save_folder = experiment_path / 'videos'
    save_folder.mkdir(exist_ok=True)

    mediapy.write_video(save_folder / f'video_{int(time.time())}.mp4', images, fps=fps)


def put_text_on_image(image, text):
    """Put text on an image.
    Args:
        image: An image.
        text: Text to put on the image.
    Returns:
        An image with text.
    """
    cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


def rename_folders():
    """Rename folders from 'renders' to 'images'."""
    experiments_dir = '/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/'
    for exp_dir in os.listdir(experiments_dir):
        if exp_dir.startswith('Exp_'):
            exp_dir_path = Path(experiments_dir, exp_dir)
            renders_path = Path(exp_dir_path, 'renders')
            images_path = Path(exp_dir_path, 'images')
            if renders_path.exists():
                print(f'Rename {renders_path} to {images_path}')
                os.rename(renders_path, images_path)


def render_video_from_render_images_script():
    experiment_path = Path('/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/Exp_06_spat_point/')

    rendered_images = load_all_images_from_dir(experiment_path)
    gt_images = load_ground_truth_images(Path('/itet-stor/azhuavlev/net_scratch/Projects/Data/HyperNerf/hand1-dense-v2/rgb/4x/'))

    concatenated_images = concatenate_images(rendered_images, gt_images)
    save_video(experiment_path, concatenated_images)


if __name__ == '__main__':

    experiment_path = Path('/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/Exp_17_ho3d_200/')
# '/home/azhuavlev/Desktop/Projects/Results/HyperNerf/Exp_17_ho3d_200/renders/00200000/train/train/'
    rgb_images, depth_images, image_ids = load_depth_rgb_from_eval_dir(
        experiment_path,
        sub_dir='renders/00200000/train/train/'
    )
    print('Loaded rendered images')

#/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/Exp_17_ho3d_200/ --gin_bindings="data_dir='
#/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/Exp_16_interhand/ --gin_bindings="data_dir='/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand_Nerfies_format/04/'"

    gt_images = load_ground_truth_images(
        # Path('/home/azhuavlev/Desktop/Projects/Data/InterHand_Nerfies_format/03/rgb/4x/'),
        Path('/home/azhuavlev/Desktop/Projects/Data/HO3D_nerfies/02/rgb/4x/'),
        image_ids,
        img_ext='png'
    )

    print('Loaded ground truth images')

    for i in range(len(gt_images)):
        gt_images[i] = put_text_on_image(gt_images[i], f'Ground truth')

    for i in range(len(rgb_images)):
        rgb_images[i] = put_text_on_image(rgb_images[i], f'Predicted RGB')

    for i in range(len(depth_images)):
        depth_images[i] = put_text_on_image(depth_images[i], f'Predicted depth')

    rgb_depth_images = concatenate_images(rgb_images, depth_images)
    concatenated_images = concatenate_images(rgb_depth_images, gt_images)

    print(len(rgb_images), len(depth_images), len(gt_images), len(concatenated_images))

    save_video(experiment_path, concatenated_images, 5)
