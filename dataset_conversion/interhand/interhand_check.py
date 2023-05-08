import pathlib
from tqdm import tqdm

basefolder = pathlib.Path('/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand/images/train/')

def count_interhand_images(basefolder):
    """
    Count how many images are for each camera, pose, and capture for the InterHand dataset.
    Print the results in sorted order.

    Parameters
    ----------
    basefolder : pathlib.Path
        Path to the root folder of the InterHand dataset. Can be either the train, val or test folder.
    """

    pose_img_counts = []
    for capture_folder in tqdm(list(basefolder.iterdir())):
        for pose_folder in capture_folder.iterdir():
            for camera_folder in pose_folder.iterdir():
                # count how many images are in folder
                num_images = len(list(camera_folder.glob('*.jpg')))
                pose_img_counts.append((f'{capture_folder.name}/{pose_folder.name}/{camera_folder.name}', num_images))
    for i in sorted(pose_img_counts, key=lambda x: x[1]):
        print(i)

if __name__ == '__main__':
    count_interhand_images(basefolder)