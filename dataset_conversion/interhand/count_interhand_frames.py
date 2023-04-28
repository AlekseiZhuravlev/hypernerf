import os
import pathlib

from tqdm import tqdm

base_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand/images/train'

frames_count = []
for capture_n in tqdm(os.listdir(base_folder)[:]):
    capture_dir = base_folder + '/' + capture_n
    for pose_n in os.listdir(capture_dir):
        pose_dir = capture_dir + '/' + pose_n
        for cam_n in os.listdir(pose_dir):
            cam_dir = pose_dir + '/' + cam_n
            frames_count.append((
                capture_n + '/' + pose_n + '/' + cam_n,
                len(os.listdir(cam_dir))
            ))

# print(sorted(frames_count, key=lambda x: x[1]))

# write sorted data to file
with open('interhand_frames_count.txt', 'w') as f:
    for item in sorted(frames_count, key=lambda x: x[1]):
        f.write(f"{item[0]} {item[1]}\n")

