import json
import os
import pathlib

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('../..')

import make_video


class InterhandToNerfiesConverter:
    def __init__(self, basefolder, split, capture_n, pose, cameras_list, experiment_n):
        """
        :param basefolder: path to the InterHand dataset
        :param split: 'train', 'test' or 'val'
        :param capture_n: capture number
        :param pose: hand pose
        :param cameras_list: list of cameras to use
        :param experiment_n: experiment number
        """
        self.split = split
        self.capture_n = capture_n
        self.pose = pose

        self.base_folder = basefolder
        self.pose_path = self.base_folder + '/' + 'images' + '/' + split + '/' + \
                         f"Capture{capture_n}" + '/' + pose
        self.cameras_list = cameras_list

        self.target_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand_Nerfies_format/' \
                             f'{experiment_n}'
        self.camera_path = self.target_folder + '/camera'
        self.rgb_path = self.target_folder + '/rgb/4x'

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)

        self.metadata = {} # dict to store time_id, warp_id, appearance_id, camera_id
        self.curr_img = 0

        # load camera parameters
        with open(self.base_folder + '/human_annot/' +\
                  f'InterHand2.6M_{self.split}_camera.json', 'r') as f:
            self.camera_params_dict = json.load(f)


    def check_camera_img_count(self):
        """
        Check that all cameras have the same number of images
        """
        camera_img_count = []
        for camera in self.cameras_list:
            camera_folder = self.pose_path + '/' + f'cam{camera}'
            camera_img_count.append(len(os.listdir(camera_folder)))
        assert len(set(camera_img_count)) == 1, 'Cameras have different number of images'
        print('All cameras have the same number of images')

    def copy_images(self):
        """
        Copy images from interhand to nerfies format
        """

        # check that all cameras have the same number of images
        self.check_camera_img_count()

        # copy images to rgb folder and create camera files
        for i_camera, camera in enumerate(self.cameras_list):

            camera_folder = self.pose_path + '/' + f'cam{camera}'
            for j_image, img in enumerate(tqdm(sorted(os.listdir(camera_folder)))):

                # copy image to rgb folder
                os.system(f'cp {camera_folder}/{img} {self.rgb_path}/{img}')

                # rename copied image to curr_img, padded to 4 digits
                os.system(f'mv {self.rgb_path}/{img} {self.rgb_path}/{self.curr_img:04d}.jpg')

                # get image size
                self.create_camera_file(camera)

                # update metadata
                self.metadata[f'{self.curr_img:04d}'] = {
                    "time_id": j_image,
                    "warp_id": j_image,
                    "appearance_id": j_image,
                    "camera_id": i_camera,
                }
                self.curr_img += 1

        # save metadata to file
        with open(f'{self.target_folder}/metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)

        # create dataset file
        self.create_dataset_file()


    def create_camera_file(self, camera):

        # load camera parameters
        campos = self.camera_params_dict[self.capture_n]['campos'][camera]
        camrot = self.camera_params_dict[self.capture_n]['camrot'][camera]
        focal = self.camera_params_dict[self.capture_n]['focal'][camera]
        princpt = self.camera_params_dict[self.capture_n]['princpt'][camera]

        # open image
        img_jpg = Image.open(f'{self.rgb_path}/{self.curr_img:04d}.jpg')

        # img_jpg.show()
        # img_jpg.save(f'example.png')
        # exit(
        # img = img_jpg

        # pad image with 50 black pixel on the left
        img = Image.new('RGB', (img_jpg.size[0] + 50, img_jpg.size[1]), (0, 0, 0))
        img.paste(img_jpg, (50, 0))

        # invert image colors
        # img_arr = np.array(img)
        # img_arr = 255 - img_arr
        # img = Image.fromarray(img_arr)



        # for i in range(img_jpg.size[0]):
        #     for j in range(img_jpg.size[1]):
        #         # invert pixel color
        #             img_jpg.putpixel((i, j), (255 - img_jpg.getpixel((i, j))[0],
        #                                           255 - img_jpg.getpixel((i, j))[1],
        #                                           255 - img_jpg.getpixel((i, j))[2]))



        # save image as png and remove jpg
        img.save(f'{self.rgb_path}/{self.curr_img:04d}.png')
        os.system(f'rm {self.rgb_path}/{self.curr_img:04d}.jpg')

        # get image size
        width, height = img.size

        # create camera parameters dict
        campos = np.array(campos)
        camera_params = {
            'orientation': camrot,
            'position': list(campos / np.linalg.norm(campos)),
            'focal_length': focal[0],
            'principal_point': princpt,
            'image_size': [width * 4, height * 4],
            'skew': 0,
            'pixel_aspect_ratio': 1,
            'radial_distortion':[1e-4, 1e-4, 1e-4],
            'tangential_distortion': [1e-4, 1e-4]
        }
        # save camera parameters to file
        with open(f'{self.camera_path}/{self.curr_img:04d}.json', 'w') as f:
            json.dump(camera_params, f, indent=4)


    def create_dataset_file(self):
        dataset_params = {
            "count": self.curr_img,
            "num_exemplars": self.curr_img,
            "ids": [f"{i:04d}" for i in range(self.curr_img)],
            "train_ids": [f"{i:04d}" for i in range(self.curr_img)],
            "val_ids": []
        }
        with open(f'{self.target_folder}/dataset.json', 'w') as f:
            json.dump(dataset_params, f, indent=4)



if __name__ == '__main__':
    converter = InterhandToNerfiesConverter(
        basefolder = '/itet-stor/azhuavlev/net_scratch/Downloads/Interhand_30fps/InterHand2.6M_30fps_batch1',
        split='test',
        capture_n='1',
        pose='ROM04_LT_Occlusion',
        cameras_list=['400262'],#, '400284'],
        experiment_n='04'
    )
    converter.copy_images()

    exit(0)
    # make video
    print(pathlib.Path(converter.rgb_path))
    imgs = make_video.load_all_images_from_dir(pathlib.Path(converter.rgb_path), 'png')
    print(len(imgs))
    concatenated_imgs = make_video.concatenate_images(imgs[:len(imgs)//2], imgs[len(imgs)//2:])
    print(len(concatenated_imgs))
    make_video.save_video(pathlib.Path(converter.target_folder), concatenated_imgs, fps=5)

