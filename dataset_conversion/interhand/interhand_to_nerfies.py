import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append('../..')


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

        # check that all cameras have the same number of images
        self.clear_root_folder()

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)

        self.metadata = {}  # dict to store time_id, warp_id, appearance_id, camera_id
        self.curr_img = 0

        # load camera parameters
        # with open(self.base_folder + '/human_annot/' + \
        #           f'InterHand2.6M_{self.split}_camera.json', 'r') as f:
        #     self.camera_params_dict = json.load(f)

        with open(self.base_folder + '/annotations/' + self.split + '/InterHand2.6M_' + split + '_camera.json') as f:
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

        return camera_img_count[0]

    def copy_images(self):
        """
        Copy images from interhand to nerfies format
        """

        n_images = self.check_camera_img_count()

        for j_image in tqdm(range(n_images)):
            # iterate over self.cameras_list in normal order on odd images and in reverse order on even images
            iteration_factor = 1 if j_image % 2 == 0 else -1

            for i_camera, camera in enumerate(self.cameras_list[::iteration_factor]):
                camera_folder = self.pose_path + '/' + f'cam{camera}'
                img = sorted(os.listdir(camera_folder))[j_image]
                self.copy_image(img, camera_folder)

                self.create_camera_file(camera)
                # self.update_metadata_unique(i_camera, j_image)
                self.update_metadata(i_camera, j_image)

                self.curr_img += 1


        # copy images to rgb folder and create camera files
        # for i_camera, camera in enumerate(self.cameras_list):
        #
        #     camera_folder = self.pose_path + '/' + f'cam{camera}'
        #     for j_image, img in enumerate(tqdm(sorted(os.listdir(camera_folder)))):
        #         # copy image
        #         self.copy_image(img, camera_folder)
        #
        #         self.create_camera_file(camera)
        #
        #         self.update_metadata(i_camera, j_image)
        #
        #         self.curr_img += 1

        # save metadata to file
        with open(f'{self.target_folder}/metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)

        # create test cameras

        # create dataset file
        self.create_dataset_file()

        # create scene file
        self.create_scene_file()

    def clear_root_folder(self):
        """
        Clear root folder from previous experiments
        """
        os.system(f'rm -rf {self.target_folder}/*')


    def update_metadata(self, i_camera, j_image):
        # update metadata
        self.metadata[f'{self.curr_img:04d}'] = {
            "time_id": j_image,
            "warp_id": j_image,
            "appearance_id": j_image,
            "camera_id": i_camera,
        }

    def update_metadata_unique(self, i_camera, j_image):
        # update metadata
        self.metadata[f'{self.curr_img:04d}'] = {
            "time_id": self.curr_img,
            "warp_id": self.curr_img,
            "appearance_id": self.curr_img,
            "camera_id": self.curr_img,
        }

    def copy_image(self, img, camera_folder):
        # copy image to rgb folder
        os.system(f'cp {camera_folder}/{img} {self.rgb_path}/{img}')

        # rename copied image to curr_img, padded to 4 digits
        os.system(f'mv {self.rgb_path}/{img} {self.rgb_path}/{self.curr_img:04d}.jpg')

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
        img = img_jpg

        # pad image with 50 black pixel on the left
        # img = Image.new('RGB', (img_jpg.size[0] + 50, img_jpg.size[1]), (0, 0, 0))
        # img.paste(img_jpg, (50, 0))

        # invert image colors
        # img_arr = np.array(img)
        # img_arr = 255 - img_arr
        # img = Image.fromarray(img_arr)

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
            'radial_distortion': [1e-4, 1e-4, 1e-4],
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

    def create_scene_file(self):
        scene_params = {
          "scale": 1,
          "scene_to_metric": 1,
          "center": [0, 0, 0],
          "near": 0.1,
          "far": 500
        }
        with open(f'{self.target_folder}/scene.json', 'w') as f:
            json.dump(scene_params, f, indent=4)


if __name__ == '__main__':
    converter = InterhandToNerfiesConverter(
        basefolder='/home/azhuavlev/Desktop/Projects/Data/Interhand_masked',
        split='test',
        capture_n='0',
        pose='ROM04_LT_Occlusion',
        cameras_list=['400262', '400263', '400264', '400265', '400284'],
        experiment_n='07_same_warp'
    )
    converter.copy_images()

    exit(0)
    # make video
    print(pathlib.Path(converter.rgb_path))
    imgs = make_video.load_all_images_from_dir(pathlib.Path(converter.rgb_path), 'png')
    print(len(imgs))
    concatenated_imgs = make_video.concatenate_images(imgs[:len(imgs) // 2], imgs[len(imgs) // 2:])
    print(len(concatenated_imgs))
    make_video.save_video(pathlib.Path(converter.target_folder), concatenated_imgs, fps=5)
