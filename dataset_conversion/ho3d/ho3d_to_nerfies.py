import json
import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('../..')


class InterhandToNerfiesConverter:
    def __init__(self, basefolder, split, pose, cameras_list, experiment_n):
        """
        :param basefolder: path to the InterHand dataset
        :param split: 'train', 'test' or 'val'
        :param capture_n: capture number
        :param pose: hand pose
        :param cameras_list: list of cameras to use
        :param experiment_n: experiment number
        """
        self.split = split
        # self.capture_n = capture_n
        self.pose = pose

        self.base_folder = basefolder
        self.pose_path = self.base_folder + '/' + split + '/' + self.pose
        self.cameras_list = cameras_list

        self.camera_calibration_path = self.base_folder + '/' + 'calibration' + '/' + self.pose + '/' + 'calibration'

        self.target_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/HO3D_nerfies/' \
                             f'{experiment_n}'
        self.camera_path = self.target_folder + '/camera'
        self.rgb_path = self.target_folder + '/rgb/4x'

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)

        self.metadata = {} # dict to store time_id, warp_id, appearance_id, camera_id
        self.curr_img = 0

        self.allowed_images = self.load_allowed_images()
        self.dataset_images = []

        # todo: load camera parameters
        # load camera parameters
        # with open(self.base_folder + '/human_annot/' +\
        #           f'InterHand2.6M_{self.split}_camera.json', 'r') as f:
        #     self.camera_params_dict = json.load(f)


    def check_camera_img_count(self):
        """
        Check that all cameras have the same number of images
        """
        camera_img_count = []
        for camera in self.cameras_list:
            camera_folder = self.pose_path + f'{camera}'
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

            camera_folder = self.pose_path + f'{camera}/rgb'
            for j_image, img in enumerate(tqdm(sorted(os.listdir(camera_folder)))):
                if f'{self.pose}{camera}/{img[:-4]}' not in self.allowed_images:
                    print(f'{self.pose}{camera}/{img[:-4]} not in allowed images')
                    self.curr_img += 1
                    continue

                # print(img)
                # print(camera_folder)
                # copy image to rgb folder
                # print(f'cp {camera_folder}/{img} {self.rgb_path}/{img}')
                os.system(f'cp {camera_folder}/{img} {self.rgb_path}/{img}')

                # rename copied image to curr_img, padded to 4 digits
                if f'{img}' != f'{self.curr_img:04d}.jpg':
                    print(f'renaming {self.rgb_path}/{img} to {self.rgb_path}/{self.curr_img:04d}.jpg')
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

                self.dataset_images.append(self.curr_img)
                self.curr_img += 1

        # save metadata to file
        with open(f'{self.target_folder}/metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)

        # create dataset file
        self.create_dataset_file()

    def load_allowed_images(self):
        allowed_images_path = self.base_folder + '/train.txt'
        with open(allowed_images_path, 'r') as f:
            allowed_images = f.readlines()
        allowed_images = [img.strip() for img in allowed_images]
        return set(allowed_images)

    def create_camera_file(self, camera):
        with open(self.pose_path + f'{camera}' + f'/meta/{self.curr_img:04d}.pkl', 'rb') as f:
            camera_metadata = pickle.load(f)

        camera_intrinsics = camera_metadata['camMat']

        with open(self.camera_calibration_path + f'/trans_{camera}.txt', 'r') as f:
            # read matrix from txt file
            camera_matrix = np.loadtxt(f)

        # get camera position and orientation from camera matrix
        R = camera_matrix[:3, :3]
        t = camera_matrix[:3, 3]

        campos = (-R.T @ t).tolist()

        # TODO maybe this is wrong, no need to transpose?
        camrot = R.T.tolist()

        # convert capmos and camrot to lists of lists
        # campos = campos.tolist()
        # camrot = camrot.tolist()



        # get intrinsic parameters for x and y from camera matrix
        focal = [camera_intrinsics[0, 0], camera_intrinsics[1, 1]]
        princpt = [camera_intrinsics[0, 2], camera_intrinsics[1, 2]]
        skew = camera_intrinsics[0, 1]

        # open image
        img_jpg = Image.open(f'{self.rgb_path}/{self.curr_img:04d}.jpg')

        # downsample image by 4
        # img_jpg = img_jpg.resize((int(img_jpg.size[0] / 4), int(img_jpg.size[1] / 4)))

        # save image as png and remove jpg
        img_jpg.save(f'{self.rgb_path}/{self.curr_img:04d}.png')
        os.system(f'rm {self.rgb_path}/{self.curr_img:04d}.jpg')

        # get image size
        width, height = img_jpg.size

        # create camera parameters dict
        # campos = np.array(campos)
        camera_params = {
            'orientation': camrot,
            'position': campos,#list(campos / np.linalg.norm(campos)),
            'focal_length': focal[0],
            'principal_point': princpt,
            'image_size': [width * 4, height * 4],
            'skew': skew,
            'pixel_aspect_ratio': 1,
            'radial_distortion':[1e-4, 1e-4, 1e-4],
            'tangential_distortion': [1e-4, 1e-4]
        }
        # save camera parameters to file
        with open(f'{self.camera_path}/{self.curr_img:04d}.json', 'w') as f:
            json.dump(camera_params, f, indent=4)


    def create_dataset_file(self):
        dataset_params = {
            "count": len(self.dataset_images),
            "num_exemplars": len(self.dataset_images),
            "ids": [f"{i:04d}" for i in self.dataset_images],
            "train_ids": [f"{i:04d}" for i in self.dataset_images],
            "val_ids": []
        }
        with open(f'{self.target_folder}/dataset.json', 'w') as f:
            json.dump(dataset_params, f, indent=4)



if __name__ == '__main__':
    converter = InterhandToNerfiesConverter(
        basefolder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/HO3D_v3',
        split='train',
        pose='ABF1',
        cameras_list=['0'],#, '400284'],
        experiment_n='02'
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

