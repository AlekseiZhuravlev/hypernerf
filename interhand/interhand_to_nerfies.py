import json
import os


class InterhandToNerfiesConverter:
    def __int__(self, split, capture_n, pose, cameras_list):
        """
        :param split: 'train', 'test' or 'val'
        :param capture_n: capture number
        :param pose: hand pose
        :param cameras_list: list of cameras to use
        """
        self.split = split
        self.capture_n = capture_n
        self.pose = pose

        self.base_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand/images'
        self.pose_path = self.base_folder + '/' + split + '/' + capture_n + '/' + pose
        self.cameras_list = cameras_list

        self.target_folder = '/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand_Nerfies_format/01'
        self.camera_path = self.target_folder + '/camera'
        self.rgb_path = self.target_folder + '/rgb/4x'

        os.makedirs(self.target_folder, exist_ok=True)
        os.makedirs(self.camera_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)

        self.images_cameras = {}
        self.curr_img = 0

        with open(f'/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand/human_annot/'
                  f'InterHand2.6M_{self.split}_camera.json', 'r') as f:
            self.camera_params_dict = json.load(f)

        with open(f'/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand/human_annot/'
                  f'InterHand2.6M_{self.split}_data.json', 'r') as f:
            self.capture_data_dict = json.load(f)


    def copy_images(self):
    # copy images to rgb folder and create camera files
        for camera in self.cameras_list:

            camera_folder = self.pose_path + '/' + camera
            for img in os.listdir(camera_folder):

                # copy image to rgb folder
                os.system(f'cp {camera_folder}/{img} {self.rgb_path}/{img}')

                # rename copied image to curr_img, padded to 4 digits
                os.system(f'mv {self.rgb_path}/{img} {self.rgb_path}/{self.curr_img:04d}.jpg')

                self.create_camera_file(camera, self.curr_img)

                # save image-camera mapping
                self.images_cameras[f'{self.curr_img:04d}'] = camera
                self.curr_img += 1

    def create_camera_file(self, camera, img):

        # load camera parameters
        campos = self.camera_params_dict[self.capture_n]['campos'][camera]
        camrot = self.camera_params_dict[self.capture_n]['camrot'][camera]
        focal = self.camera_params_dict[self.capture_n]['focal'][camera]
        principt = self.camera_params_dict[self.capture_n]['principt'][camera]

        width = self.capture_data_dict[self.capture_n]['width']

        # create camera file
        with open(f'{self.camera_path}/{self.curr_img:04d}.txt', 'w') as f:
            f.write(f'{img} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0')


            """
            {
  "orientation": [
    [
      0.7264979481697083,
      0.5328468084335327,
      -0.433906614780426
    ],
    [
      -0.5865591168403625,
      0.8098118305206299,
      0.01237974688410759
    ],
    [
      0.3579792082309723,
      0.2455180287361145,
      0.9008727669715881
    ]
  ],
  "position": [
    -4.769414901733398,
    -1.0138546228408813,
    -0.2024407833814621
  ],
  "focal_length": 1528.52734375,
  "principal_point": [
    532.5533447265625,
    950.5166625976562
  ],
  "image_size": [
    1072,
    1920
  ]
}
"""



if __name__ == '__main__':
    interhand_to_nerfies(split='train',
                         capture_n='Capture0',
                            pose='0001_neutral_rigid',
                         cameras_list=['cam400002', 'cam400004','cam400006']
                         )



