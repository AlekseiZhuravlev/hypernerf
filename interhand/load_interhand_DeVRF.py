import os
import numpy as np
import imageio
import json
import cv2
import matplotlib.pyplot as plt

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_images(base_folder):
    """
    For a scene, load images from all cameras.
    :param base_folder: contains folders with the images per camera
    :return: dict with camera names as keys and list of images as values
    """
    images = {}

    for camera_name in os.listdir(base_folder):
        image_folder = os.path.join(base_folder, camera_name)

        # Camera folder name is 'cam0001', keep only the id
        images[camera_name[3:]] = []

        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            image = (imageio.imread(image_path) / 255.).astype(np.float32)

            image = invert_image(image)

            # show image opencv
            # cv2.imshow('image', image)
            # cv2.waitKey(0)

            images[camera_name[3:]].append(image)
    return images

# TODO remove only pitch black color
# replace img intensity values < 0.01 with 1
def invert_image(img):
    # convert to grayscale opencv
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert colors
    gray = 1 - gray


    return np.stack((gray,)*3, axis=-1)

def make_camera_matrix(campos, camrot):
    """
    Create a camera matrix from the camera position and rotation
    :param campos: camera position
    :param camrot: camera rotation
    :return: camera matrix
    """
    camrot = np.array(camrot).reshape(3, 3)
    campos = np.array(campos)

    # my implementation may be wrong, this is what the original code does
    # print(-np.dot(camrot,campos.reshape(3,1)).reshape(3))

    camrot = np.transpose(camrot)
    campos = -np.matmul(camrot, campos)

    camera_matrix = np.zeros((4, 4))
    camera_matrix[:3, :3] = camrot
    camera_matrix[:, 3] = np.append(campos, 1)

    return camera_matrix


def check_if_img_corrupt(img, camera_name):
    """
    Check if image is corrupt by checking if the image is more than 90% black
    :param img: image to check
    :param camera_name: name of the camera
    :return: True if corrupt, False otherwise
    """
    if np.sum(img < 0.05) / (img.shape[0] * img.shape[1] * img.shape[2]) > 0.9:
        print(f'Image corrupt, camera: {camera_name}')
        return True
    return False


def check_if_horizontal(img):
    return img.shape[0] < img.shape[1]


def load_interhand_data(basedir, half_res=False, testskip=1,
                        run_type='train', capture_n='0', scene_name='0000_neutral_relaxed'):
    with open(os.path.join(basedir, f'human_annot/InterHand2.6M_{run_type}_camera.json'), 'r') as fp:
        camera_params = json.load(fp)

    cameras_imgs_dict = load_images(os.path.join(basedir, 'images', run_type, f'Capture{capture_n}', scene_name))

    all_poses = []
    all_imgs = []
    all_Ks = []
    for camera_name, img_list in cameras_imgs_dict.items():
        for img in img_list:
            # drop if image is > 90% black or is horizontal
            if check_if_img_corrupt(img, camera_name) or check_if_horizontal(img):
                continue

            campos = camera_params[capture_n]['campos'][camera_name]
            camrot = camera_params[capture_n]['camrot'][camera_name]
            focal = camera_params[capture_n]['focal'][camera_name]
            princpt = camera_params[capture_n]['princpt'][camera_name]

            camera_matrix = make_camera_matrix(campos, camrot)

            all_imgs.append(img)
            all_poses.append(camera_matrix)
            all_Ks.append(np.array([[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]]))

        # print(f'Camera {camera_name}')
        # plt.imshow(img)
        # plt.show()

    all_imgs = np.array(all_imgs)  # keep all 4 channels (RGBA)
    all_poses = np.array(all_poses).astype(np.float32)
    all_Ks = np.array(all_Ks).astype(np.float32)

    # load meta data
    H, W = all_imgs[0].shape[:2]

    # TODO: we use the focal length from the last camera, but it should be the different for all cameras, and same for x and y

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # if half_res:
    #     H = H // 2
    #     W = W // 2
    #     focal = focal / 2.
    #     princpt = princpt / 2.
    #
    #     imgs_half_res = np.zeros((all_imgs.shape[0], H, W, 4))
    #     for i, img in enumerate(all_imgs):
    #         imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    #     all_imgs = imgs_half_res

    # split into train/val/test
    # TODO which one is which?

    arr = np.arange(len(all_imgs))
    train, val, test = np.split(arr, [int(.7*len(arr)), int(.85*len(arr))])
    i_split = np.array([train, val, test])

    # print(all_imgs.shape, all_poses.shape, render_poses.shape, [H, W, focal], i_split)

    return all_imgs, all_poses, render_poses, [H, W, focal[0]], i_split, all_Ks[0]


if __name__ == '__main__':
    load_interhand_data('/itet-stor/azhuavlev/net_scratch/Projects/Data/InterHand', half_res=False,
                         run_type='train', capture_n='0', scene_name='0000_neutral_relaxed')
