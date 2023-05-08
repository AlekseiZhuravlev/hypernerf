import glob
import json
import sys
import numpy as np

# import this /home/azhuavlev/PycharmProjects/extrinsic2pyramid/
sys.path.append('/home/azhuavlev/PycharmProjects/extrinsic2pyramid')

from util.camera_pose_visualizer import CameraPoseVisualizer

# visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 100])
# visualizer.extrinsic2pyramid(np.eye(4), 'c', 10)
# visualizer.show()

def load_camera_json(path_list):
    """
    Load camera parameters from json files
    :param path_list: list of paths to json files
    :return: list of camera parameters (dicts)
    """
    camera_params_list = []
    for path in path_list:
        with open(path) as f:
            camera_params_list.append(json.load(f))
    return camera_params_list

def construct_extrinsic_camera_matrix(camera_params_list):
    """
    Construct extrinsic camera matrix from camera parameters
    :param camera_params_list: list of camera parameters (dicts)
    :return: list of camera matrices
    """
    camera_matrix_list = []
    for camera_params in camera_params_list:
        cam_orientation = np.array(camera_params['orientation'])
        cam_position = np.array(camera_params['position'])

        t = -cam_orientation.T @ cam_position
        R = cam_orientation.T



        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = np.array(camera_params['orientation'])
        camera_matrix[:3, 3] = np.array(camera_params['position'])
        camera_matrix_list.append(camera_matrix)
    return camera_matrix_list


def visualize_cameras(camera_matrices_list):
    """
    Visualize cameras from camera parameters
    :param camera_matrices_list: list of camera matrices
    """
    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 10])
    for i, camera_matrix in enumerate(camera_matrices_list):
        # generate color using indexes for matplotlib
        color = 'C' + str(i)



        visualizer.extrinsic2pyramid(camera_matrix, color)
    visualizer.show()




if __name__ == '__main__':

    # basedir = '/home/azhuavlev/Desktop/Projects/Data/InterHand_Nerfies_format/06/camera'
    basedir = '/home/azhuavlev/Desktop/Projects/Data/InterHand_Nerfies_format/06/camera-paths/orbit-mild'
    # # basedir = '/home/azhuavlev/Desktop/Projects/Data/HyperNerf/hand1-dense-v2/camera/'

    # get all json paths from directory, as absolute paths
    json_paths = sorted(glob.glob(f'{basedir}/*.json'))

    print(json_paths[:300])
    camera_params = load_camera_json(
        json_paths[:300]
    )
    camera_matrix = construct_extrinsic_camera_matrix(camera_params)
    visualize_cameras(camera_matrix)