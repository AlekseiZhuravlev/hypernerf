# @title Generate camera trajectory.
import json
import math
import os
import sys

import numpy as np
from jax import numpy as jnp
from scipy import interpolate
from tensorflow_graphics.geometry.representation.ray import triangulate as ray_triangulate

sys.path.append('..')
from hypernerf.utils import points_bounding_size
from hypernerf.camera import Camera


class TestCameraGenerator:

  def __init__(self, root_dir):
    self.root_dir = root_dir

    self.ref_cameras = self.load_ref_cameras()
    self.ref_cameras = self.ref_cameras
    print(f'Loaded {len(self.ref_cameras)} reference cameras.')
    self.camera_paths = None


  def load_ref_cameras(self):
    """Loads the reference cameras for the scene."""

    cameras_list = []
    cameras_dir = self.root_dir + '/camera'

    for camera_path in sorted(os.listdir(cameras_dir)):

        camera = Camera.from_json(cameras_dir + '/' + camera_path)
        cameras_list.append(camera)

    return cameras_list


  def compute_camera_rays(self, points, camera):
    origins = np.broadcast_to(camera.position[None, :], (points.shape[0], 3))
    directions = camera.pixels_to_rays(points.astype(jnp.float32))
    endpoints = origins + directions
    return origins, endpoints


  def triangulate_rays(self, origins, directions):
    origins = origins[np.newaxis, ...].astype('float32')
    directions = directions[np.newaxis, ...].astype('float32')
    weights = np.ones(origins.shape[:2], dtype=np.float32)
    points = np.array(ray_triangulate(origins, origins + directions, weights))
    return points.squeeze()


  def generate_test_cameras(self):
    """Generates a set of test cameras for the scene."""

    origins = np.array([c.position for c in self.ref_cameras])
    directions = np.array([c.optical_axis for c in self.ref_cameras])
    look_at = self.triangulate_rays(origins, directions)
    print('look_at', look_at)

    avg_position = np.mean(origins, axis=0)
    print('avg_position', avg_position)

    up = -np.mean([c.orientation[..., 1] for c in self.ref_cameras], axis=0)
    print('up', up)

    bounding_size = points_bounding_size(origins) / 2
    x_scale =   0.75# @param {type: 'number'}
    y_scale = 0.75  # @param {type: 'number'}
    xs = x_scale * bounding_size
    ys = y_scale * bounding_size
    radius = 0.75  # @param {type: 'number'}
    num_frames = 100  # @param {type: 'number'}


    origin = np.zeros(3)

    ref_camera = self.ref_cameras[0]
    print(ref_camera.position)
    z_offset = -0.1

    angles = np.linspace(0, 2*math.pi, num=num_frames)
    positions = []
    for angle in angles:
      x = np.cos(angle) * radius * xs
      y = np.sin(angle) * radius * ys
      # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
      # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

      position = np.array([x, y, z_offset])
      # Make distance to reference point constant.
      position = avg_position + position
      positions.append(position)

    positions = np.stack(positions)

    orbit_cameras = []
    for position in positions:
      camera = ref_camera.look_at(position, look_at, up)
      orbit_cameras.append(camera)

    self.camera_paths = {'orbit-mild': orbit_cameras}


  def save_test_cameras(self):
    """Save test cameras to disk."""

    test_camera_dir = self.root_dir + '/camera-paths'
    for test_path_name, test_cameras in self.camera_paths.items():
      out_dir = test_camera_dir + '/' + test_path_name
      os.makedirs(out_dir, exist_ok=True)

      for i, camera in enumerate(test_cameras):
        camera_path = out_dir + f'/{i:04d}.json'
        print(f'Saving camera to {camera_path!s}')
        with open(camera_path, 'w') as f:
          json.dump(camera.to_json(), f, indent=2)


if __name__ == '__main__':
  camera_generator = TestCameraGenerator(
    '/home/azhuavlev/Desktop/Projects/Data/InterHand_Nerfies_format/07_same_warp'
  )
  print(camera_generator.ref_cameras)
  camera_generator.generate_test_cameras()
  camera_generator.save_test_cameras()