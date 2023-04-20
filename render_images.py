import functools
import os
import time
from pathlib import Path

import gin
import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from IPython.display import Markdown
from IPython.display import display
from absl import logging
from flax import jax_utils
from flax import optim
from flax.training import checkpoints
from jax import random
from tqdm import tqdm

from hypernerf import configs
from hypernerf import datasets
from hypernerf import evaluation
from hypernerf import image_utils
from hypernerf import model_utils
from hypernerf import models
from hypernerf import schedules
from hypernerf import training
from hypernerf import utils
from hypernerf import visualization as viz


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
    print(msg % args)




if __name__ == '__main__':


    logging.info = myprint
    logging.warn = myprint
    logging.error = myprint

    # sys.path.append('..')
    # print current working directory
    print('Current working directory:', Path.cwd())

    # change working directory to /home/azhuavlev/PycharmProjects/hypernerf/
    os.chdir("/home/azhuavlev/PycharmProjects/hypernerf/")

    # @markdown The working directory where the trained model is.
    train_dir = '/itet-stor/azhuavlev/net_scratch/Projects/Results/HyperNerf/Exp_03/'  # @param {type: "string"}
    # @markdown The directory to the dataset capture.
    data_dir = '/itet-stor/azhuavlev/net_scratch/Projects/Data/HyperNerf/hand1-dense-v2/'  # @param {type: "string"}

    camera_path = '/itet-stor/azhuavlev/net_scratch/Projects/Data/HyperNerf/hand1-dense-v2/camera/'  # @param {type: 'string'}
    images_path = Path(train_dir, 'renders')
    config_path = Path(train_dir, 'config.gin')


    checkpoint_dir = Path(train_dir, 'checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)


    # Load configs
    with open(config_path, 'r') as f:
        logging.info('Loading config from %s', config_path)
        config_str = f.read()
    gin.parse_config(config_str)

    with open(config_path, 'w') as f:
        logging.info('Saving config to %s', config_path)
        f.write(config_str)

    exp_config = configs.ExperimentConfig()
    train_config = configs.TrainConfig()
    eval_config = configs.EvalConfig()


    # Load dataset
    dummy_model = models.NerfModel({}, 0, 0)
    datasource = exp_config.datasource_cls(
        image_scale=exp_config.image_scale,
        random_seed=exp_config.random_seed,
        # Enable metadata based on model needs.
        use_warp_id=dummy_model.use_warp,
        use_appearance_id=(
                dummy_model.nerf_embed_key == 'appearance'
                or dummy_model.hyper_embed_key == 'appearance'),
        use_camera_id=dummy_model.nerf_embed_key == 'camera',
        use_time=True)
        # use_time=dummy_model.warp_embed_key == 'time')


    # Load the pre-trained model.
    rng = random.PRNGKey(exp_config.random_seed)
    np.random.seed(exp_config.random_seed + jax.process_index())
    devices_to_use = jax.devices()

    learning_rate_sched = schedules.from_config(train_config.lr_schedule)
    nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
    warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
    elastic_loss_weight_sched = schedules.from_config(
        train_config.elastic_loss_weight_schedule)
    hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
    hyper_sheet_alpha_sched = schedules.from_config(
        train_config.hyper_sheet_alpha_schedule)

    rng, key = random.split(rng)
    params = {}
    model, params['model'] = models.construct_nerf(
        key,
        batch_size=train_config.batch_size,
        embeddings_dict=datasource.embeddings_dict,
        near=datasource.near,
        far=datasource.far)

    optimizer_def = optim.Adam(learning_rate_sched(0))
    optimizer = optimizer_def.create(params)

    state = model_utils.TrainState(
        optimizer=optimizer,
        nerf_alpha=nerf_alpha_sched(0),
        warp_alpha=warp_alpha_sched(0),
        hyper_alpha=hyper_alpha_sched(0),
        hyper_sheet_alpha=hyper_sheet_alpha_sched(0))
    scalar_params = training.ScalarParams(
        learning_rate=learning_rate_sched(0),
        elastic_loss_weight=elastic_loss_weight_sched(0),
        warp_reg_loss_weight=train_config.warp_reg_loss_weight,
        warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
        warp_reg_loss_scale=train_config.warp_reg_loss_scale,
        background_loss_weight=train_config.background_loss_weight,
        hyper_reg_loss_weight=train_config.hyper_reg_loss_weight)

    logging.info('Restoring checkpoint from %s', checkpoint_dir)
    state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    step = state.optimizer.state.step + 1
    state = jax_utils.replicate(state, devices=devices_to_use)
    del params


    # Define pmapped render function.

    devices = jax.devices()
    def _model_fn(key_0, key_1, params, rays_dict, extra_params):
        # for key, value in extra_params.items():
        #     print(key, value)
        # # print(extra_params)
        # exit(0)

        out = model.apply({'params': params},
                          rays_dict,
                          extra_params=extra_params,
                          rngs={
                              'coarse': key_0,
                              'fine': key_1
                          },
                          mutable=False)
        return jax.lax.all_gather(out, axis_name='batch')


    pmodel_fn = jax.pmap(
        # Note rng_keys are useless in eval mode since there's no randomness.
        _model_fn,
        in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
        devices=devices_to_use,
        axis_name='batch',
    )

    render_fn = functools.partial(evaluation.render_image,
                                  model_fn=pmodel_fn,
                                  device_count=len(devices),
                                  chunk=eval_config.chunk)


    # Load cameras.
    camera_dir = Path(data_dir, camera_path)
    print(f'Loading cameras from {camera_dir}')
    test_camera_paths = datasource.glob_cameras(camera_dir)
    test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)


    # Render video frames.
    rng = rng + jax.process_index()  # Make random seed separate across hosts.
    keys = random.split(rng, len(devices))

    results = []
    frames = []
    for i in tqdm(range(len(test_cameras))):
        print(f'Rendering frame {i + 1}/{len(test_cameras)}')
        camera = test_cameras[i]
        batch = datasets.camera_to_rays(camera)
        batch['metadata'] = {
            'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
            'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
        }

        # Render the image.
        render = render_fn(state, batch, rng=rng)

        # Save the rendered image.
        rgb = np.array(render['rgb'])
        depth_med = np.array(render['med_depth'])
        results.append((rgb, depth_med))
        depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)

        frame = np.concatenate([rgb, depth_viz], axis=1)
        mediapy.write_image(image=image_utils.image_to_uint8(frame),
                            path=Path(images_path, f'frame_{i:04d}.png'))

        frames.append(image_utils.image_to_uint8(frame))

    # Save the video.
    mediapy.write_video(path=Path(images_path, f'video.mp4'), images=frames, fps=15)
