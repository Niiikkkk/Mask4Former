import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer.panoptic import Panoptic
from utils.utils import flatten_dict
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import torch


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg.general.ckpt_path = f"{cfg.general.save_dir}/last.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    #model = Panoptic4D(cfg)
    model = Panoptic(cfg)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers

def voxelize(coordinates, features, labels, voxel_size):
    coordinates = np.floor(coordinates / voxel_size).astype(np.int32)
    coordinates = coordinates - np.min(coordinates, axis=0)

    _, unique_map, inverse_map = np.unique(
        coordinates, return_index=True, return_inverse=True, axis=0
    )

    sample_c = torch.from_numpy(coordinates[unique_map])
    sample_f = torch.from_numpy(features[unique_map])
    sample_l = torch.from_numpy(labels[unique_map])
    return sample_c, sample_f, sample_l, inverse_map

def read_lidar(filepath, voxel_size):
    lidar_data = np.load(filepath)
    points = np.frombuffer(lidar_data, dtype="float32").reshape(-1, 4)
    features = points[: , -1]
    points = points[:, :3]
    center_coordinate = points.mean(0)
    features = np.vstack(features)

    v_min = np.percentile(features, 1)  # Ignore lowest 1%
    v_max = np.percentile(features, 99)  # Ignore highest 1%
    features = np.clip(features, v_min, v_max)
    features = (features - v_min) / (v_max - v_min + 1e-8)

    time_array = np.zeros((features.shape[0], 1), dtype=np.float32)
    features = np.hstack((time_array, features))

    features = np.hstack(
        (
            features,
            np.linalg.norm(points - center_coordinate, axis=1)[
                :, np.newaxis
            ],
        )
    )
    features = np.hstack((points, features))
    acc_num_points = len(points)

    #Voxelize the point cloud
    sample_c, sample_f, sample_l, inverse_map = voxelize(points, features, np.zeros((len(points), 2)), voxel_size)
    sample_c = torch.hstack((torch.zeros((sample_c.shape[0], 1)) , sample_c)).int()
    raw_coordinates = sample_f[:, :4]
    features = sample_f[:, 4:]

    return {
        "raw_coordinates": raw_coordinates,
        "features": features,
        "num_points": acc_num_points,
        "coordinates": sample_c,
    }

def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(cfg.general.save_dir),
    )
    runner.test(model=model, ckpt_path=cfg.general.ckpt_path)

@hydra.main(
    #version_base=None, config_path="conf", config_name="config_panoptic_4d.yaml"
    version_base=None, config_path="conf", config_name="config_panoptic_4d_carla.yaml"
)
def main(cfg: DictConfig):

    test(cfg)
    exit()
    cfg, model, loggers = get_parameters(cfg)
    lidar_dir = "/anomlay_carla/1/lidar/raw/lidar-3414.npy"
    x = read_lidar(lidar_dir, voxel_size=0.1)
    print(model.device)
    output = model.model(x["coordinates"], x["features"], x["raw_coordinates"], model.device ,is_eval=True)
    print(output)



if __name__ == "__main__":
    main()
