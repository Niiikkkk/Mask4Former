import os
import re
import sys

import numpy as np
import random
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from fire import Fire
from datasets.utils import (
    parse_calibration,
    parse_poses,
    load_yaml,
    save_database,
    merge_trainval,
)


class CarlaPreprocessing:
    def __init__(
        self,
        data_dir: str = "/home/nicholas/Desktop/main_UE4/output",
        save_dir: str = "/home/nicholas/preprocessed_anomalies/",
        generate_instances: bool = True,
        modes: tuple = ("train", "validation", "test"),
    ):
        random.seed(42)
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.generate_instances = generate_instances
        self.modes = modes

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if generate_instances:
            self.instances_dir = self.save_dir / "instances"
            if not self.instances_dir.exists():
                self.instances_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_yaml("conf/carla.yaml")
        self.files = {}
        self.pose = {}
        self.files["train"] = []
        self.files["validation"] = []
        self.files["test"] = []

        train_path = self.data_dir / "train"
        val_path = self.data_dir / "val"
        test_path = self.data_dir / "test"
        for path,mode in zip([train_path, val_path, test_path],["train", "validation", "test"]):
            dirs_ = os.listdir(path)
            for dir in dirs_:
                filepaths = list(path.glob(f"{dir}/lidar/raw/*npy"))
                self.files[mode].extend(sorted([str(file) for file in filepaths]))

    def preprocess(self):
        for mode in self.modes:
            database = []
            instance_database = {}
            for filepath in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(filepath, mode)
                database.append(filebase)
                if self.generate_instances and mode in ["train", "validation"]:
                    instances = self.extract_instance_from_file(filebase)
                    for instance in instances:
                        unique_identifier = (
                            f"{instance['sequence']}_{instance['panoptic_label']}"
                        )
                        if unique_identifier in instance_database:
                            instance_database[unique_identifier]["filepaths"].append(
                                instance["instance_filepath"]
                            )
                        else:
                            instance_database[unique_identifier] = {
                                "semantic_label": instance["semantic_label"],
                                "filepaths": [instance["instance_filepath"]],
                            }
            save_database(database, mode, self.save_dir)
            if self.generate_instances and mode in ["train", "validation"]:
                save_database(
                    list(instance_database.values()), f"{mode}_instances", self.save_dir
                )
        merge_trainval(self.save_dir, self.generate_instances)

    def process_file(self, filepath, mode):
        if mode == "validation":
            mode = "val"
        sequence, scan = re.search(
            r"/home/nicholas/Desktop/main_UE4/output/"+mode+"/(\d+)/lidar/raw/lidar-(\d+)\.npy$", filepath
        ).group(1, 2)
        filebase = {
            "filepath": filepath,
            "sequence": int(sequence),
            #"pose": self.pose[mode][int(sequence)][int(scan)].tolist(),
        }
        if mode in ["train", "val"]:
            label_filepath = filepath.replace("lidar", "semantic_lidar")
            filebase["label_filepath"] = label_filepath
        return filebase

    #This return the instances inside the file (ONE FILE)
    def extract_instance_from_file(self, filebase):
        file = np.load(filebase["filepath"])
        points = np.frombuffer(file, dtype=np.float32).reshape(-1, 4)
        #pose = np.array(filebase["pose"]).T
        #points[:, :3] = points[:, :3] @ pose[:3, :3] + pose[3, :3]
        file_label = np.load(filebase["label_filepath"])
        label_data_ = np.frombuffer(file_label, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

        if len(points) < len(label_data_):
            print("MISMATCHED POINT AND LABEL LENGTHS", len(points), len(label_data_))
            points_ = points[:, :3]
            lbls_ = np.vstack((label_data_['x'], label_data_['y'], label_data_['z'])).T

            points_set = set(map(tuple, points_))

            # 2. Create a boolean mask: True if the label exists in points
            # This handles duplicates because every instance of the label is checked
            mask = np.array([tuple(lbl) in points_set for lbl in lbls_])

            # 3. Apply the mask to label_data_
            label_data_ = label_data_[mask]
            print(len(points_), len(label_data_))

        label_data = [label_data_['ObjTag'][i] + (label_data_['ObjIdx'][i] << 16) for i in range(len(label_data_))]

        mode = "train" if "train" in filebase["filepath"] else "val" if "val" in filebase["filepath"] else "test"

        sequence, scan = re.search(
            r"/home/nicholas/Desktop/main_UE4/output/"+mode+"/(\d+)/lidar/raw/lidar-(\d+)\.npy$", filebase["filepath"]
        ).group(1, 2)
        file_instances = []

        #print(np.unique(label_data, return_counts=True))

        for panoptic_label in np.unique(label_data):
            semantic_label = panoptic_label & 0xFFFF
            #print(semantic_label, panoptic_label)
            #semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(
            #    semantic_label
            #)
            if np.isin(semantic_label, range(12, 20)):  #12 to 19 are the instances classes (cars, pedestrians, etc)
                instance_mask = label_data == panoptic_label
                instance_points = points[instance_mask, :]
                filename = f"{sequence}_{panoptic_label:010d}_{scan}.npy"
                instance_filepath = self.instances_dir / filename
                instance = {
                    "sequence": sequence,
                    "panoptic_label": f"{panoptic_label:010d}",
                    "instance_filepath": str(instance_filepath),
                    "semantic_label": semantic_label.item(),
                }
                np.save(instance_filepath, instance_points.astype(np.float32))
                file_instances.append(instance)
        return file_instances


if __name__ == "__main__":
    Fire(CarlaPreprocessing)
