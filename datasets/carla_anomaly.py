import numpy as np
import volumentations as V
from loguru import logger
from pathlib import Path
from typing import List, Optional, Union
from random import random, choice, uniform
from torch.utils.data import Dataset
from datasets.utils import load_yaml


class SemanticCARLADataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = "/home/nicholas/preprocessed_anomalies",
        mode: Optional[str] = "train",
        add_distance: Optional[bool] = False,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        instance_population: Optional[int] = 0,
        sweep: Optional[int] = 1,
    ):
        self.mode = mode
        self.data_dir = data_dir
        self.ignore_label = ignore_label
        self.add_distance = add_distance
        self.instance_population = instance_population
        self.sweep = sweep
        self.config = load_yaml("conf/carla_anomaly.yaml")

        # loading database file
        database_path = Path(self.data_dir)
        if not (database_path / f"{mode}_database.yaml").exists():
            logger.error("Database doesn't exist")
            raise FileNotFoundError
        self.data = load_yaml(database_path / f"{mode}_database.yaml")

        self.label_info = self._select_correct_labels(self.config["learning_ignore"])
        # augmentations
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(
                volume_augmentations_path, data_format="yaml"
            )
        # reformulating in sweeps
        data = [[]]
        last_scene = self.data[0]["sequence"]
        #Put all the elements with equal sequence into a list [78,78,...] [4,4,...] ...
        for x in self.data:
            if x["sequence"] == last_scene:
                data[-1].append(x)
            else:
                last_scene = x["sequence"]
                data.append([x])

        for i in range(len(data)): #Here data is a list of lists, so data[i] is a list
            #we are going to use a sweep of 1
            data[i] = list(self.chunks(data[i], sweep))
        self.data = [val for sublist in data for val in sublist]

        if instance_population > 0:
            self.instance_data = load_yaml(
                database_path / f"{mode}_instances_database.yaml"
            )

    def chunks(self, lst, n):
        if "train" in self.mode or n == 1:
            for i in range(len(lst) - n + 1):
                yield lst[i : i + n]
        else:
            for i in range(0, len(lst) - n + 1, n - 1):
                yield lst[i : i + n]
            if i != len(lst) - n:
                yield lst[i + n - 1 :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        coordinates_list = []
        features_list = []
        labels_list = []
        acc_num_points = [0]
        for time, scan in enumerate(self.data[idx]):
            point_file = np.load(scan["filepath"])
            points = np.frombuffer(point_file, dtype=np.float32).reshape(-1, 4)
            coordinates = points[:, :3]
            # rotate and translate
            #pose = np.array(scan["pose"], dtype=np.float32).T
            #coordinates = coordinates @ pose[:3, :3] + pose[3, :3]
            coordinates_list.append(coordinates)
            acc_num_points.append(acc_num_points[-1] + len(coordinates))
            features = points[:, 3:4]

            v_min = np.percentile(features, 1)  # Ignore lowest 1%
            v_max = np.percentile(features, 99)  # Ignore highest 1%
            features = np.clip(features, v_min, v_max)
            features = (features - v_min) / (v_max - v_min + 1e-8)

            time_array = np.ones((features.shape[0], 1), dtype=np.float32) * time
            features = np.hstack((time_array, features))
            features_list.append(features)
            if "test" in self.mode:
                labels = np.zeros_like(features).astype(np.int64)
                labels_list.append(labels)
            else:
                label_file = np.load(scan["label_filepath"])
                panoptic_label = np.frombuffer(label_file, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

                #CARLA semantic lidar, sometimes has 1 point more than the lidar file, so remove it
                if len(points) < len(panoptic_label):
                    print("MISMATCHED POINT AND LABEL LENGTHS", len(points), len(panoptic_label))
                    points_ = points[:, :3]
                    lbls_ = np.vstack((panoptic_label['x'], panoptic_label['y'], panoptic_label['z'])).T

                    points_set = set(map(tuple, points_))

                    # 2. Create a boolean mask: True if the label exists in points
                    # This handles duplicates because every instance of the label is checked
                    mask = np.array([tuple(lbl) in points_set for lbl in lbls_])

                    # 3. Apply the mask to label_data_
                    panoptic_label = panoptic_label[mask]
                    print(len(points_), len(panoptic_label))

                panoptic_label = np.array([panoptic_label['ObjTag'][i] + (panoptic_label['ObjIdx'][i] << 16) for i in
                              range(len(panoptic_label))])


                # ====================================== ANOMALIES =============
                #I have different kind of anomalies (from 30 to 36) idx
                # WE should filter out class 30, which is pothole, in order to see if the model can get it during test time
                if "train" or "validation" in self.mode:
                    labels = panoptic_label & 0xFFFF
                    instances = panoptic_label >> 16
                    #Set to class 0, which will be ignored

                    labels[labels == 30] = 0
                    panoptic_label = (instances << 16) + labels

                labels = panoptic_label & 0xFFFF
                instances = panoptic_label >> 16

                lbl = [l for l in panoptic_label if l & 0xFFFF >= 30]
                num_anomalies = np.unique(lbl)

                # Put all the amoomalies under the same class, in this case class 30 and give them different instance ids
                instance = 0
                for anoamly_class in num_anomalies:
                    mask = panoptic_label == anoamly_class
                    labels[mask] = 30
                    instances[mask] = instance
                    instance += 1

                panoptic_label = (instances << 16) + labels

                lbl = [l for l in panoptic_label if l & 0xFFFF >= 30]
                num_anomalies = np.unique(lbl)

                #=====================================================================

                semantic_label, instance_lbl = self.label_parser(panoptic_label)
                labels = np.hstack((semantic_label[:, None], panoptic_label[:, None]))
                labels_list.append(labels)

        coordinates = np.vstack(coordinates_list)
        features = np.vstack(features_list)
        labels = np.vstack(labels_list)

        if "train" in self.mode and self.instance_population > 0:
            max_instance_id = np.amax(labels[:, 1])
            pc_center = coordinates.mean(axis=0)
            #INSTANCE POPULATION HERE IS A DATA AUGMENTATION TECHNIQUE!
            instance_c, instance_f, instance_l = self.populate_instances(
                max_instance_id, pc_center, self.instance_population
            )
            coordinates = np.vstack((coordinates, instance_c))
            features = np.vstack((features, instance_f))
            labels = np.vstack((labels, instance_l))


        if self.add_distance:
            center_coordinate = coordinates.mean(0)
            features = np.hstack(
                (
                    features,
                    np.linalg.norm(coordinates - center_coordinate, axis=1)[
                        :, np.newaxis
                    ],
                )
            )

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
                )
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

        features = np.hstack((coordinates, features))

        #Maps labels to other values according to label_info

        labels[:, 0] = np.vectorize(self.label_info.__getitem__)(labels[:, 0])
        labels = labels.astype(np.long)

        # In case of anomalies, after the vectorize, the anomaly label will be 29, meanwhile the pothole (for training) will be 255

        return {
            "num_points": acc_num_points,
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "sequence": scan["sequence"],
        }

    def _select_correct_labels(self, learning_ignore):
        count = 0
        label_info = dict()
        for k, v in learning_ignore.items():
            if v:
                label_info[k] = self.ignore_label
            else:
                label_info[k] = count
                count += 1
        return label_info

    def _remap_model_output(self, output):
        inv_map = {v: k for k, v in self.label_info.items()}
        output = np.vectorize(inv_map.__getitem__)(output)
        return output

    def label_parser(self, panoptic_label):
        semantic_label = panoptic_label & 0xFFFF
        #semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(
        #    semantic_label
        #)
        instance_label = panoptic_label >> 16
        return semantic_label, instance_label

    def populate_instances(self, max_instance_id, pc_center, instance_population):
        coordinates_list = []
        features_list = []
        labels_list = []
        for _ in range(instance_population):
            instance_dict = choice(self.instance_data)
            idx = np.random.randint(len(instance_dict["filepaths"]))
            instance_list = []
            for time in range(self.sweep):
                if idx < len(instance_dict["filepaths"]):
                    filepath = instance_dict["filepaths"][idx]
                    instance = np.load(filepath)
                    time_array = (
                        np.ones((instance.shape[0], 1), dtype=np.float32) * time
                    )
                    instance = np.hstack(
                        (instance[:, :3], time_array, instance[:, 3:4])
                    )
                    instance_list.append(instance)
                    idx = idx + 1
            instances = np.vstack(instance_list)
            coordinates = instances[:, :3] - instances[:, :3].mean(0)
            coordinates += pc_center + np.array(
                [uniform(-10, 10), uniform(-10, 10), uniform(-1, 1)]
            )
            features = instances[:, 3:]
            semantic_label = instance_dict["semantic_label"]
            labels = np.zeros_like(features, dtype=np.int64)
            labels[:, 0] = semantic_label
            max_instance_id = max_instance_id + 1
            labels[:, 1] = max_instance_id
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]
            coordinates_list.append(coordinates)
            features_list.append(features)
            labels_list.append(labels)
        return (
            np.vstack(coordinates_list),
            np.vstack(features_list),
            np.vstack(labels_list),
        )
