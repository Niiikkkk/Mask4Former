import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import yaml


def load_points_xyz(points_npy: Path) -> np.ndarray:
    lidar_data = np.load(points_npy)
    try:
        points = np.frombuffer(lidar_data, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]
    except Exception:
        arr = np.asarray(lidar_data)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        raise ValueError(f"Unsupported point format in {points_npy}")


def load_scores(scores_txt: Path) -> np.ndarray:
    scores = np.loadtxt(scores_txt).astype(np.float32)
    if np.asarray(scores).ndim == 0:
        scores = np.array([float(scores)], dtype=np.float32)
    return scores


def load_labels(label_file: Path, label_format: str) -> np.ndarray:
    if label_format == "carla_semantic_npy":
        raw = np.load(label_file)
        label_data = np.frombuffer(
            raw,
            dtype=np.dtype(
                [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("intensity", np.float32),
                    ("insta", np.uint32),
                    ("sem", np.uint32),
                ]
            ),
        )
        return np.asarray(label_data["sem"], dtype=np.int32)

    if label_format == "carla_panoptic_npy":
        raw = np.load(label_file)
        label_data = np.frombuffer(
            raw,
            dtype=np.dtype(
                [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("CosAngle", np.float32),
                    ("ObjIdx", np.uint32),
                    ("ObjTag", np.uint32),
                ]
            ),
        )
        return np.asarray(label_data["ObjTag"], dtype=np.int32)

    # semantic_kitti_label
    lbl = np.fromfile(label_file, dtype=np.uint32)
    return (lbl & 0xFFFF).astype(np.int32)


def load_color_map(config_yaml: Path) -> Dict[int, List[int]]:
    cfg = yaml.safe_load(config_yaml.read_text())
    color_map = cfg.get("color_map", {})
    return {int(k): [int(v[0]), int(v[1]), int(v[2])] for k, v in color_map.items()}


def labels_to_colors(labels: np.ndarray, color_map_bgr: Dict[int, List[int]]) -> np.ndarray:
	rgb = np.zeros((labels.shape[0], 3), dtype=np.float32)
	default = np.array([0.5, 0.5, 0.5], dtype=np.float32)
	labels_color = np.array([
		(0, 0, 0),  # unlabeled = 0
		# cityscape
		(128, 64, 128),  # road = 1
		(244, 35, 232),  # sidewalk = 2
		(70, 70, 70),  # bilding = 3
		(102, 102, 156),  # wall = 4
		(190, 153, 153),  # fence = 5
		(153, 153, 153),  # pole = 6
		(250, 170, 30),  # trafficlight = 7
		(220, 220, 0),  # trafficsign = 8
		(107, 142, 35),  # vegetation = 9
		(152, 251, 152),  # terrain = 10
		(70, 130, 180),  # sky = 11
		(220, 20, 60),  # pedestrian = 12
		(255, 0, 0),  # rider = 13
		(0, 0, 142),  # Car = 14
		(0, 0, 70),  # trck = 15
		(0, 60, 100),  # bs = 16
		(0, 80, 100),  # train = 17
		(0, 0, 230),  # motorcycle = 18
		(119, 11, 32),  # bicycle = 19
		# cstom
		(110, 190, 160),  # static = 20
		(170, 120, 50),  # dynamic = 21
		(55, 90, 80),  # other = 22
		(45, 60, 150),  # water = 23
		(157, 234, 50),  # roadline = 24
		(81, 0, 81),  # grond = 25
		(150, 100, 100),  # bridge = 26
		(230, 150, 140),  # railtrack = 27
		(180, 165, 180),  # gardrail = 28
		(180, 130, 70),  # rock = 29
		# anomalies
		(193, 71, 71),  # Static_Anomaly = 30
		(102, 102, 255),  # Dynamic_Anomaly = 31
		(175, 83, 83),  # Animal = 32
		(232, 188, 188),  # Tinyanomaly = 33
		(229, 137, 137),  # Smallanomaly = 34
		(189, 47, 47),  # Mediumanomaly = 35
		(131, 7, 7),  # Largeanomaly = 36
	]) / 255.0

	return labels_color[labels]


def normalize_scores(scores: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(scores, p_low), np.percentile(scores, p_high)
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float32)
    return np.clip((scores - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def colorize(values_01: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    from matplotlib import cm

    rgba = cm.get_cmap(cmap)(values_01)
    return rgba[:, :3].astype(np.float32)


def compute_ood_metrics(
    scores: np.ndarray,
    semantic_labels: np.ndarray,
    anomaly_label_start: int = 30,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC and FPR@95TPR for anomaly detection.

    Every semantic class id >= anomaly_label_start is treated as anomalous.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    scores = scores[semantic_labels!=30]
    semantic_labels = semantic_labels[semantic_labels!=30]

    y_true = (semantic_labels >= anomaly_label_start).astype(np.uint8)
    y_score = scores.astype(np.float64)

    if y_true.size != y_score.size:
        raise ValueError(f"Metric input mismatch: labels={y_true.size} scores={y_score.size}")

    unique = np.unique(y_true)
    if unique.size < 2:
        raise ValueError("Need both normal and anomaly samples to compute AUROC/AUPRC/FPR")

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(tpr >= 0.95)[0]
    if valid.size == 0:
        fpr95 = 1.0
    else:
        fpr95 = float(np.min(fpr[valid]))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr95": fpr95,
        "num_points": float(y_true.size),
        "num_anomaly": float(np.sum(y_true)),
        "num_normal": float(y_true.size - np.sum(y_true)),
    }


def show_open3d(
    points_xyz: np.ndarray,
    base_colors: np.ndarray,
    point_size: float,
    gt_colors: Optional[np.ndarray] = None,
    pred_colors: Optional[np.ndarray] = None,
    side_by_side: bool = False,
) -> None:
    import open3d as o3d

    geometries = []

    def make_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    if side_by_side:
        extent_x = float(points_xyz[:, 0].max() - points_xyz[:, 0].min())
        offset = max(15.0, extent_x + 5.0)
        geometries.append(make_pcd(points_xyz - np.array([offset, 0.0, 0.0], dtype=np.float32), base_colors))
        if gt_colors is not None:
            geometries.append(make_pcd(points_xyz, gt_colors))
        if pred_colors is not None:
            geometries.append(make_pcd(points_xyz + np.array([offset, 0.0, 0.0], dtype=np.float32), pred_colors))
    else:
        geometries.append(make_pcd(points_xyz, base_colors))
        if gt_colors is not None:
            geometries.append(make_pcd(points_xyz, gt_colors))
        if pred_colors is not None:
            geometries.append(make_pcd(points_xyz, pred_colors))

    supports_key_callbacks = hasattr(o3d.visualization, "VisualizerWithKeyCallback")
    if supports_key_callbacks:
        vis = o3d.visualization.VisualizerWithKeyCallback()
    else:
        vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Anomaly / GT / Pred", width=1600, height=900)
    for geo in geometries:
        vis.add_geometry(geo)

    can_register = hasattr(vis, "register_key_callback")
    if not side_by_side and len(geometries) > 1:
        # In overlay mode, keep only anomaly visible initially.
        for geo in geometries[1:]:
            vis.remove_geometry(geo, reset_bounding_box=False)
        if can_register:
            print("Overlay mode: press 1=anomaly, 2=gt, 3=pred")
        else:
            print("Overlay mode on this Open3D build has no key callbacks; showing anomaly layer only.")

        def set_single(index: int) -> bool:
            for geo in geometries:
                vis.remove_geometry(geo, reset_bounding_box=False)
            vis.add_geometry(geometries[index], reset_bounding_box=False)
            vis.update_renderer()
            return False

        if can_register and len(geometries) >= 1:
            vis.register_key_callback(ord("1"), lambda _v: set_single(0))
        if can_register and len(geometries) >= 2:
            vis.register_key_callback(ord("2"), lambda _v: set_single(1))
        if can_register and len(geometries) >= 3:
            vis.register_key_callback(ord("3"), lambda _v: set_single(2))

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    vis.run()
    vis.destroy_window()


def show_matplotlib(points_xyz: np.ndarray, colors_rgb: np.ndarray, marker_size: float, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c=colors_rgb, s=marker_size)
    ax.set_title(title)
    plt.show()


def save_ply(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
    rgb_u8 = np.clip(colors_rgb * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points_xyz, rgb_u8):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize anomaly scores with GT/pred labels on LiDAR points")
    parser.add_argument("--points-npy", type=Path, help="Path to lidar-XXXX.npy")
    parser.add_argument("--scores-txt", type=Path, help="Path to exported scores .txt")
    parser.add_argument("--gt-labels", type=Path, default=None, help="Optional ground-truth labels file")
    parser.add_argument("--pred-labels", type=Path, default=None, help="Optional predicted labels file")
    parser.add_argument(
        "--label-format",
        choices=["carla_semantic_npy", "carla_panoptic_npy", "semantic_kitti_label"],
        default="carla_semantic_npy",
        help="Format used by --gt-labels and --pred-labels",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("conf/carla_anomaly.yaml"),
        help="YAML with color_map (BGR)",
    )
    parser.add_argument("--p-low", type=float, default=2.0, help="Lower percentile for score normalization")
    parser.add_argument("--p-high", type=float, default=98.0, help="Upper percentile for score normalization")
    parser.add_argument("--cmap", type=str, default="hot", help="Matplotlib colormap name for anomaly scores")
    parser.add_argument("--viewer", choices=["open3d", "matplotlib"], default="open3d")
    parser.add_argument("--matplot-layer", choices=["anomaly", "gt", "pred"], default="anomaly")
    parser.add_argument("--side-by-side", action="store_true", help="Show anomaly/gt/pred side by side in Open3D")
    parser.add_argument("--point-size", type=float, default=1.5, help="Open3D point size")
    parser.add_argument("--marker-size", type=float, default=0.5, help="Matplotlib marker size")
    parser.add_argument("--save-ply", type=Path, default=None, help="Optional output PLY for anomaly colors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = "/home/nicholas/Desktop/main_UE4/"
    args.scores_txt = "saved/2026-03-24_093551/msp/47/output_test_Sunny_47_lidar_raw_lidar-521851.npy.txt"
    path=args.scores_txt.split("/")[-1].split(".txt")[0].split("_")
    point_path = base_path + "/".join(path)
    args.points_npy = point_path
    args.gt_labels = args.points_npy.replace("lidar","semantic_lidar")

    if args.p_high <= args.p_low:
        raise ValueError("--p-high must be greater than --p-low")

    points = load_points_xyz(args.points_npy)
    scores = load_scores(args.scores_txt)

    if len(points) != len(scores):
        raise ValueError(f"points={len(points)} scores={len(scores)} mismatch")

    scores_01 = normalize_scores(scores, args.p_low, args.p_high)
    anomaly_colors = colorize(scores_01, matplotlib.colormaps["spring"].reversed())

    gt_colors = None
    pred_colors = None
    if args.gt_labels is not None or args.pred_labels is not None:
        color_map = load_color_map(args.dataset_config)

        if args.gt_labels is not None:
            gt_labels = load_labels(args.gt_labels, args.label_format)
            if len(gt_labels) != len(points):
                raise ValueError(f"gt labels length {len(gt_labels)} != points length {len(points)}")
            gt_colors = labels_to_colors(gt_labels, color_map)

        if args.pred_labels is not None:
            pred_labels = load_labels(args.pred_labels, args.label_format)
            if len(pred_labels) != len(points):
                raise ValueError(f"pred labels length {len(pred_labels)} != points length {len(points)}")
            pred_colors = labels_to_colors(pred_labels, color_map)


    if gt_labels is None:
        raise ValueError("--compute-metrics requires --gt-labels")
    print(np.unique(gt_labels, return_counts=True))
    metrics = compute_ood_metrics(scores, gt_labels, anomaly_label_start=31)
    print("OOD metrics (labels >= {} are anomalies):".format(31))
    print("AUROC : {:.6f}".format(metrics["auroc"]))
    print("AUPRC : {:.6f}".format(metrics["auprc"]))
    print("FPR95 : {:.6f}".format(metrics["fpr95"]))
    print(
        "Counts: total={:.0f}, anomaly={:.0f}, normal={:.0f}".format(
            metrics["num_points"], metrics["num_anomaly"], metrics["num_normal"]
        )
    )

    if args.save_ply is not None:
        save_ply(args.save_ply, points, anomaly_colors)
        print(f"Saved anomaly PLY: {args.save_ply}")

    if args.viewer == "open3d":
        show_open3d(
            points_xyz=points,
            base_colors=anomaly_colors,
            point_size=args.point_size,
            gt_colors=gt_colors,
            pred_colors=pred_colors,
            side_by_side=args.side_by_side,
        )
        return

    # matplotlib fallback shows one selected layer.
    layer_colors = {
        "anomaly": anomaly_colors,
        "gt": gt_colors,
        "pred": pred_colors,
    }
    colors = layer_colors[args.matplot_layer]
    if colors is None:
        raise ValueError(f"Requested matplotlib layer '{args.matplot_layer}' but labels were not provided")
    show_matplotlib(points, colors, args.marker_size, f"Layer: {args.matplot_layer}")


if __name__ == "__main__":
    main()

