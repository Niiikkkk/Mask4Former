import statistics
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import faulthandler
import os
import signal
import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from sklearn.cluster import DBSCAN

from utils.utils import associate_instances, generate_logs
from pytorch_lightning.utilities import rank_zero_only


faulthandler.enable()


_FAULTHANDLER_FILES = []


def _register_faulthandler_signal(sig: int) -> None:
    """Register faulthandler for a signal, writing to a per-process /tmp file.

    Keeping the file handle referenced prevents it from being GC-closed.
    """
    path = f"/tmp/pq_faulthandler_{os.getpid()}.log"
    try:
        f = open(path, "a", buffering=1)
        f.write(f"\n===== faulthandler registered (pid={os.getpid()}, sig={sig}) =====\n")
        faulthandler.register(sig, file=f, all_threads=True, chain=True)
        _FAULTHANDLER_FILES.append(f)
    except Exception:
        # Best-effort only; fall back to stderr handler below.
        return


def _dump_traceback_to_tmp(signum, frame):
    path = f"/tmp/pq_faulthandler_{os.getpid()}.log"
    try:
        with open(path, "a", buffering=1) as f:
            sig_name = getattr(signal, "Signals", None)
            if sig_name is not None:
                try:
                    sig_name = signal.Signals(signum).name
                except Exception:
                    sig_name = str(signum)
            else:
                sig_name = str(signum)
            f.write(
                f"\n===== {sig_name} received (pid={os.getpid()}, t={time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n"
            )
            faulthandler.dump_traceback(file=f, all_threads=True)
    except Exception:
        # Fall back to stderr if /tmp is not writable for any reason.
        faulthandler.dump_traceback(all_threads=True)


def _install_traceback_signal_handlers() -> None:
    """Install signal handlers to dump Python stack traces.

    Notes:
      - PyTorch Lightning / launchers may install their own SIGUSR1 handler.
      - SIGUSR2 is typically unused and is more reliable for ad-hoc debugging.
      - We chain any previous handler to avoid breaking other tooling.
    """

    def _chain(previous_handler, signum, frame):
        try:
            _dump_traceback_to_tmp(signum, frame)
        finally:
            if callable(previous_handler):
                try:
                    previous_handler(signum, frame)
                except Exception:
                    pass

    for sig in (signal.SIGUSR2, signal.SIGUSR1):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, lambda s, f, _prev=prev: _chain(_prev, s, f))
        except Exception:
            # Can fail if not in main thread, or on platforms without the signal.
            pass


_register_faulthandler_signal(signal.SIGUSR2)
_register_faulthandler_signal(signal.SIGUSR1)
_install_traceback_signal_handlers()


class PanopticSegmentationAnomalies(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        
        # Track freezing state
        self.freeze_first_epoch = False
        self.freeze_backbone = False  # NEW: Control backbone freezing
        self.current_epoch_num = 0
        self.defreezing_epoch = 1  # Epoch to unfreeze entire model

        self.val_min_eval_distance = 2.5
        self.val_max_eval_distance = 50.0
        self.val_min_anomaly_points = 5

        # Validation: evaluate a fixed number of random samples then stop
        self.val_num_samples = config.general.get('val_num_samples', 800)
        self.val_sample_seed = config.general.get('val_sample_seed', 0)
        self._val_sampler_generator = torch.Generator()
        self._val_sampler_generator.manual_seed(int(self.val_sample_seed))

        # Validation dataloader stability knobs (useful when workers occasionally hang)
        self.val_num_workers = int(config.general.get('val_num_workers', getattr(config.data, 'num_workers', 0)))
        self.val_timeout_s = int(config.general.get('val_dataloader_timeout_s', 300))
        self.val_persistent_workers = bool(config.general.get('val_persistent_workers', False))
        
        # Clustering aggregation control
        self.save_clustered_scores = config.general.get('save_clustered_scores', False)
    
        # Define which parts to keep unfrozen during first epoch
        self.unfrozen_parts = [
            'class_embed_head',  # Classification head
            'anomaly_embed_head',  # Anomaly head
            #'mask_embed_head',   # Mask embedding head
            # Add other specific parts you want to train in first epoch
        ]
        
        # Define backbone parts to freeze
        self.backbone_parts = [
            'backbone',  # Main backbone network
            'input_conv',  # Input convolution
            # Add other backbone-related parts
        ]
        
        self.optional_freeze = nullcontext

        self.training_step_outputs = []
        self.validation_step_outputs = []

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
            "loss_box": matcher.cost_box,
            "loss_anomaly": matcher.cost_anomaly,
            "loss_contrastive": matcher.cost_contrastive, #15 for bce, 30 for margin 0.94, 15 for entropy, 3 for energy
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )
        
        # Select anomaly score used in validation (mirrors criterion.anomaly_score_name)
        self.validation_anomaly_score_name = config.general.get(
            'anomaly_score_name', getattr(self.criterion, 'anomaly_score_name', 'msp')
        ).lower()
        print(f"Using '{self.validation_anomaly_score_name}' as validation anomaly score.")
        
        # metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)

        # ADD: Anomaly evaluator
        from models.metrics.anomaly_eval import AnomalyEval
        anomaly_eval_max_points = config.general.get('anomaly_eval_max_points', 1_000_000_000)
        anomaly_eval_seed = config.general.get('anomaly_eval_seed', 0)
        self.anomaly_evaluator = AnomalyEval(
            anomaly_class_id=30,
            threshold=0.5,
            max_points=anomaly_eval_max_points,
            seed=anomaly_eval_seed,
        )
        self.last_seq = None
    
        # Apply backbone freezing if enabled
        if self.freeze_backbone:
            self._freeze_backbone()


    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        print("=" * 60)
        print("FREEZING BACKBONE")
        print("=" * 60)
        
        frozen_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            total_params += sum(p.numel() for p in module.parameters())
            
            # Check if this module is part of the backbone
            is_backbone = any(part in name for part in self.backbone_parts)
            
            if is_backbone:
                for param in module.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                print(f"✓ Frozen: {name}")
        
        print(f"Frozen {frozen_params}/{total_params} parameters ({100*frozen_params/total_params:.2f}%)")
        print("=" * 60 + "\n")

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        print("=" * 60)
        print("UNFREEZING BACKBONE")
        print("=" * 60)
        
        unfrozen_params = 0
        
        for name, module in self.model.named_modules():
            # Check if this module is part of the backbone
            is_backbone = any(part in name for part in self.backbone_parts)
            
            if is_backbone:
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                print(f"✓ Unfrozen: {name}")
        
        print(f"Unfrozen {unfrozen_params} backbone parameters")
        print("=" * 60 + "\n")

    def forward(self, x, raw_coordinates=None, is_eval=False, inverse_maps=None):
        with self.optional_freeze():
            x = self.model(x, raw_coordinates=raw_coordinates, is_eval=is_eval, inverse_maps=inverse_maps)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch

        # # Check if there are anomalies (class 20) in the target
        has_anomalies = False
        anomaly_count = 0
        total_points = 0

        for _, batch_target in enumerate(target):
            if 'labels' in batch_target:
                labels = batch_target['labels']
                # Check if class 29 (anomaly) exists in labels
                if torch.any(labels == 29):
                    has_anomalies = True
                    anomaly_count += torch.sum(labels == 29).item()
                total_points += labels.numel()
                #print(f"Labels in batch target: {labels.unique().cpu().numpy()}")

        if batch_idx == 0 and self.trainer.is_global_zero:
            if has_anomalies:
                print(f"Train batch contains {anomaly_count}/{total_points} anomaly masks (class 29)")
            else:
                print(f"Train batch contains no anomalies (0/{total_points} masks)")

        raw_coordinates = data.raw_coordinates
        inverse_maps = data.inverse_maps

        output = self.model(
            data.coordinates, data.features, raw_coordinates, self.device
        )

        losses = self.criterion(output, target, raw_coordinates=raw_coordinates)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        if "loss_contrastive" in losses.keys():
            print(f"Contrastive Loss present: {losses['loss_contrastive'].item():.4f}")
        else:
            print("Contrastive Loss not present in this step.")

        print(
            f"loss_ce : {losses.get('loss_ce', torch.tensor(0.0)).item()} | "
            f"loss_mask : {losses.get('loss_mask', torch.tensor(0.0)).item()} | "
            f"loss_dice : {losses.get('loss_dice', torch.tensor(0.0)).item()} | "
            f"loss_box : {losses.get('loss_box', torch.tensor(0.0)).item()} | "
        )

        logs = {f"train_{k}": v.detach().cpu().item() for k, v in losses.items()}

        # Safely compute mean losses only if they exist
        loss_ce_values = [v for k, v in logs.items() if "loss_ce" in k]
        if loss_ce_values:
            logs["train_mean_loss_ce"] = statistics.mean(loss_ce_values)

        loss_mask_values = [v for k, v in logs.items() if "loss_mask" in k]
        if loss_mask_values:
            logs["train_mean_loss_mask"] = statistics.mean(loss_mask_values)

        loss_dice_values = [v for k, v in logs.items() if "loss_dice" in k]
        if loss_dice_values:
            logs["train_mean_loss_dice"] = statistics.mean(loss_dice_values)

        loss_box_values = [v for k, v in logs.items() if "loss_box" in k]
        if loss_box_values:
            logs["train_mean_loss_box"] = statistics.mean(loss_box_values)
        
        # Add anomaly loss mean if present
        loss_anomaly_values = [v for k, v in logs.items() if "loss_anomaly" in k]
        if loss_anomaly_values:
            logs["train_mean_loss_anomaly"] = statistics.mean(loss_anomaly_values)

        loss = sum(losses.values())
        self.training_step_outputs.append(loss.cpu().item())

        self.log_dict(logs)
        return loss

    def test_step(self, batch, batch_idx):
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    
        # ========== CONFIGURE WHICH SCORES TO COMPUTE ==========
        # Available scores (comment/uncomment to enable/disable):
        ENABLED_SCORES = [
            'msp',           # Maximum Softmax Probability (inverted)
            # 'max_logit',   # Maximum logit (pre-softmax, inverted)
            # 'entropy',     # Entropy score (uncertainty)
            # 'energy',      # Energy score (logsumexp)
            # 'rba',         # RBA score (tanh-based)
        ]
        # =======================================================

        data, target = batch
        inverse_maps = data.inverse_maps
        raw_coordinates = data.raw_coordinates
        sequences = data.sequences

        output = self.model(
            data.coordinates, data.features, raw_coordinates, self.device, is_eval=True
        )

        if self.config.inference.output_mode == "confidence":

            def get_maxlogit(logit, mask):
                """Max logit using RAW logits (pre-softmax) - all 19 semantic classes"""
                logit_19 = logit[:, :30]  # Only first 19 semantic classes
                confid = mask.float().sigmoid().matmul(logit_19)
                max_logit = torch.max(confid, dim=1).values
                max_logit = (max_logit * -1) + 1
                return max_logit
            
            def get_msp_score(logit, mask):
                """Maximum Softmax Probability (MSP) - consistent with validation"""
                # Apply softmax to get probabilities
                logit_probs = torch.functional.F.softmax(logit, dim=-1)  # [Q, 20]
                logit_19 = logit_probs[:, :30]  # Only first 19 semantic classes
                confid = mask.float().sigmoid().matmul(logit_19)  # [num_points, 19]
                max_prob = torch.max(confid, dim=1).values  # Max probability per point
                anomaly_score = (max_prob * -1) + 1  # Invert: high prob → low anomaly score
                return anomaly_score

            # NEW: Entropy (higher => more uncertain => more anomalous)
            def get_entropy(logit, mask):
                probs_q = torch.functional.F.softmax(logit, dim=-1)[:, :30]  # [Q,19]
                probs_point = mask.float().sigmoid().matmul(probs_q)         # [N,19]
                probs_point = torch.clamp(probs_point, min=1e-8)
                entropy = -(probs_point * probs_point.log()).sum(dim=1)      # [N]
                return entropy

            # NEW: Energy score (higher => more anomalous)
            def get_energy(logit, mask):
                semantic_logits_q = logit[:, :30]                            # [Q,19]
                point_logits = mask.float().sigmoid().matmul(semantic_logits_q)  # [N,19]
                energy = -torch.logsumexp(point_logits, dim=1)               # [N]
                # Optionally invert so higher = more anomalous; keep as-is (less negative => higher)
                return energy
            
            def get_rba(logit, mask):
                """RBA score using only semantic classes"""
                logit_semantic = logit[:, :30]  # [Q, 19]
                confid = mask.float().sigmoid().matmul(logit_semantic)
                rba = -confid.tanh().sum(dim=1)
                rba[rba < -1] = -1
                rba = rba + 1
                return rba
            
            def connected_components_clustering(coords, radius=0.5):
                """Ultra-fast clustering using connected components on adjacency graph."""
                tree = cKDTree(coords)
                pairs = tree.query_pairs(radius, output_type='ndarray')
                
                if len(pairs) == 0:
                    return np.arange(len(coords)), len(coords)
                
                n_points = len(coords)
                row = np.concatenate([pairs[:, 0], pairs[:, 1]])
                col = np.concatenate([pairs[:, 1], pairs[:, 0]])
                data_vals = np.ones(len(row), dtype=bool)
                
                adjacency = csr_matrix((data_vals, (row, col)), shape=(n_points, n_points))
                num_instances, labels = connected_components(adjacency, directed=False)
                
                return labels, num_instances
            
            def radius_based_clustering(coords, radius=0.5):
                """Fast radius-based clustering using KD-Tree."""
                tree = cKDTree(coords)
                labels = np.full(len(coords), -1, dtype=np.int32)
                current_label = 0
                
                for i in range(len(coords)):
                    if labels[i] != -1:
                        continue
                    
                    neighbors = tree.query_ball_point(coords[i], radius)
                    
                    if len(neighbors) > 0:
                        for n in neighbors:
                            if labels[n] == -1:
                                labels[n] = current_label
                        current_label += 1
                
                num_instances = current_label
                return labels, num_instances
            
            def aggregate_score_with_instances(scores, instance_labels):
                """Aggregate scores using pre-computed instance labels."""
                aggregated_scores = np.zeros_like(scores)
                unique_labels = np.unique(instance_labels)
                
                for label in unique_labels:
                    if label == -1:
                        mask = instance_labels == label
                        aggregated_scores[mask] = scores[mask]
                    else:
                        mask = instance_labels == label
                        mean_score = scores[mask].mean()
                        aggregated_scores[mask] = mean_score
                
                return aggregated_scores

            pred_logits = output["pred_logits"]  # [B, Q, 20]
            pred_masks = output["pred_masks"]    # [B, num_points, Q]
                    
            # Define save paths for enabled scores only
            save_paths = {}
            save_paths_instance = {}
            
            if 'rba' in ENABLED_SCORES:
                save_paths['rba'] = Path(self.config.general.save_dir) / "prediction"
                if self.save_clustered_scores:
                    save_paths_instance['rba'] = Path(self.config.general.save_dir) / "rba_instance"
            
            if 'max_logit' in ENABLED_SCORES:
                save_paths['max_logit'] = Path(self.config.general.save_dir) / "max_logit"
                if self.save_clustered_scores:
                    save_paths_instance['max_logit'] = Path(self.config.general.save_dir) / "max_logit_instance"
            
            if 'msp' in ENABLED_SCORES:
                save_paths['msp'] = Path(self.config.general.save_dir) / "msp"
                if self.save_clustered_scores:
                    save_paths_instance['msp'] = Path(self.config.general.save_dir) / "msp_instance"
            
            if 'entropy' in ENABLED_SCORES:
                save_paths['entropy'] = Path(self.config.general.save_dir) / "entropy"
                if self.save_clustered_scores:
                    save_paths_instance['entropy'] = Path(self.config.general.save_dir) / "entropy_instance"
            
            if 'energy' in ENABLED_SCORES:
                save_paths['energy'] = Path(self.config.general.save_dir) / "energy"
                if self.save_clustered_scores:
                    save_paths_instance['energy'] = Path(self.config.general.save_dir) / "energy_instance"
            
            # Print configuration
            if self.save_clustered_scores:
                cluster_radius = self.config.general.get('cluster_radius', 1.0)
                cluster_method = self.config.general.get('cluster_method', 'radius')
                print(f"\n{'='*60}")
                print(f"CLUSTERING ENABLED: method={cluster_method}, radius={cluster_radius}m")
                print(f"ENABLED SCORES: {', '.join(ENABLED_SCORES)}")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"CLUSTERING DISABLED: Only saving point-level scores")
                print(f"ENABLED SCORES: {', '.join(ENABLED_SCORES)}")
                print(f"{'='*60}\n")
        
            offset_coords_idx = 0
            
            for b_idx in range(len(pred_logits)):
                verbose = b_idx == 0  # only print for first batch element
                # Get coordinates (needed for clustering if enabled)
                curr_coords_idx = pred_masks[b_idx].shape[0]
                curr_coords = raw_coordinates[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords_mapped = curr_coords[inverse_maps[b_idx]].detach().cpu().numpy()
                offset_coords_idx += curr_coords_idx
                
                if verbose:
                    print(f"\nProcessing sample {sequences[b_idx][0]}/{sequences[b_idx][1]}:")
                    print(f"  Total points: {len(curr_coords_mapped)}")
                
                # Perform clustering only if enabled
                if self.save_clustered_scores:
                    import time
                    start_time = time.time()
                    
                    if cluster_method == 'connected_components':
                        instance_labels, num_instances = connected_components_clustering(
                            curr_coords_mapped, radius=cluster_radius
                        )
                    elif cluster_method == 'radius':
                        instance_labels, num_instances = radius_based_clustering(
                            curr_coords_mapped, radius=cluster_radius
                        )
                    else:
                        raise ValueError(f"Unknown cluster_method: {cluster_method}")
                    
                    clustering_time = time.time() - start_time
                    
                    unique_labels, counts = np.unique(instance_labels, return_counts=True)
                    mean_points_per_instance = counts.mean()
                    max_points_per_instance = counts.max()
                    min_points_per_instance = counts.min()
                    
                    if verbose:
                        print(f"  ✓ Clustering time: {clustering_time:.3f}s")
                        print(f"  ✓ Instances found: {num_instances}")
                        print(f"  ✓ Points per instance (mean/min/max): {mean_points_per_instance:.1f} / {min_points_per_instance} / {max_points_per_instance}")
                
                # Compute only enabled scores
                computed_scores = {}
                computed_scores_instance = {}
                
                if 'rba' in ENABLED_SCORES:
                    computed_scores['rba'] = get_rba(pred_logits[b_idx], pred_masks[b_idx])
                    computed_scores['rba'] = computed_scores['rba'][inverse_maps[b_idx]].detach().cpu().numpy()
                
                if 'max_logit' in ENABLED_SCORES:
                    computed_scores['max_logit'] = get_maxlogit(pred_logits[b_idx], pred_masks[b_idx])
                    computed_scores['max_logit'] = computed_scores['max_logit'][inverse_maps[b_idx]].detach().cpu().numpy()
                
                if 'msp' in ENABLED_SCORES:
                    computed_scores['msp'] = get_msp_score(pred_logits[b_idx], pred_masks[b_idx])
                    computed_scores['msp'] = computed_scores['msp'][inverse_maps[b_idx]].detach().cpu().numpy()
                
                if 'entropy' in ENABLED_SCORES:
                    computed_scores['entropy'] = get_entropy(pred_logits[b_idx], pred_masks[b_idx])
                    computed_scores['entropy'] = computed_scores['entropy'][inverse_maps[b_idx]].detach().cpu().numpy()
                
                if 'energy' in ENABLED_SCORES:
                    computed_scores['energy'] = get_energy(pred_logits[b_idx], pred_masks[b_idx])
                    computed_scores['energy'] = computed_scores['energy'][inverse_maps[b_idx]].detach().cpu().numpy()
                
                # Aggregate scores only if clustering is enabled
                if self.save_clustered_scores:
                    if verbose:
                        print(f"  Aggregating scores...")
                    for score_name, score_array in computed_scores.items():
                        computed_scores_instance[score_name] = aggregate_score_with_instances(
                            score_array, instance_labels
                        )
                
                # Save point-level scores
                for score_name, score_array in computed_scores.items():
                    score_base_path = save_paths[score_name] / f"{sequences[b_idx][0]}"
                    score_file_path = score_base_path / f"{sequences[b_idx][1]}.txt"
                    
                    if not score_base_path.exists():
                        score_base_path.mkdir(exist_ok=True, parents=True)
                    
                    np.savetxt(score_file_path, score_array)
                
                # Save instance-aggregated scores (conditionally)
                if self.save_clustered_scores:
                    for score_name, score_array in computed_scores_instance.items():
                        score_instance_base_path = save_paths_instance[score_name] / f"{sequences[b_idx][0]}"
                        score_instance_file_path = score_instance_base_path / f"{sequences[b_idx][1]}.txt"
                        
                        if not score_instance_base_path.exists():
                            score_instance_base_path.mkdir(exist_ok=True, parents=True)
                        
                        np.savetxt(score_instance_file_path, score_array)
                    
                    if verbose:
                        print(f"  ✓ Saved point-level and instance-level scores: {list(computed_scores.keys())}")
                else:
                    if verbose:
                        print(f"  ✓ Saved point-level scores only: {list(computed_scores.keys())}")
        
            print(f"\n{'='*60}")
            print(f"TEST STEP COMPLETE")
            print(f"{'='*60}\n")
            
            return {}
    
        elif self.config.inference.output_mode == "labels":
            raise NotImplementedError(
                "Output mode 'labels' is not implemented yet. Please use 'confidence' or 'both'."
            )

        else:
            raise NotImplementedError(
                "Available output mode: 'confidence', 'labels', or 'both'. This output mode is not implemented: {}".format(
                    self.config.inference.output_mode
                )
            )

    def validation_step(self, batch, batch_idx):
        data, target = batch

        # Check if there are anomalies (class 29) in the target
        has_anomalies = False
        anomaly_count = 0
        total_points = 0

        for _, batch_target in enumerate(target):
            if 'labels' in batch_target:
                labels = batch_target['labels']
                # Check if class 19 (anomaly) exists in labels
                print(f"Labels in val batch target: {labels.unique().cpu().numpy()}")
                if torch.any(labels == 29):
                    has_anomalies = True
                    anomaly_count += torch.sum(labels == 29).item()
                total_points += labels.numel()
        
        print(f"Validation batch contains {anomaly_count}/{total_points} anomaly masks (class 29)")

        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        raw_coordinates = data.raw_coordinates
        num_points = data.num_points
        sequences = data.sequences

        output = self.model(
            data.coordinates, data.features, raw_coordinates, self.device, is_eval=True
        )
        
        losses = self.criterion(output, target, raw_coordinates=raw_coordinates)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        # Get raw logits & mask predictions (no pre-softmax aggregation here)
        pred_logits_raw = output["pred_logits"]  # [B, Q, 20]
        pred_masks = output["pred_masks"]       # [B, N, Q]

        def _compute_point_anomaly_score(logits_q: torch.Tensor, mask_point_q: torch.Tensor, mode: str) -> torch.Tensor:
            """Compute per-point anomaly score with selectable mode.
            logits_q: [Q, C] including no-object last; mask_point_q: [N,Q] or [Q,N].
            Modes: 'msp','entropy','energy','max_logit'. Returns [N].
            """
            # Ensure mask is [N,Q]
            if mask_point_q.shape[0] == logits_q.shape[0] and mask_point_q.shape[1] != logits_q.shape[0]:
                mask_point_q = mask_point_q.transpose(0, 1).contiguous()
            elif mask_point_q.shape[1] == logits_q.shape[0] and mask_point_q.shape[0] != logits_q.shape[0]:
                pass
            mask_probs = mask_point_q.float().sigmoid()  # [N,Q]
            semantic_logits_q = logits_q[:, :-1]
            probs_q = F.softmax(logits_q, dim=-1)[:, :-1]
            probs_point = mask_probs.matmul(probs_q)            # [N,19]
            point_logits = mask_probs.matmul(semantic_logits_q) # [N,19]
            m = mode.lower()
            if m == 'msp':
                score = 1.0 - probs_point.max(dim=1).values
            elif m == 'entropy':
                p = torch.clamp(probs_point, min=1e-8)
                score = -(p * p.log()).sum(dim=1)
            elif m == 'energy':
                score = -torch.logsumexp(point_logits, dim=1)
            elif m == 'max_logit':
                score = (-point_logits.max(dim=1).values) + 1.0
            else:
                score = 1.0 - probs_point.max(dim=1).values
            return score
    
        offset_coords_idx = 0

        for logit_raw, mask, map, label, n_point, seq in zip(
            pred_logits_raw,
            pred_masks,
            inverse_maps,
            original_labels,
            num_points,
            sequences,
        ):
            
            if seq != self.last_seq:
                self.last_seq = seq
                self.previous_instances = None
                self.max_instance_id = self.config.model.num_queries
                self.scene = 0

            # ========== ANOMALY DETECTION (configurable score) ==========        
            point_scores = _compute_point_anomaly_score(
                logit_raw, mask, self.validation_anomaly_score_name
            )  # [N]
            point_scores = point_scores[map]
            point_scores_np = point_scores.detach().cpu().numpy()

            # Ground-truth semantic labels (original order, so here anomaly will be 30)
            #print(np.unique(label[:,0], return_counts=True))
            sem_labels = self.validation_dataset._remap_model_output(label[:, 0])


            # Distance filtering to mirror compute_point_level_ood
            curr_coords_idx = mask.shape[0]
            start_idx = offset_coords_idx
            end_idx = start_idx + curr_coords_idx
            coords_slice = raw_coordinates[start_idx:end_idx, :3]
            coords_mapped = coords_slice[map].detach().cpu().numpy()
            dists = np.linalg.norm(coords_mapped, axis=1)
            dist_mask = (dists > self.val_min_eval_distance) & (dists < self.val_max_eval_distance)

            #print(np.unique(sem_labels, return_counts=True))

            # FILTER IGNORE LABELS then apply distance mask
            ignore_id = 0
            valid_mask = (sem_labels != ignore_id) & dist_mask

            scores_valid = point_scores_np[valid_mask]
            labels_valid = sem_labels[valid_mask]

            # Skip samples with too few anomaly points (like compute_point_level_ood)

            #print(np.unique(labels_valid, return_counts=True))

            if (labels_valid == 30).sum() < self.val_min_anomaly_points:
                # Advance offset for this sample before skipping
                offset_coords_idx += curr_coords_idx
                print(f"Skipping sample {seq} for anomaly eval: only {(labels_valid == 30).sum()} anomaly points")
                continue

            # Add to anomaly evaluator
            self.anomaly_evaluator.addBatch(
                anomaly_logits=scores_valid,
                semantic_labels=labels_valid,
            )

            # ========== SEMANTIC AND INSTANCE PREDICTIONS ==========
            logit_semantic = F.softmax(logit_raw, dim=-1)[:, :-1]  # [Q, 19]
            print(logit_semantic.shape)
            class_confidence, classes = torch.max(logit_semantic.detach().cpu(), dim=1)
            foreground_confidence = mask.detach().cpu().float().sigmoid()
            confidence = class_confidence[None, ...] * foreground_confidence
            confidence = confidence[map].numpy()

            ins_preds = np.argmax(confidence, axis=1)
            sem_preds = classes[ins_preds].numpy() + 1
            ins_preds += 1
            ins_preds[np.isin(sem_preds, self.config.data.stuff_cls_ids)] = 0
            ins_labels = label[:, 1] >> 16


            #PERCHE non fare direttamente così??? AL posto di usare le labels invertite?
            #CAPIRE COSA SUCCEDE NEL TRAINING, Per vedere se allena a predirre 29 o 30
            #sem_labels = label[:, 0]

            db_max_instance_id = self.config.model.num_queries
            if self.config.general.dbscan_eps is not None:
                # Advance offset only here to keep consistency with existing code
                curr_coords = raw_coordinates[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords = curr_coords[map].detach().cpu().numpy()

                ins_ids = np.unique(ins_preds)
                for ins_id in ins_ids:
                    if ins_id != 0:
                        instance_mask = ins_preds == ins_id
                        clusters = (
                            DBSCAN(
                                eps=self.config.general.dbscan_eps,
                                min_samples=1,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[instance_mask])
                            .labels_
                        )
                        new_mask = np.zeros(ins_preds.shape, dtype=np.int64)
                        new_mask[instance_mask] = clusters + 1
                        for cluster_id in np.unique(new_mask):
                            if cluster_id != 0:
                                db_max_instance_id += 1
                                ins_preds[new_mask == cluster_id] = db_max_instance_id

            self.max_instance_id = max(db_max_instance_id, self.max_instance_id)
            for i in range(len(n_point) - 1):
                indices = range(n_point[i], n_point[i + 1])
                if i == 0 and self.previous_instances is not None:
                    current_instances = ins_preds[indices]
                    associations = associate_instances(
                        self.previous_instances, current_instances
                    )
                    for id in np.unique(ins_preds):
                        if associations.get(id) is None:
                            self.max_instance_id += 1
                            associations[id] = self.max_instance_id
                    ins_preds = np.vectorize(associations.__getitem__)(ins_preds)
                else:
                    self.class_evaluator.addBatch(
                        sem_preds, ins_preds, sem_labels, ins_labels, indices, seq
                    )
            if i > 0:
                self.previous_instances = ins_preds[indices]
            
            offset_coords_idx += curr_coords_idx

        logs = generate_logs(losses,"val")
        logs["val_loss_mean"] = sum(losses.values()).cpu().item()
        self.validation_step_outputs.append(logs)
        return {f"val_{k}": v.detach().cpu().item() for k, v in losses.items()}

    def on_train_epoch_end(self):
        self.current_epoch_num += 1
        train_loss_mean = sum(self.training_step_outputs) / len(
            self.training_step_outputs
        )
        results = {"train_loss_mean": train_loss_mean}
        self.log_dict(results, sync_dist=True)
        self.training_step_outputs.clear()
        print(results)

    def on_validation_epoch_end(self):
        self.last_seq = None
        
        # ========== PANOPTIC SEGMENTATION METRICS ==========
        class_names = self.config.data.class_names
        pq, sq, rq, all_pq, all_sq, all_rq = self.class_evaluator.getPQ()
        self.class_evaluator.reset()

        #SMTHING SHERE IS BROKEN.... CHECK THE VALIDATION STEP

        results = {}
        results["val_mean_pq"] = float(pq)
        results["val_mean_sq"] = float(sq)
        results["val_mean_rq"] = float(rq)
        
        for i, (pq, sq, rq) in enumerate(zip(all_pq, all_sq, all_rq)):
            results[f"val_{class_names[i-1]}_pq"] = pq.item()
            results[f"val_{class_names[i-1]}_sq"] = sq.item()
            results[f"val_{class_names[i-1]}_rq"] = rq.item()
        
        # ========== ANOMALY DETECTION METRICS ==========
        if self.trainer.is_global_zero:
            print("\n" + "="*60)
            print("ANOMALY DETECTION RESULTS")
            print("="*60)
        
        try:
            anomaly_metrics = self.anomaly_evaluator.getAllMetrics()
            
            # Add anomaly metrics to results
            results["val_aupr"] = anomaly_metrics['AUPR']  # Primary metric for model selection
            results["val_auroc"] = anomaly_metrics['AUROC']
            results["val_fpr95"] = anomaly_metrics['FPR@95']

            
            # Print anomaly metrics
            if self.trainer.is_global_zero:
                self.anomaly_evaluator.printMetrics()
            
        except Exception as e:
            if self.trainer.is_global_zero:
                print(f"ERROR computing anomaly metrics: {e}")
                import traceback
                traceback.print_exc()
            
            # Add default values to prevent crashes
            results["val_auroc"] = 0.0
            results["val_aupr"] = 0.0
            results["val_fpr95"] = 1.0
            results["val_anomaly_precision"] = 0.0
            results["val_anomaly_recall"] = 0.0
            results["val_anomaly_f1"] = 0.0
            results["val_anomaly_accuracy"] = 0.0
            results["val_anomaly_tp"] = 0
            results["val_anomaly_fp"] = 0
            results["val_anomaly_tn"] = 0
            results["val_anomaly_fn"] = 0
    
        finally:
            # Always reset anomaly evaluator for next epoch
            self.anomaly_evaluator.reset()
            if self.trainer.is_global_zero:
                print("="*60 + "\n")
    
        # ========== LOG ALL METRICS ==========
        # Log on rank 0 only to avoid DDP all-reduce hangs; callbacks run on rank 0.
        self.log_dict(results, sync_dist=False, rank_zero_only=True)
        if self.trainer.is_global_zero:
            print(results)

        # ========== LOSS AGGREGATION ==========
        dd = defaultdict(list)
        for output in self.validation_step_outputs:
            for key, val in output.items():
                dd[key].append(val)
        
        dd = {k: statistics.mean(v) for k, v in dd.items()}
        
        # dd["val_mean_loss_ce"] = statistics.mean(
        #     [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        # )
        # dd["val_mean_loss_mask"] = statistics.mean(
        #     [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        # )
        # dd["val_mean_loss_dice"] = statistics.mean(
        #     [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        # )
        # dd["val_mean_loss_box"] = statistics.mean(
        #     [item for item in [v for k, v in dd.items() if "loss_box" in k]]
        # )
        #self.log_dict(dd)
        self.validation_step_outputs.clear()
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        if self.freeze_first_epoch and self.current_epoch_num == 0:
            print("=" * 50)
            print("FIRST EPOCH: Freezing model except specific parts")
            print("Keeping unfrozen parts:", self.unfrozen_parts)
            print("=" * 50)
            
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze specific parts
            unfrozen_params = 0
            total_params = 0
            
            for name, module in self.model.named_modules():
                total_params += sum(p.numel() for p in module.parameters())
                
                # Check if this module should remain unfrozen
                should_unfreeze = any(part in name for part in self.unfrozen_parts)
                
                if should_unfreeze:
                    for param in module.parameters():
                        param.requires_grad = True
                        unfrozen_params += param.numel()
                    print(f"✓ Unfrozen: {name}")
            
            print(f"Unfrozen {unfrozen_params}/{total_params} parameters ({100*unfrozen_params/total_params:.2f}%)")
            print("=" * 50)

        elif self.freeze_first_epoch and self.current_epoch_num == self.defreezing_epoch:
            print("=" * 50) 
            print("EPOCH 1+: Unfreezing entire model")
            print("=" * 50)
            
            # Unfreeze all parameters
            for param in self.model.parameters():
                param.requires_grad = True
                
            # Disable further freezing
            self.freeze_first_epoch = False
            print("✓ All parameters unfrozen")
        if self.freeze_backbone:
            self._freeze_backbone()


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def setup(self, stage=None):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def prepare_data(self):
        return

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        sampler = None
        if self.val_num_samples is not None:
            n = len(self.validation_dataset)
            k = min(int(self.val_num_samples), n)
            sampler = RandomSampler(
                self.validation_dataset,
                replacement=False,
                num_samples=k,
                generator=self._val_sampler_generator,
            )

        num_workers = int(self.val_num_workers)
        persistent_workers = bool(self.val_persistent_workers) and num_workers > 0
        timeout_s = int(self.val_timeout_s) if num_workers > 0 else 0
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
            sampler=sampler,
            num_workers=num_workers,
            timeout=timeout_s,
            persistent_workers=persistent_workers,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
    
    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading - insert anomaly class at penultimate position"""        
        reset_optimizer = True
        if reset_optimizer:
            print("\n--- Resetting training state ---")
            checkpoint['epoch'] = 0
            checkpoint['global_step'] = 0
            if 'optimizer_states' in checkpoint:
                checkpoint['optimizer_states'] = []
                print("  ✓ Reset optimizer states")
            
            if 'lr_schedulers' in checkpoint:
                checkpoint['lr_schedulers'] = []
                print("  ✓ Reset lr_scheduler states")

