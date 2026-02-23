import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def box_loss(inputs: torch.Tensor, targets: torch.Tensor, num_bboxs: float):
    loss = F.l1_loss(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_bboxs


box_loss_jit = torch.jit.script(box_loss)  # type: torch.jit.ScriptModule


# ========== PEBAL Loss Components ==========

def smooth(arr, lamda1):
    """Smoothness regularization for energy maps"""
    new_array = arr
    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss


def sparsity(arr, lamda2):
    """Sparsity regularization"""
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def energy_loss(logits, targets, num_class=19, ood_ind=254, void_ind=255):
    """
    Energy-based OOD loss.
    
    Args:
        logits: [B, C, H, W] or [B, C, N] - model logits (C includes num_class + no-object class)
        targets: [B, H, W] or [B, N] - ground truth labels
        num_class: number of in-distribution classes (default 19 for SemanticKITTI)
        ood_ind: label index for OOD/anomaly points (default 254)
        void_ind: label index for void/ignore points (default 255)
    
    Returns:
        loss: scalar energy loss
        energy: [B, H, W] or [B, N] - energy map
    """
    T = 1.
    m_in = -12
    m_out = -6

    # Handle different input shapes (2D images or 1D point clouds)
    if logits.dim() == 4:  # [B, C, H, W]
        energy = -(T * torch.logsumexp(logits[:, :num_class, :, :] / T, dim=1))
    else:  # [B, C, N] or [B, N]
        if logits.dim() == 3:
            energy = -(T * torch.logsumexp(logits[:, :num_class, :] / T, dim=1))
        else:  # [B, N]
            energy = -(T * torch.logsumexp(logits[:, :num_class] / T, dim=1))
    
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


class Gambler(torch.nn.Module):
    """Gambler's loss for OOD detection with reservation mechanism"""
    
    def __init__(self, reward, device, pretrain=-1, ood_reg=.1, ood_ind=19):
        super(Gambler, self).__init__()
        self.reward = torch.tensor([reward]).cuda(device)
        self.pretrain = pretrain
        self.ood_reg = ood_reg
        self.device = device
        self.ood_ind = ood_ind

    def forward(self, pred, targets, wrong_sample=False):
        """
        Args:
            pred: [B, C+1, H, W] or [B, C+1, N] - logits with reservation class
            targets: [B, H, W] or [B, N] - ground truth labels
            wrong_sample: whether OOD pixels are present
        """
        pred_prob = torch.softmax(pred, dim=1)

        assert torch.all(pred_prob > 0), print(pred_prob[pred_prob <= 0])
        assert torch.all(pred_prob <= 1), print(pred_prob[pred_prob > 1])
        
        true_pred, reservation = pred_prob[:, :-1, :, :], pred_prob[:, -1, :, :]

        # compute the reward via the energy score;
        reward = torch.logsumexp(pred[:, :-1, :, :], dim=1).pow(2)

        if reward.nelement() > 0:
            gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
            reward = reward.unsqueeze(0)
            reward = gaussian_smoothing(reward)
            reward = reward.squeeze(0)
        else:
            reward = self.reward

        # Get number of classes from true_pred shape
        num_classes = true_pred.shape[1]
        
        if wrong_sample:  # if there's ood pixels inside the image
            reservation = torch.div(reservation, reward)
            mask = targets == self.ood_ind  # Anomaly class
            # mask out each of the ood output channel
            reserve_boosting_energy = torch.add(true_pred, reservation.unsqueeze(1))[mask.unsqueeze(1).
                repeat(1, num_classes, 1, 1)]
            
            gambler_loss_out = torch.tensor([.0], device=self.device)
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                gambler_loss_out = self.ood_reg * reserve_boosting_energy

            # gambler loss for in-lier pixels
            # Void/no-object mask (background class which is num_classes or higher)
            void_mask = targets >= num_classes
            targets_copy = targets.clone()
            targets_copy[void_mask] = 0  # make void pixel to 0
            targets_copy[mask] = 0  # make ood pixel to 0
            gambler_loss_in = torch.gather(true_pred, index=targets_copy.unsqueeze(1), dim=1).squeeze()
            gambler_loss_in = torch.add(gambler_loss_in, reservation)

            # exclude the ood pixel mask and void pixel mask
            gambler_loss_in = gambler_loss_in[(~mask) & (~void_mask)].log()
            return -(gambler_loss_in.mean() + gambler_loss_out.mean())
        else:
            # Void/no-object mask
            mask = targets >= num_classes
            targets_copy = targets.clone()
            targets_copy[mask] = 0
            reservation = torch.div(reservation, reward)
            gambler_loss = torch.gather(true_pred, index=targets_copy.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, reservation)
            gambler_loss = gambler_loss[~mask].log()
            return -gambler_loss.mean()


class SetCriterionContrastive(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, **kwargs):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            **kwargs: additional parameters from config (OHEM, focal, distance weighting, PEBAL, etc.)
        """
        super().__init__()
        self.num_classes = num_classes
        print(f"loss num_classes: {num_classes}")
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # Hard-example mining / focal-loss knobs (defaults keep old behavior)
        self.ohem_enable = kwargs.get("ohem_enable", False)   # set True to enable OHEM
        self.ohem_frac = kwargs.get("ohem_frac", 0.3)         # top fraction per class (ID/OOD)
        self.ohem_min_k = kwargs.get("ohem_min_k", 512)       # minimum top-k per class
        self.use_focal = kwargs.get("use_focal", False)       # set True to enable focal
        self.focal_gamma = kwargs.get("focal_gamma", 1.5)
        self.focal_alpha = kwargs.get("focal_alpha", 0.5)
        self.contrastive_name = kwargs.get("contrastive_name", "bce")  # 'bce' or 'margin' 
        # Which anomaly score to use for training ('msp' | 'entropy' | 'energy' | 'max_logit')
        self.anomaly_score_name = kwargs.get("anomaly_score_name", "msp")
        self.anomaly_class_id = kwargs.get("anomaly_class_id", 18)

        # Alternative reduction for anomaly BCE loss (set via config)
        self.anomaly_instance_mean = kwargs.get("anomaly_instance_mean", False)
        
        # Margin-based loss parameters
        self.margin = kwargs.get("margin", 2.0)
        
        # Distance weighting parameters
        self.use_distance_weighting = kwargs.get("use_distance_weighting", False)
        self.distance_weight_mode = kwargs.get("distance_weight_mode", "linear")
        self.distance_weight_center = kwargs.get("distance_weight_center", 45.0)
        self.distance_weight_sigma = kwargs.get("distance_weight_sigma", 10.0)
        self.distance_weight_min = kwargs.get("distance_weight_min", 1.0)
        self.distance_weight_max = kwargs.get("distance_weight_max", 3.0)
        self.distance_weight_linear_end = kwargs.get("distance_weight_linear_end", 50.0)
        
        # Advanced BCE weighting configuration
        self.use_bce_weighting_product = kwargs.get("use_bce_weighting_product", False)
        self.use_bce_weighting_sum = kwargs.get("use_bce_weighting_sum", False)
        self.bce_weight_sum_coeffs = kwargs.get("bce_weight_sum_coeffs", [1.0, 1.0, 1.0])
        self.bce_weight_distance_max = kwargs.get("bce_weight_distance_max", 80.0)
        self.bce_weight_spatial_k = kwargs.get("bce_weight_spatial_k", 16)
        self.bce_weight_spatial_max_points = kwargs.get("bce_weight_spatial_max_points", 6000)
        self.bce_weight_target_mean = kwargs.get("bce_weight_target_mean", 1.0)
        self.bce_weight_min_value = kwargs.get("bce_weight_min_value", 0.5)
        self.bce_weight_max_value = kwargs.get("bce_weight_max_value", 2.0)
        self.bce_weight_ohem_ratio = kwargs.get("bce_weight_ohem_ratio", 0.7)

        # PEBAL loss configuration
        self.pebal_reward = kwargs.get("pebal_reward", 4.5)  # Gambler reward parameter
        self.pebal_energy_weight = kwargs.get("pebal_energy_weight", 0.1)  # Energy loss weight
        self.instance_variance_weight = kwargs.get("instance_variance_weight", 1.0)
        self.instance_variance_min_points = kwargs.get("instance_variance_min_points", 5)
        
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef

        # self.register_buffer("empty_weight", empty_weight)
        self.register_buffer("ce_class_weights", empty_weight)
        
        # Gambler will be lazily instantiated when needed
        self.gambler = None

    def loss_contrastive(self, outputs, targets, indices, raw_coordinates=None):
        type = self.contrastive_name  # 'bce' or 'margin'
        if type == 'bce':
            return self.loss_contrastive_bce(outputs, targets, indices, raw_coordinates=raw_coordinates)
        elif type == 'margin':
            return self.loss_contrastive_margin(outputs, targets, indices, raw_coordinates=raw_coordinates)
        
    def _compute_anomaly_score(self, pred_logits_batch: torch.Tensor, mask_logits_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute per-point anomaly score based on self.anomaly_score_name.
        Choices:
          - 'msp':      1 - max softmax probability (over 19 classes)
          - 'entropy':  entropy of softmax probabilities
          - 'energy':   -logsumexp of logits (over 19 classes)
          - 'max_logit':(-max logit) + 1 (over 19 classes)
        Returns a tensor of shape [num_points], higher should mean more anomalous.
        """
        # Mask probs: [N, Q]
        mask_probs = mask_logits_batch.float().sigmoid()
        # Class probabilities per query: [Q, 19] (exclude no-object)
        probs_q = F.softmax(pred_logits_batch, dim=-1)[:, :-1]
        point_softmax = mask_probs.matmul(probs_q)
        point_logits = mask_probs.matmul(pred_logits_batch[:, :-1])

        name = self.anomaly_score_name.lower()
        if name == "msp":
            score = 1.0 - point_softmax.max(dim=1).values
        elif name == "entropy":
            p = torch.clamp(point_softmax, min=1e-8)
            score = -(p * p.log()).sum(dim=1)
        elif name == "energy":
            score = -torch.logsumexp(point_logits, dim=1)
        elif name == "max_logit":
            score = (-point_logits.max(dim=1).values) + 1.0
        else:
            # Fallback to MSP
            score = 1.0 - point_softmax.max(dim=1).values
        return score

    def _compute_bce_distance_component(self, coords, num_points, device):
        """Return distance-based weights in [0.5, 1.0]; farther points get larger scores."""
        if coords is None or coords.numel() == 0:
            return torch.full((num_points,), 0.75, device=device)

        coords = coords.float()
        distances = torch.norm(coords, dim=1)
        normalized = torch.clamp(distances / max(self.bce_weight_distance_max, 1e-3), 0.0, 1.0)
        return 0.5 + 0.5 * normalized

    def _compute_bce_confidence_component(self, anomaly_scores, targets):
        """Return OHEM-style weights based on anomaly score difficulty."""
        if anomaly_scores is None or targets is None or anomaly_scores.numel() == 0:
            return torch.ones_like(targets, dtype=torch.float32)

        scores = anomaly_scores.detach()
        targets = targets.float()
        score_min = scores.min()
        score_max = scores.max()
        if (score_max - score_min) < 1e-6:
            normalized_scores = torch.zeros_like(scores)
        else:
            normalized_scores = (scores - score_min) / (score_max - score_min + 1e-8)

        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        difficulty = torch.zeros_like(normalized_scores)
        difficulty[pos_mask] = 1.0 - normalized_scores[pos_mask]
        difficulty[neg_mask] = normalized_scores[neg_mask]

        weights = torch.zeros_like(normalized_scores)

        def _assign(mask):
            count = int(mask.sum().item())
            if count == 0:
                return
            keep = max(1, int(count * self.bce_weight_ohem_ratio))
            keep = min(count, keep)
            vals = difficulty[mask]
            top_vals, top_idx = torch.topk(vals, keep, largest=True)
            if top_vals.numel() > 1:
                local_min = top_vals.min()
                local_max = top_vals.max()
                if (local_max - local_min) < 1e-6:
                    norm_vals = torch.ones_like(top_vals)
                else:
                    norm_vals = (top_vals - local_min) / (local_max - local_min + 1e-8)
            else:
                norm_vals = torch.ones_like(top_vals)
            idxs = torch.where(mask)[0][top_idx]
            weights[idxs] = norm_vals

        _assign(pos_mask)
        _assign(neg_mask)

        if weights.max() < 1e-6:
            return torch.ones_like(weights)

        return torch.clamp(weights, 0.0, 1.0)

    def _compute_bce_spatial_component(self, coords, anomaly_scores):
        """Return weights emphasizing regions with high neighbour variance of anomaly scores."""
        if anomaly_scores is None or anomaly_scores.numel() == 0:
            return None

        num_points = anomaly_scores.shape[0]
        device = anomaly_scores.device

        if coords is None or coords.numel() == 0 or num_points <= 1:
            return torch.full((num_points,), 0.3, device=device)

        if num_points > self.bce_weight_spatial_max_points:
            return torch.full((num_points,), 0.3, device=device)

        k = min(self.bce_weight_spatial_k, num_points - 1)
        if k <= 0:
            return torch.full((num_points,), 0.3, device=device)

        coords = coords.float()
        dists = torch.cdist(coords, coords)
        knn_dists, knn_indices = torch.topk(dists, k + 1, largest=False, dim=1)
        knn_indices = knn_indices[:, 1:]

        neighbor_scores = anomaly_scores[knn_indices]
        variance = torch.var(neighbor_scores, dim=1, unbiased=False)

        var_min = variance.min()
        var_max = variance.max()
        if (var_max - var_min) < 1e-6:
            weights = torch.full_like(variance, 0.3)
        else:
            weights = (variance - var_min) / (var_max - var_min + 1e-8)

        return torch.clamp(weights, 0.0, 1.0)

    def _build_bce_point_weights(self, coords, anomaly_scores, targets):
        """Assemble per-point weights using configured combination strategy."""
        device = anomaly_scores.device
        num_points = anomaly_scores.shape[0]

        distance_component = self._compute_bce_distance_component(coords, num_points, device)
        confidence_component = self._compute_bce_confidence_component(anomaly_scores, targets)
        spatial_component = self._compute_bce_spatial_component(coords, anomaly_scores)
        if spatial_component is None:
            spatial_component = torch.full((num_points,), 0.3, device=device)

        if self.use_bce_weighting_product:
            combined = distance_component * confidence_component * spatial_component
            return self._normalize_bce_weight_map(combined)

        if self.use_bce_weighting_sum:
            coeffs = self.bce_weight_sum_coeffs
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=anomaly_scores.dtype, device=device)
                self.bce_weight_sum_coeffs = coeffs
            else:
                coeffs = coeffs.to(device=device, dtype=anomaly_scores.dtype)
            coeffs = coeffs.flatten()
            if coeffs.numel() != 3:
                coeffs = torch.ones(3, device=device, dtype=anomaly_scores.dtype)
            total = torch.clamp(coeffs.sum(), min=1e-6)
            stacked = torch.stack([distance_component, confidence_component, spatial_component], dim=0)
            combined = torch.matmul(coeffs, stacked) / total
            return self._normalize_bce_weight_map(combined)

        return None

    def _normalize_bce_weight_map(self, weights):
        """Rescale weights to keep mean near target and clamp extremes for stability."""
        if weights is None:
            return None
        target_mean = max(self.bce_weight_target_mean, 1e-6)
        current_mean = weights.mean()
        if current_mean > 1e-6:
            weights = weights * (target_mean / current_mean)
        weights = torch.clamp(weights, min=self.bce_weight_min_value, max=self.bce_weight_max_value)
        return weights

    def loss_contrastive_margin(self, outputs, targets, indices, raw_coordinates=None):
        """
        Point-level anomaly loss using margin-based approach.
        
        All targets passed here are already anomaly masks (label 19).
        Uses squared loss with margin for separating anomaly and normal points:
        - Normal (ID) points: minimize score^2 (want score close to 0)
        - Anomaly (OOD) points: minimize max(margin - score, 0)^2 (want score >= margin)
        """

        # Extract predictions
        pred_masks_list = outputs["pred_masks"]  # [B, Q, num_points]
        pred_logits_list = outputs["pred_logits"]  # [B, Q, num_classes+1]

        # Margin hyperparameter - points above this are considered anomalies
        margin = self.margin

        total_loss = torch.tensor(0.0, device=pred_logits_list[0].device)
        total_loss_id = torch.tensor(0.0, device=pred_logits_list[0].device)
        total_loss_ood = torch.tensor(0.0, device=pred_logits_list[0].device)
        total_batches = 0

        # Statistics
        total_points = 0
        total_anomaly_points = 0
        avg_score_on_anomaly_points = []
        avg_score_on_normal_points = []

        # Collect all scores for quartile computation and min/max
        all_anomaly_scores = []
        all_normal_scores = []

        # Process each batch
        for batch_id in range(len(targets)):
            # ========== CREATE GT ANOMALY MASK ==========
            target_masks = targets[batch_id]["masks"]  # [num_anomaly_masks, num_points]
            
            # Create binary OOD mask: 1 for any point covered by ANY anomaly mask
            if target_masks.shape[0] == 0:
                # No anomaly masks in this batch
                num_points = pred_masks_list[batch_id].shape[0]
                ood_mask = torch.zeros(num_points, dtype=torch.bool, device=pred_logits_list[batch_id].device)
            else:
                ood_mask = (target_masks.sum(dim=0) > 0)  # [num_points]
            
            id_mask = ~ood_mask  # Normal/inlier mask
            
            num_points = ood_mask.shape[0]
            num_anomaly_points = ood_mask.sum().item()
            num_normal_points = id_mask.sum().item()
            
            # ========== COMPUTE ANOMALY SCORES ==========
            # Get predictions for this batch
            mask_logits_batch = pred_masks_list[batch_id]  # [num_points, Q]
            pred_logits_batch = pred_logits_list[batch_id]  # [Q, num_classes+1]
            
            # Use configured anomaly score
            score = self._compute_anomaly_score(pred_logits_batch, mask_logits_batch)  # [num_points]
            
            # Extract scores for OOD and ID points
            ood_score = score[ood_mask] if ood_mask.sum() > 0 else None
            id_score = score[id_mask] if id_mask.sum() > 0 else None
            
            # ========== COMPUTE LOSS ==========
            # ID points: minimize score^2 (want score close to 0)
            if id_score is not None:
                loss_id = torch.pow(id_score, 2).mean()
            else:
                loss_id = torch.tensor(0.0, device=score.device)
            
            # OOD points: minimize max(margin - score, 0)^2 (want score >= margin)
            if ood_score is not None:
                loss_ood = torch.pow(torch.clamp(margin - ood_score, min=0.0), 2).mean()
            else:
                loss_ood = torch.tensor(0.0, device=score.device)
            
            # Combined loss (average of ID and OOD losses)
            batch_loss = 0.5 * (loss_id + loss_ood)

            if self.instance_variance_weight > 0:
                instance_variance_loss = torch.tensor(0.0, device=score.device)
                if target_masks.shape[0] > 0:
                    for i in range(target_masks.shape[0]):
                        mask_i = target_masks[i] > 0.5
                        if mask_i.sum() >= self.instance_variance_min_points:
                            inst_scores = score[mask_i]
                            # unbiased=True is default for torch.var (Bessel correction)
                            instance_variance_loss += torch.var(inst_scores)
                
                batch_loss += self.instance_variance_weight * instance_variance_loss

            total_loss += batch_loss
            total_loss_id += loss_id
            total_loss_ood += loss_ood
            total_batches += 1
            
            # ========== STATISTICS ==========
            total_points += num_points
            total_anomaly_points += num_anomaly_points
            
            if num_anomaly_points > 0:
                avg_score_on_anomaly_points.append(score[ood_mask].mean().item())
                all_anomaly_scores.append(score[ood_mask].detach().cpu())
            
            if num_normal_points > 0:
                avg_score_on_normal_points.append(score[id_mask].mean().item())
                all_normal_scores.append(score[id_mask].detach().cpu())

        # Average loss across batches
        if total_batches > 0:
            final_loss = total_loss / total_batches
            final_loss_id = total_loss_id / total_batches
            final_loss_ood = total_loss_ood / total_batches
        else:
            final_loss = total_loss
            final_loss_id = total_loss_id
            final_loss_ood = total_loss_ood

        # Compute quartiles, min, and max
        if len(all_anomaly_scores) > 0:
            all_anomaly_scores_cat = torch.cat(all_anomaly_scores)
            q25_anomaly, q50_anomaly, q75_anomaly = torch.quantile(
                all_anomaly_scores_cat, torch.tensor([0.25, 0.5, 0.75])
            ).tolist()
            min_anomaly = all_anomaly_scores_cat.min().item()
            max_anomaly = all_anomaly_scores_cat.max().item()
        else:
            q25_anomaly = q50_anomaly = q75_anomaly = 0.0
            min_anomaly = max_anomaly = 0.0
    
        if len(all_normal_scores) > 0:
            all_normal_scores_cat = torch.cat(all_normal_scores)
            q25_normal, q50_normal, q75_normal = torch.quantile(
                all_normal_scores_cat, torch.tensor([0.25, 0.5, 0.75])
            ).tolist()
            min_normal = all_normal_scores_cat.min().item()
            max_normal = all_normal_scores_cat.max().item()
        else:
            q25_normal = q50_normal = q75_normal = 0.0
            min_normal = max_normal = 0.0

        # Statistics
        stats = {
            "avg_score_on_anomaly_points": sum(avg_score_on_anomaly_points) / len(avg_score_on_anomaly_points) if len(avg_score_on_anomaly_points) > 0 else 0.0,
            "avg_score_on_normal_points": sum(avg_score_on_normal_points) / len(avg_score_on_normal_points) if len(avg_score_on_normal_points) > 0 else 0.0,
            "total_anomaly_points": total_anomaly_points,
            "total_points": total_points,
        }

        print(f"Point-level Anomaly Loss (Margin={margin:.2f}): {final_loss.item():.6f} | "
              f"ID Loss: {final_loss_id.item():.6f} | OOD Loss: {final_loss_ood.item():.6f}")
        print(f"  Anomaly Points: {total_anomaly_points} / {total_points}")
        print(f"  Anomaly scores - Mean: {stats['avg_score_on_anomaly_points']:.6f} | "
              f"Min: {min_anomaly:.6f} | Q25: {q25_anomaly:.6f} | Median: {q50_anomaly:.6f} | Q75: {q75_anomaly:.6f} | Max: {max_anomaly:.6f}")
        print(f"  Normal scores  - Mean: {stats['avg_score_on_normal_points']:.6f} | "
              f"Min: {min_normal:.6f} | Q25: {q25_normal:.6f} | Median: {q50_normal:.6f} | Q75: {q75_normal:.6f} | Max: {max_normal:.6f}")

        return {"loss_contrastive": final_loss}
    
    # Helper to compute (optionally focal-weighted) per-point BCE loss vector
    def per_point_bce_with_optional_focal(self, logits_vec, target_vec, is_pos: bool):
        # BCE per point (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits_vec, target_vec, reduction="none")
        if not self.use_focal:
            return bce  # [N]

        # Focal reweighting
        p = torch.sigmoid(logits_vec)
        # pt = p if y=1 else (1-p)
        pt = torch.where(target_vec > 0.5, p, 1.0 - p)
        weight = (1.0 - pt).pow(self.focal_gamma)

        # Class-balance alpha (apply different alpha to pos/neg)
        alpha = self.focal_alpha if is_pos else (1.0 - self.focal_alpha)
        return alpha * weight * bce  # [N]

    # Reduce with OHEM or mean
    def reduce_loss(self, vec, device, count_hint: int, is_pos: bool):
        if vec is None or vec.numel() == 0:
            return torch.tensor(0.0, device=device)
        if self.ohem_enable:
            # pick top-k hardest per class
            k = max(1, min(vec.numel(), max(self.ohem_min_k, int(self.ohem_frac * max(1, count_hint)))))
            topk_vals, _ = torch.topk(vec, k, largest=True, sorted=False)
            return topk_vals.mean()
        else:
            return vec.mean()

    def loss_contrastive_bce(self, outputs, targets, indices, raw_coordinates=None):
        """
        Point-level anomaly loss using BCE with logits.
        
        All targets passed here are already anomaly masks (label 19).
        Computes negative max logit as anomaly score, then applies BCE loss:
        - Anomaly points (covered by masks) should have score close to 1 (low max logit)
        - Normal points (not covered) should have score close to 0 (high max logit)
        
        Optionally applies traditional distance weighting and/or the new 3-component
        weighting (distance, confidence, spatial contrast) combined via product or
        weighted sum.
        """
        # Extract predictions
        pred_masks_list = outputs["pred_masks"]  # [B, Q, num_points]
        pred_logits_list = outputs["pred_logits"]  # [B, Q, num_classes+1]
        
        # Distance weighting parameters (configured via config)
        use_distance_weighting = self.use_distance_weighting
        distance_weight_mode = self.distance_weight_mode
        distance_center = self.distance_weight_center  # meters
        distance_sigma = self.distance_weight_sigma    # meters
        weight_min = self.distance_weight_min
        weight_max = self.distance_weight_max
        linear_end = self.distance_weight_linear_end   # meters
        
        total_loss = torch.tensor(0.0, device=pred_logits_list[0].device)
        total_batches = 0
        
        # Statistics
        total_points = 0
        total_anomaly_points = 0
        avg_score_on_anomaly_points = []
        avg_score_on_normal_points = []
        
        # Offset for raw_coordinates indexing (now for voxelized coords)
        offset_coords_idx = 0
        weight_mode_enabled = self.use_bce_weighting_product or self.use_bce_weighting_sum
        
        # Process each batch
        for batch_id in range(len(targets)):
            # ========== CREATE GT ANOMALY MASK ==========
            target_masks = targets[batch_id]["masks"]  # [num_anomaly_masks, num_points]
            
            # Create binary OOD mask: 1 for any point covered by ANY anomaly mask
            if target_masks.shape[0] == 0:
                # No anomaly masks in this batch
                num_points = pred_masks_list[batch_id].shape[0]
                ood_mask = torch.zeros(num_points, dtype=torch.bool, device=pred_logits_list[batch_id].device)
            else:
                ood_mask = (target_masks.sum(dim=0) > 0)  # [num_points]
            
            id_mask = ~ood_mask  # Normal/inlier mask
            
            num_points = ood_mask.shape[0]
            num_anomaly_points = ood_mask.sum().item()
            num_normal_points = id_mask.sum().item()
            
            # ========== COMPUTE DISTANCE WEIGHTS ==========
            distance_weights = None
            curr_coords = None
            curr_coords_idx = pred_masks_list[batch_id].shape[0]
            if raw_coordinates is not None:
                curr_coords = raw_coordinates[
                    offset_coords_idx : offset_coords_idx + curr_coords_idx, :3
                ].to(pred_logits_list[batch_id].device)

            if use_distance_weighting and curr_coords is not None:
                # Compute Euclidean distance from origin (sensor position)
                distances = torch.norm(curr_coords, dim=1)  # [num_points_voxelized]
                if distance_weight_mode == "gaussian":
                    # Gaussian weighting centered at distance_center
                    gaussian = torch.exp(-0.5 * ((distances - distance_center) / distance_sigma) ** 2)
                    distance_weights = weight_min + (weight_max - weight_min) * gaussian
                elif distance_weight_mode == "linear":
                    # Linear kernel: linearly ramp from 0 to linear_end, then flat
                    t = torch.clamp(distances / max(1e-6, linear_end), min=0.0, max=1.0)
                    distance_weights = weight_min + (weight_max - weight_min) * t
                elif distance_weight_mode == "quadratic":
                    t = torch.clamp(distances / max(1e-6, linear_end), min=0.0, max=1.0)
                    distance_weights = weight_min + (weight_max - weight_min) * (t ** 1.5)  
                else:
                    raise ValueError(f"Unknown distance weight mode: {distance_weight_mode}")
                distance_weights = distance_weights.to(pred_logits_list[batch_id].device)
                
                # # DEBUG: Print first 10 distances and weights
                # print(f"\n[Batch {batch_id}] Distance Weighting Debug:")
                # print(f"  Total points: {len(distances)}")
                # print(f"  Center: {distance_center:.2f}m, Sigma: {distance_sigma:.2f}m")
                # print(f"  Weight range: [{weight_min:.2f}, {weight_max:.2f}]")
                # n_show = min(10, len(distances))
                # print(f"  First {n_show} distances (m):", distances[:n_show].cpu().numpy().round(2))
                # print(f"  First {n_show} weights:     ", distance_weights[:n_show].cpu().numpy().round(4))
                # print(f"  Distance stats - Min: {distances.min().item():.2f}m, "
                #       f"Mean: {distances.mean().item():.2f}m, "
                #       f"Max: {distances.max().item():.2f}m")
                # print(f"  Weight stats   - Min: {distance_weights.min().item():.4f}, "
                #       f"Mean: {distance_weights.mean().item():.4f}, "
                #       f"Max: {distance_weights.max().item():.4f}")
                
            if raw_coordinates is not None:
                # Update offset for next batch once coordinates are consumed
                offset_coords_idx += curr_coords_idx
            
            # ========== COMPUTE ANOMALY SCORES ==========
            # Get predictions for this batch
            mask_logits_batch = pred_masks_list[batch_id]  # [num_points, Q]
            pred_logits_batch = pred_logits_list[batch_id]  # [Q, num_classes+1]
            
            # Use configured anomaly score
            score = self._compute_anomaly_score(pred_logits_batch, mask_logits_batch)  # [num_points]
            
            if ood_mask.sum() > 0:
                ood_score = score[ood_mask]
                tgt_ood = torch.ones_like(ood_score)
                weight_ood = distance_weights[ood_mask] if distance_weights is not None else None
            else:
                ood_score, tgt_ood, weight_ood = None, None, None

            if id_mask.sum() > 0:
                id_score = score[id_mask]
                tgt_id = torch.zeros_like(id_score)
                weight_id = distance_weights[id_mask] if distance_weights is not None else None
            else:
                id_score, tgt_id, weight_id = None, None, None
            
            # Build per-point loss vectors (optionally focal-weighted)
            loss_vec_ood = self.per_point_bce_with_optional_focal(ood_score, tgt_ood, is_pos=True) if ood_score is not None else None
            loss_vec_id  = self.per_point_bce_with_optional_focal(id_score,  tgt_id,  is_pos=False) if id_score  is not None else None

            # Advanced per-point weighting (product or weighted-sum)
            point_weight_map = None
            if weight_mode_enabled:
                binary_targets = ood_mask.float()
                point_weight_map = self._build_bce_point_weights(
                    curr_coords if curr_coords is not None else None,
                    score,
                    binary_targets.to(score.device)
                )

            # Apply distance weighting if enabled
            if distance_weights is not None:
                if loss_vec_ood is not None:
                    loss_vec_ood = loss_vec_ood * weight_ood
                if loss_vec_id is not None:
                    loss_vec_id = loss_vec_id * weight_id

            if point_weight_map is not None:
                if loss_vec_ood is not None:
                    loss_vec_ood = loss_vec_ood * point_weight_map[ood_mask]
                if loss_vec_id is not None:
                    loss_vec_id = loss_vec_id * point_weight_map[id_mask]

            # Reduce (OHEM/mean OR per-instance mean)
            if self.anomaly_instance_mean and (target_masks is not None) and target_masks.shape[0] > 0 and (loss_vec_ood is not None) and loss_vec_ood.numel() > 0:
                instance_means = []
                # loss_vec_ood is aligned with ood_mask order
                for i in range(target_masks.shape[0]):
                    inst_mask = target_masks[i] > 0.5  # [num_points]
                    if inst_mask.any():
                        inst_mask_in_ood = inst_mask[ood_mask]
                        if inst_mask_in_ood.any():
                            instance_means.append(loss_vec_ood[inst_mask_in_ood].mean())
                if len(instance_means) > 0:
                    loss_ood = torch.stack(instance_means).mean()
                else:
                    loss_ood = torch.tensor(0.0, device=pred_logits_list[batch_id].device)
            else:
                loss_ood = self.reduce_loss(loss_vec_ood, pred_logits_list[batch_id].device, int(ood_mask.sum().item()), is_pos=True)

            loss_id  = self.reduce_loss(loss_vec_id,  pred_logits_list[batch_id].device,  int(id_mask.sum().item()),  is_pos=False)

            batch_loss = 0.5 * (loss_id + loss_ood)
            
            if self.instance_variance_weight > 0:
                instance_variance_loss = torch.tensor(0.0, device=score.device)
                if target_masks.shape[0] > 0:
                    for i in range(target_masks.shape[0]):
                        mask_i = target_masks[i] > 0.5
                        if mask_i.sum() >= self.instance_variance_min_points:
                            inst_scores = score[mask_i]
                            # unbiased=True by default
                            instance_variance_loss += torch.var(inst_scores)
                
                batch_loss += self.instance_variance_weight * instance_variance_loss

            total_loss += batch_loss
            if self.instance_variance_weight > 0:
                print(f"variance: {self.instance_variance_weight * instance_variance_loss.item()}")
                print(f"batch_loss.item(): {batch_loss.item() - self.instance_variance_weight * instance_variance_loss.item()}")
            total_batches += 1
            
            # ========== STATISTICS ==========
            total_points += num_points
            total_anomaly_points += num_anomaly_points
            
            # Convert scores to probabilities for statistics
            score_probs = torch.sigmoid(score)
            
            if num_anomaly_points > 0:
                avg_score_on_anomaly_points.append(score_probs[ood_mask].mean().item())
            if num_normal_points > 0:
                avg_score_on_normal_points.append(score_probs[id_mask].mean().item())
        
        # Average loss across batches
        if total_batches > 0:
            final_loss = total_loss / total_batches
        else:
            final_loss = total_loss
        
        # Aggregate statistics
        stats = {
            "avg_score_on_anomaly_points": sum(avg_score_on_anomaly_points) / len(avg_score_on_anomaly_points) if len(avg_score_on_anomaly_points) > 0 else 0.0,
            "avg_score_on_normal_points": sum(avg_score_on_normal_points) / len(avg_score_on_normal_points) if len(avg_score_on_normal_points) > 0 else 0.0,
            "total_anomaly_points": total_anomaly_points,
            "total_points": total_points,
        }
        
        weight_status = "WITH distance weighting" if use_distance_weighting else "WITHOUT distance weighting"
        print(f"Point-level Anomaly Loss (BCE {weight_status}): {final_loss.item():.4f} | "
            f"Anomaly Points: {total_anomaly_points} / {total_points} | "
            f"Avg Score on Anomaly Points: {stats['avg_score_on_anomaly_points']:.4f} | "
            f"Avg Score on Normal Points: {stats['avg_score_on_normal_points']:.4f}")
        
        return {"loss_contrastive": final_loss}    

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.ce_class_weights, ignore_index=255
            # src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=255
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices):
        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id]["masks"][target_id].float()
            num_masks = target_mask.shape[0]

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

    def loss_bboxs(self, outputs, targets, indices):
        loss_box = torch.zeros(1, device=outputs["pred_bboxs"].device)
        for batch_id, (map_id, target_id) in enumerate(indices):
            pred_bboxs = outputs["pred_bboxs"][batch_id, map_id, :]
            target_bboxs = targets[batch_id]["bboxs"][target_id]
            target_classes = targets[batch_id]["labels"][target_id]
            #keep_things = target_classes < 8
            keep_things = target_classes >= 11
            keep_things = keep_things <= 18
            if torch.any(keep_things):
                target_bboxs = target_bboxs[keep_things]
                pred_bboxs = pred_bboxs[keep_things]
                num_bboxs = target_bboxs.shape[0]
                loss_box += box_loss_jit(pred_bboxs, target_bboxs, num_bboxs)
        return {
            "loss_box": loss_box,
        }

    def loss_pebal(self, outputs, targets):
        """
        Compute PEBAL loss (Gambler + Energy).
        
        Args:
            outputs: dict with 'pred_logits' [B, Q, C+1] and 'pred_masks' [B, Q, N]
            targets: list of dicts with 'labels' and 'masks'
            
        Returns:
            dict with 'loss_pebal_gambler' and 'loss_pebal_energy'
        """
        # Extract predictions
        pred_masks_list = outputs["pred_masks"]  # [B, Q, num_points]
        pred_logits_list = outputs["pred_logits"]  # [B, Q, num_classes+1]
        
        device = pred_logits_list[0].device
        
        # Lazily instantiate Gambler if needed
        if self.gambler is None:
            self.gambler = Gambler(reward=self.pebal_reward, device=device, pretrain=-1, ood_reg=0.1, ood_ind=self.anomaly_class_id)
        elif self.gambler.device != device:
            self.gambler.device = device
            self.gambler.reward = self.gambler.reward.to(device)
        
        total_gambler_loss = torch.tensor(0.0, device=device)
        total_energy_loss = torch.tensor(0.0, device=device)
        total_batches = 0
        
        # Statistics
        total_points = 0
        total_normal_points = 0
        total_anomaly_points = 0
        
        # Process each batch
        for batch_id in range(len(targets)):
            # ========== CREATE TARGET LABELS ==========
            target_masks = targets[batch_id]["masks"]  # [num_masks, num_points]
            target_labels = targets[batch_id]["labels"]  # [num_masks]
            
            # Get predictions for this batch
            mask_logits_batch = pred_masks_list[batch_id]  # [num_points, Q]
            pred_logits_batch = pred_logits_list[batch_id]  # [Q, num_classes+1]
            
            num_points = mask_logits_batch.shape[0]
            
            # Initialize all points as no-object class (num_classes)
            point_targets = torch.full((num_points,), self.num_classes, dtype=torch.long, device=device)
            
            # Assign labels to points covered by masks (keep original labels)
            for mask_idx, (mask, label) in enumerate(zip(target_masks, target_labels)):
                point_targets[mask > 0.5] = label
            
            # ========== COMPUTE PER-POINT LOGITS ==========
            # Compute per-point logits: [num_points, num_classes+1]
            # Each point's logit is a weighted combination of query logits
            mask_probs = mask_logits_batch.float().sigmoid()  # [num_points, Q]
            point_logits = mask_probs.matmul(pred_logits_batch)  # [num_points, num_classes+1]
            
            # ========== PREPARE FOR PEBAL LOSS ==========
            # Reshape for PEBAL: [1, C+1, 1, num_points] format
            point_logits_2d = point_logits.T.unsqueeze(0).unsqueeze(-2)  # [1, C+1, 1, num_points]
            point_targets_2d = point_targets.unsqueeze(0).unsqueeze(1)  # [1, 1, num_points]
            
            print(point_targets_2d.unique())
            print(self.anomaly_class_id)
            # Check if there are OOD samples (anomaly class is 19)
            has_ood = (point_targets == self.anomaly_class_id).any().item()
            
            # ========== COMPUTE LOSSES ==========
            # Compute Gambler loss
            gambler_loss = self.gambler(point_logits_2d, point_targets_2d, wrong_sample=has_ood)
            
            # Compute Energy loss (using original labels: 19 for anomaly, num_classes for void/no-object)
            energy_loss_val, _ = energy_loss(point_logits_2d, point_targets_2d, 
                                            num_class=self.num_classes, 
                                            ood_ind=self.anomaly_class_id, void_ind=self.num_classes)
            
            total_gambler_loss += gambler_loss
            total_energy_loss += energy_loss_val
            total_batches += 1
            
            # ========== STATISTICS ==========
            total_points += num_points
            total_anomaly_points += (point_targets == self.anomaly_class_id).sum().item()
            total_normal_points += ((point_targets < self.num_classes) & (point_targets != self.anomaly_class_id)).sum().item()
        
        # Average across batches
        if total_batches > 0:
            avg_gambler_loss = total_gambler_loss / total_batches
            avg_energy_loss = total_energy_loss / total_batches
        else:
            avg_gambler_loss = total_gambler_loss
            avg_energy_loss = total_energy_loss
        
        print(f"PEBAL Losses | "
              f"Gambler: {avg_gambler_loss.item():.4f} | "
              f"Energy: {avg_energy_loss.item():.4f} | "
              f"Points: {total_points} (Normal: {total_normal_points}, Anomaly: {total_anomaly_points})")
        
        return {
            "loss_pebal_gambler": avg_gambler_loss,
            "loss_pebal_energy": avg_energy_loss
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    

    def get_loss(self, loss, outputs, targets, indices, raw_coordinates=None):
        loss_map = {
            "labels": self.loss_labels, 
            "masks": self.loss_masks, 
            "bboxs": self.loss_bboxs, 
            "contrastive": self.loss_contrastive,
            "pebal": self.loss_pebal,
        }
        if loss == "contrastive":
            return loss_map[loss](outputs, targets, indices, raw_coordinates=raw_coordinates)
        elif loss == "pebal":
            return loss_map[loss](outputs, targets)
        else:
            return loss_map[loss](outputs, targets, indices)

    def forward(self, outputs, targets, raw_coordinates=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Split targets into normal and anomaly
        anomaly_class_id = self.anomaly_class_id
        normal_targets = []
        anomaly_targets = []
        
        for target in targets:
            labels = target["labels"]

            is_normal = labels != anomaly_class_id
            is_anomaly = labels == anomaly_class_id
            
            normal_target = {}
            anomaly_target = {}

            for key, value in target.items():
                if key == "labels" or key == "masks" or key == "bboxs":
                    normal_target[key] = value[is_normal]
                    anomaly_target[key] = value[is_anomaly]
                else:
                    # Other fields pass through to both
                    normal_target[key] = value
                    anomaly_target[key] = value
            
            normal_targets.append(normal_target)
            anomaly_targets.append(anomaly_target)

        # Retrieve the matching between the outputs of the last layer and the normal targets
        indices = self.matcher(outputs_without_aux, normal_targets)

        # Compute losses on normal masks
        losses = {}
        for loss in self.losses:
            if loss == 'contrastive':
                # Compute contrastive loss on anomaly masks
                losses.update(self.get_loss(loss, outputs, anomaly_targets, None, raw_coordinates=raw_coordinates))
            elif loss == 'pebal':
                # Compute PEBAL loss (uses all targets including anomalies)
                losses.update(self.get_loss(loss, outputs, targets, None))
            else:
                # Regular losses on normal targets
                losses.update(self.get_loss(loss, outputs, normal_targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, normal_targets)
                for loss in self.losses:
                    # Skip anomaly and contrastive loss for auxiliary outputs
                    if loss not in ['anomaly', 'contrastive']:
                        l_dict = self.get_loss(loss, aux_outputs, normal_targets, indices)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
        return losses
