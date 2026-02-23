#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


class AnomalyEval:
    """
    Anomaly evaluation for binary anomaly detection.
    
    Evaluates predictions from a 2-class anomaly head:
    - Class 0: Inlier/Normal
    - Class 1: Anomaly
    
    Metrics computed:
    - AUROC (Area Under ROC Curve)
    - AUPR (Area Under Precision-Recall Curve)
    - FPR@95 (False Positive Rate at 95% True Positive Rate)
    - Precision, Recall, F1 at different thresholds
    - Confusion Matrix (TP, FP, TN, FN)

    AUPR is used as the primary metric for model selection.
    """

    def __init__(self, anomaly_class_id=19, threshold=0.5, max_points=100_000_000, seed=0):
        """
        Args:
            anomaly_class_id: The class ID that represents anomalies in ground truth (default: 19)
            threshold: Default threshold for binary classification (default: 0.5)
            max_points: Maximum number of points kept for ranking-based metrics (None keeps all).
            seed: RNG seed for sampling when max_points is set.
        """
        self.anomaly_class_id = anomaly_class_id
        self.threshold = threshold
        self.max_points = max_points
        self.rng = np.random.default_rng(seed)
        self.eps = 1e-15
        
        # Storage for predictions and ground truth
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and ground truth."""
        self.anomaly_scores = []  # Used only when max_points is None
        self.ground_truth = []     # Used only when max_points is None
        self._sample_scores = np.empty((0,), dtype=np.float32)
        self._sample_gt = np.empty((0,), dtype=np.int8)
        self._sample_keys = np.empty((0,), dtype=np.float32)
        self._cached_scores = None
        self._cached_gt = None
        self._latest_metrics = None
        
        # Confusion matrix at default threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def addBatch(self, anomaly_logits, semantic_labels):
        """
        Add a batch of predictions and ground truth.
        
        Args:
            anomaly_logits: [N, 2] array of logits [inlier_logit, anomaly_logit] or 
                           [N] array of anomaly scores (probabilities)
            semantic_labels: [N] array of semantic class labels
        """
        # Convert logits to anomaly scores (probability of being anomaly)
        if len(anomaly_logits.shape) == 2:
            # If 2D [N, 2], apply softmax and take anomaly probability
            exp_logits = np.exp(anomaly_logits - np.max(anomaly_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            anomaly_score = probs[:, 1]  # Probability of anomaly (second class)
        else:
            # If 1D [N], assume it's already anomaly scores
            anomaly_score = anomaly_logits
        
        # Create binary ground truth: 1 if semantic label is anomaly class, 0 otherwise
        gt_binary = (semantic_labels == self.anomaly_class_id).astype(np.int64)
        
        # Store for later metric computation (bounded if max_points is set)
        if self.max_points is None:
            self.anomaly_scores.append(anomaly_score)
            self.ground_truth.append(gt_binary)
        else:
            scores_new = np.asarray(anomaly_score, dtype=np.float32).reshape(-1)
            gt_new = np.asarray(gt_binary, dtype=np.int8).reshape(-1)
            keys_new = self.rng.random(scores_new.shape[0], dtype=np.float32)

            if self._sample_scores.size == 0:
                scores_cat = scores_new
                gt_cat = gt_new
                keys_cat = keys_new
            else:
                scores_cat = np.concatenate([self._sample_scores, scores_new])
                gt_cat = np.concatenate([self._sample_gt, gt_new])
                keys_cat = np.concatenate([self._sample_keys, keys_new])

            if scores_cat.size > self.max_points:
                keep_idx = np.argpartition(keys_cat, -self.max_points)[-self.max_points :]
                self._sample_scores = scores_cat[keep_idx]
                self._sample_gt = gt_cat[keep_idx]
                self._sample_keys = keys_cat[keep_idx]
            else:
                self._sample_scores = scores_cat
                self._sample_gt = gt_cat
                self._sample_keys = keys_cat

        self._cached_scores = None
        self._cached_gt = None
        
        # Update confusion matrix at default threshold
        predictions = (anomaly_score >= self.threshold).astype(np.int64)
        self.tp += np.sum((predictions == 1) & (gt_binary == 1))
        self.fp += np.sum((predictions == 1) & (gt_binary == 0))
        self.tn += np.sum((predictions == 0) & (gt_binary == 0))
        self.fn += np.sum((predictions == 0) & (gt_binary == 1))
        self._latest_metrics = None
    
    def getAUPR(self):
        """Compute Area Under Precision-Recall Curve."""
        scores, gt = self._get_cached_arrays()
        if scores is None or scores.size == 0:
            return 0.0
        
        # Check if we have both classes
        if len(np.unique(gt)) < 2:
            print("Warning: Only one class present in ground truth. AUPR undefined.")
            return 0.0
        
        return average_precision_score(gt, scores)
    
    def getAUROC(self):
        """Compute Area Under ROC Curve."""
        scores, gt = self._get_cached_arrays()
        if scores is None or scores.size == 0:
            return 0.0
            
        if len(np.unique(gt)) < 2:
            return 0.0
            
        return roc_auc_score(gt, scores)

    def getFPR95(self):
        """Compute False Positive Rate at 95% True Positive Rate."""
        scores, gt = self._get_cached_arrays()
        if scores is None or scores.size == 0:
            return 0.0
            
        if len(np.unique(gt)) < 2:
            return 0.0
            
        fpr, tpr, thresholds = roc_curve(gt, scores)
        
        # Find threshold where TPR is at least 0.95
        # tpr is increasing, so we look for the first index where tpr >= 0.95
        # But roc_curve output is sorted by threshold descending, so TPR is increasing?
        # Scikit-learn roc_curve: thresholds are decreasing. TPR starts at 0 and goes to 1.
        
        target_tpr = 0.95
        idx = np.searchsorted(tpr, target_tpr)
        if idx >= len(fpr):
            idx = len(fpr) - 1
            
        return fpr[idx]

    def getAllMetrics(self):
        """
        Compute and return all metrics.
        
        Returns:
            Dictionary with all computed metrics
        """
        scores, gt = self._get_cached_arrays()
        
        # Default metrics
        metrics = {
            'AUPR': 0.0,
            'AUROC': 0.0,
            'FPR@95': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Accuracy': 0.0,
            'TP': self.tp,
            'FP': self.fp,
            'TN': self.tn,
            'FN': self.fn
        }
        
        if scores is None or scores.size == 0:
            self._latest_metrics = metrics
            return metrics
            
        # Compute main metrics
        metrics['AUPR'] = self.getAUPR()
        metrics['AUROC'] = self.getAUROC()
        metrics['FPR@95'] = self.getFPR95()
        
        # Confusion-based metrics (at threshold)
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.eps)
        
        metrics['Precision'] = precision
        metrics['Recall'] = recall
        metrics['F1'] = f1
        metrics['Accuracy'] = accuracy
        
        self._latest_metrics = metrics
        return self._latest_metrics
    
    def printMetrics(self):
        """Print all metrics in a formatted way."""
        if self._latest_metrics is None:
            metrics = self.getAllMetrics()
        else:
            metrics = self._latest_metrics
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION METRICS")
        print("="*60)
        print(f"AUROC (↑):              {metrics['AUROC']:.4f}")
        print(f"AUPR (↑):               {metrics['AUPR']:.4f}")
        print(f"FPR@95 (↓):             {metrics['FPR@95']:.4f}")
        print("-" * 60)
        print(f"Precision:              {metrics['Precision']:.4f} (at thr={self.threshold})")
        print(f"Recall:                 {metrics['Recall']:.4f} (at thr={self.threshold})")
        print(f"F1-Score:               {metrics['F1']:.4f} (at thr={self.threshold})")
        print(f"Accuracy:               {metrics['Accuracy']:.4f} (at thr={self.threshold})")
        print("-" * 60)
        print(f"TP: {metrics['TP']}, FP: {metrics['FP']}")
        print(f"FN: {metrics['FN']}, TN: {metrics['TN']}")
        print("="*60 + "\n")

    def _get_cached_arrays(self):
        """Return concatenated scores/ground truth, caching the result."""
        if self.max_points is None:
            if len(self.anomaly_scores) == 0:
                return None, None
            if self._cached_scores is None or self._cached_gt is None:
                self._cached_scores = np.concatenate(self.anomaly_scores)
                self._cached_gt = np.concatenate(self.ground_truth)
            return self._cached_scores, self._cached_gt

        if self._sample_scores.size == 0:
            return None, None
        return self._sample_scores, self._sample_gt