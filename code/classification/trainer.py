import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


class TStageClassificationLoss(nn.Module):
    def __init__(self, class_weights=None, loss_type="ce_focal", focal_gamma=2.0, focal_alpha=None):
        super().__init__()
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(Config.device)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        if focal_alpha is None:
            if self.class_weights is not None:
                alpha = self.class_weights.detach().cpu().numpy().astype(np.float32)
                alpha = alpha / (alpha.sum() + 1e-8)
                self.focal_alpha = torch.tensor(alpha, dtype=torch.float32, device=Config.device)
            else:
                self.focal_alpha = None
        else:
            self.focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32, device=Config.device)

        print(f"Loss type: {self.loss_type}")
        if self.class_weights is not None:
            print(f"Class weights: {self.class_weights}")
        if self.focal_alpha is not None:
            print(f"Focal alpha: {self.focal_alpha}")

    def focal_loss(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce)
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha[targets]
            focal = alpha_t * ((1 - pt) ** self.focal_gamma) * ce
        else:
            focal = ((1 - pt) ** self.focal_gamma) * ce
        return focal.mean()

    def forward(self, logits, targets):
        if self.loss_type == "ce":
            return self.ce_loss(logits, targets)
        elif self.loss_type == "focal":
            return self.focal_loss(logits, targets)
        elif self.loss_type == "ce_focal":
            ce = self.ce_loss(logits, targets)
            focal = self.focal_loss(logits, targets)
            return 0.5 * ce + 0.5 * focal
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


# =========================================================
# 2. Early stopping 
# =========================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode="max", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1
        self.early_stop = False

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        elif self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1
        self.early_stop = False


# =========================================================
# 3. Trainer
# =========================================================
class ClassificationTrainer:
    def __init__(self, model, model_name=None):
        if model_name is None:
            model_name = Config.model_name

        self.model = model.to(Config.device)
        self.model_name = model_name

        # experiment directory
        exp_name = f"{Config.get_experiment_name()}_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path(Config.experiments_dir) / exp_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # model-specific config
        model_cfg = Config.get_model_config(model_name)
        self.learning_rate = model_cfg.get("learning_rate", Config.learning_rate)
        self.batch_size = model_cfg.get("batch_size", Config.batch_size)

        # loss config
        self.loss_type = getattr(Config, "loss_type", "ce_focal")
        self.focal_gamma = getattr(Config, "focal_gamma", 2.0)
        self.criterion = TStageClassificationLoss(
            class_weights=None,         
            loss_type=self.loss_type,    
            focal_gamma=self.focal_gamma,
            focal_alpha=[0.6, 0.4],       
        )

        # optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=Config.weight_decay,
            betas=(0.9, 0.999)
        )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )

        # ★ 修改：从 Config 读取监控指标，默认 'auc'
        self.monitor_metric = getattr(Config, "monitor_metric", "auc")
        print(f"Monitor metric: {self.monitor_metric}")
        self.early_stopping = EarlyStopping(
            patience=Config.early_stopping_patience,
            min_delta=Config.early_stopping_min_delta,
            mode="max",
            verbose=True
        )

        # amp
        self.use_amp = Config.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # history (增加 youden)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": [],
            "train_f1": [],
            "val_f1": [],
            "train_specificity": [],
            "val_specificity": [],
            "train_auc": [],
            "val_auc": [],
            "train_youden": [],
            "val_youden": [],
            "learning_rates": [],
            "epoch_times": [],
        }

        # best model tracking
        self.best_score = -np.inf
        self.best_epoch = -1
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.final_model_path = self.checkpoint_dir / "final_model.pth"
        self.last_model_path = self.checkpoint_dir / "last_model.pth"

        # 最佳阈值（由验证集约登指数确定，不受监控指标影响）
        self.best_threshold = 0.5

        self.start_time = None
        self.total_training_time = 0

        print("\n" + "=" * 70)
        print("ClassificationTrainer initialized")
        print("=" * 70)
        print(f"Model name        : {self.model_name}")
        print(f"Experiment dir    : {self.experiment_dir}")
        print(f"Device            : {Config.device}")
        print(f"Input mode        : {Config.input_mode}")
        print(f"Label mode        : {Config.label_mode}")
        print(f"Class names       : {Config.get_class_names()}")
        print(f"Num classes       : {Config.get_num_classes()}")
        print(f"Loss type         : {self.loss_type}")
        print(f"Learning rate     : {self.learning_rate}")
        print(f"Batch size        : {self.batch_size}")
        print(f"Use AMP           : {self.use_amp}")
        print("=" * 70)

    # =========================================================
    # 4. Metrics
    # =========================================================
    def compute_specificity(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            return 0.0
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def compute_youden(self, sensitivity, specificity):
        if sensitivity is not None and specificity is not None:
            return sensitivity + specificity - 1
        return None

    def find_best_threshold(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_arr = tpr - fpr
        best_idx = np.argmax(youden_arr)
        best_threshold = float(thresholds[best_idx])
        best_youden = float(youden_arr[best_idx])
        return best_threshold, best_youden

    def compute_metrics(self, preds, targets, probs=None, threshold=None):
        preds_np = np.asarray(preds).squeeze()
        targets_np = np.asarray(targets).squeeze()

        if preds_np.size == 0 or targets_np.size == 0:
            print("Warning: Empty predictions or targets")
            return {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "specificity": 0.0, "youden": None,
                "roc_auc": None, "confusion_matrix": np.zeros((2,2), dtype=int),
                "classification_report": {}
            }

        if probs is not None and threshold is not None:
            probs_np = np.asarray(probs)
            if probs_np.ndim == 2 and probs_np.shape[1] == 2:
                prob_class1 = probs_np[:, 1]
                preds_class = (prob_class1 >= threshold).astype(int)
            else:
                preds_class = (probs_np >= threshold).astype(int)
        else:
            if preds_np.ndim > 1 and preds_np.shape[1] > 1:
                preds_class = np.argmax(preds_np, axis=1)
            else:
                preds_class = preds_np

        preds_class = np.asarray(preds_class).astype(int).reshape(-1)
        targets_np = np.asarray(targets_np).astype(int).reshape(-1)

        if preds_class.shape != targets_np.shape:
            min_len = min(len(preds_class), len(targets_np))
            preds_class = preds_class[:min_len]
            targets_np = targets_np[:min_len]

        try:
            accuracy = accuracy_score(targets_np, preds_class)
            precision = precision_score(targets_np, preds_class, average="binary", zero_division=0)
            recall = recall_score(targets_np, preds_class, average="binary", zero_division=0)
            f1 = f1_score(targets_np, preds_class, average="binary", zero_division=0)
            specificity = self.compute_specificity(targets_np, preds_class)
            youden = self.compute_youden(recall, specificity)

            cm = confusion_matrix(targets_np, preds_class, labels=[0, 1])
            report = classification_report(
                targets_np, preds_class, labels=[0, 1],
                target_names=Config.get_class_names(), output_dict=True, zero_division=0
            )

            roc_auc = None
            if probs is not None:
                probs_arr = np.asarray(probs)
                if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
                    if len(np.unique(targets_np)) == 2:
                        try:
                            roc_auc = roc_auc_score(targets_np, probs_arr[:, 1])
                        except Exception:
                            roc_auc = None
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            accuracy, precision, recall, f1, specificity, youden = 0.0, 0.0, 0.0, 0.0, 0.0, None
            roc_auc = None
            cm = np.zeros((2, 2), dtype=int)
            report = {}

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "youden": youden,
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_auc": roc_auc
        }

    # =========================================================
    # 5. One epoch train / val (增加预测分布打印)
    # =========================================================
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels, case_ids) in enumerate(pbar):
            images = images.to(Config.device, non_blocking=True)
            labels = labels.to(Config.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_targets.extend(labels.detach().cpu().numpy().tolist())
                all_probs.extend(probs.detach().cpu().numpy().tolist())

            epoch_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        epoch_loss /= max(len(train_loader), 1)
        metrics = self.compute_metrics(all_preds, all_targets, all_probs)
        return epoch_loss, metrics

    def validate_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_targets, all_probs, all_case_ids = [], [], [], []

        with torch.no_grad():
            for images, labels, case_ids in val_loader:
                images = images.to(Config.device, non_blocking=True)
                labels = labels.to(Config.device, non_blocking=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_targets.extend(labels.detach().cpu().numpy().tolist())
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_case_ids.extend(list(case_ids))
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        metrics = self.compute_metrics(all_preds, all_targets, all_probs)
        print(f"  Predicted class distribution: {np.bincount(all_preds)}")
        return val_loss, metrics, np.array(all_probs), all_case_ids, np.array(all_targets), np.array(all_preds)

    # =========================================================
    # 6. Save checkpoint
    # =========================================================
    def save_checkpoint(self, epoch, score, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "history": self.history,
            "config": {
                "task_name": Config.task_name,
                "input_mode": Config.input_mode,
                "label_mode": Config.label_mode,
                "class_names": Config.get_class_names(),
                "num_classes": Config.get_num_classes(),
                "input_size_3d": Config.input_size_3d,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "loss_type": self.loss_type,
            }
        }
        torch.save(checkpoint, self.last_model_path)
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"✅ Best model saved to: {self.best_model_path}")

    # =========================================================
    # 7. Plots and history
    # =========================================================
    def save_plots(self):
        if len(self.history["train_loss"]) == 0:
            print("No training history to plot")
            return
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))

        axes[0,0].plot(epochs, self.history["train_loss"], "b-", label="Train Loss")
        axes[0,0].plot(epochs, self.history["val_loss"], "r-", label="Val Loss")
        axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(epochs, self.history["train_acc"], "b-", label="Train Acc")
        axes[0,1].plot(epochs, self.history["val_acc"], "r-", label="Val Acc")
        axes[0,1].set_title("Accuracy"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

        axes[0,2].plot(epochs, self.history["train_f1"], "b-", label="Train F1")
        axes[0,2].plot(epochs, self.history["val_f1"], "r-", label="Val F1")
        axes[0,2].set_title("F1"); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

        axes[0,3].plot(epochs, self.history["train_auc"], "b-", label="Train AUC")
        axes[0,3].plot(epochs, self.history["val_auc"], "r-", label="Val AUC")
        axes[0,3].set_title("AUC"); axes[0,3].legend(); axes[0,3].grid(True, alpha=0.3)

        axes[1,0].plot(epochs, self.history["train_youden"], "b-", label="Train Youden")
        axes[1,0].plot(epochs, self.history["val_youden"], "r-", label="Val Youden")
        axes[1,0].axhline(y=0, color='gray', linestyle='--')
        axes[1,0].set_title("Youden Index"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(epochs, self.history["train_specificity"], "b-", label="Train Spec")
        axes[1,1].plot(epochs, self.history["val_specificity"], "r-", label="Val Spec")
        axes[1,1].set_title("Specificity"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

        axes[1,2].plot(epochs, self.history["learning_rates"], "g-")
        axes[1,2].set_title("Learning Rate"); axes[1,2].set_yscale("log"); axes[1,2].grid(True, alpha=0.3)

        axes[1,3].bar(list(epochs), self.history["epoch_times"], color="purple", alpha=0.7)
        axes[1,3].set_title("Epoch Time"); axes[1,3].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = self.plots_dir / f"training_curves_{Config.input_mode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to: {plot_path}")

    def save_history(self):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
            if isinstance(obj, Path): return str(obj)
            return obj

        history_data = {
            "model_name": self.model_name,
            "task_name": Config.task_name,
            "input_mode": Config.input_mode,
            "label_mode": Config.label_mode,
            "class_names": Config.get_class_names(),
            "num_classes": Config.get_num_classes(),
            "history": {k: [convert_to_serializable(x) for x in v] for k, v in self.history.items()},
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "training_time_seconds": self.total_training_time,
            "early_stopping_triggered": self.early_stopping.early_stop,
            "config": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": Config.num_epochs,
                "input_size_3d": Config.input_size_3d,
                "class_names": Config.get_class_names(),
                "input_mode": Config.input_mode,
                "label_mode": Config.label_mode,
                "class_weights": Config.class_weights.tolist() if Config.class_weights is not None else None,
                "loss_type": self.loss_type,
            }
        }
        history_path = self.results_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Training history saved to: {history_path}")
        return history_path

    # =========================================================
    # 8. Save case-level results
    # =========================================================
    def save_case_level_results(self, case_ids, targets, preds, probs, save_name="case_level_results.csv"):
        rows = []
        for i in range(len(case_ids)):
            row = {
                "case_id": case_ids[i],
                "y_true": int(targets[i]),
                "y_pred": int(preds[i]),
            }
            if probs is not None and len(probs) > i:
                row["prob_class0"] = float(probs[i][0])
                row["prob_class1"] = float(probs[i][1])
            rows.append(row)
        df = pd.DataFrame(rows)
        save_path = self.results_dir / save_name
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Case-level results saved to: {save_path}")
        return save_path

    # =========================================================
    # 9. Train
    # =========================================================
    def fit(self, train_loader, val_loader, start_epoch=1):          
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print(f"Start Training (monitor: {self.monitor_metric})")
        print("=" * 70)


        for epoch in range(start_epoch, Config.num_epochs + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch [{epoch}/{Config.num_epochs}] | LR: {current_lr:.6e}")

            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics, val_probs, val_case_ids, val_targets, val_preds = self.validate_epoch(val_loader)

            self.scheduler.step(epoch)                
   
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["train_acc"].append(float(train_metrics["accuracy"]))
            self.history["val_acc"].append(float(val_metrics["accuracy"]))
            self.history["train_precision"].append(float(train_metrics["precision"]))
            self.history["val_precision"].append(float(val_metrics["precision"]))
            self.history["train_recall"].append(float(train_metrics["recall"]))
            self.history["val_recall"].append(float(val_metrics["recall"]))
            self.history["train_f1"].append(float(train_metrics["f1"]))
            self.history["val_f1"].append(float(val_metrics["f1"]))
            self.history["train_specificity"].append(float(train_metrics["specificity"]))
            self.history["val_specificity"].append(float(val_metrics["specificity"]))
            self.history["train_auc"].append(float(train_metrics["roc_auc"]) if train_metrics["roc_auc"] is not None else 0.0)
            self.history["val_auc"].append(float(val_metrics["roc_auc"]) if val_metrics["roc_auc"] is not None else 0.0)
            self.history["train_youden"].append(float(train_metrics["youden"]) if train_metrics["youden"] is not None else 0.0)
            self.history["val_youden"].append(float(val_metrics["youden"]) if val_metrics["youden"] is not None else 0.0)
            self.history["learning_rates"].append(float(current_lr))
            self.history["epoch_times"].append(float(time.time() - epoch_start))

            print(f"Train Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, "
                f"F1={train_metrics['f1']:.4f}, AUC={train_metrics['roc_auc']}, "
                f"Youden={train_metrics['youden']:.4f}")
            print(f"Val   Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                f"F1={val_metrics['f1']:.4f}, AUC={val_metrics['roc_auc']}, "
                f"Youden={val_metrics['youden']:.4f}")

            if self.monitor_metric == "auc":
                current_score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
            elif self.monitor_metric == "youden":
                current_score = val_metrics["youden"] if val_metrics["youden"] is not None else 0.0
            elif self.monitor_metric == "f1":
                current_score = val_metrics["f1"]
            elif self.monitor_metric == "accuracy":
                current_score = val_metrics["accuracy"]
            elif self.monitor_metric == "loss":
                current_score = -val_loss
            else:
                raise ValueError(f"Unknown monitor_metric: {self.monitor_metric}")

      
            is_best = current_score > self.best_score
            if is_best:
                self.best_score = current_score
                self.best_epoch = epoch               
         
                if val_probs is not None and len(val_probs) > 0:
                    probs_arr = np.asarray(val_probs)
                    if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
                        val_prob_class1 = probs_arr[:, 1]
                        best_thresh, best_youden = self.find_best_threshold(val_targets, val_prob_class1)
                        self.best_threshold = best_thresh
                        print(f"  -> New best threshold (Youden): {best_youden:.4f} at {best_thresh:.4f}")
                self.save_checkpoint(epoch, current_score, is_best=True)  
                self.save_case_level_results(val_case_ids, val_targets, val_preds, val_probs,
                                            "best_val_case_level_results.csv")

            self.save_checkpoint(epoch, current_score, is_best=False)     

            stop = self.early_stopping(current_score, epoch)
            if stop:
                print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                break

        self.total_training_time = time.time() - self.start_time

        final_checkpoint = {
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "history": self.history,
        }
        torch.save(final_checkpoint, self.final_model_path)

        self.save_plots()
        self.save_history()

        print("\n" + "=" * 70)
        print("Training Finished")
        print("=" * 70)
        print(f"Best epoch      : {self.best_epoch}")
        print(f"Best {self.monitor_metric:6s}: {self.best_score:.6f}")
        print(f"Best threshold  : {self.best_threshold:.4f}")
        print(f"Best model path : {self.best_model_path}")
        print(f"Training time   : {self.total_training_time/3600:.2f} hours")
        print("=" * 70)

        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_threshold": self.best_threshold,
            "best_model_path": str(self.best_model_path),
            "final_model_path": str(self.final_model_path),
            "experiment_dir": str(self.experiment_dir),
        }

    # =========================================================
    # 10. Evaluate test
    # =========================================================
    def evaluate_test(self, test_loader, use_best=True, test_name=None):
        if use_best and self.best_model_path.exists():
            print(f"Loading best model from: {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=Config.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_threshold = checkpoint.get("best_threshold", 0.5)
        elif self.final_model_path.exists():
            print(f"Loading final model from: {self.final_model_path}")
            checkpoint = torch.load(self.final_model_path, map_location=Config.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_threshold = checkpoint.get("best_threshold", 0.5)

        self.model.eval()
        test_loss = 0.0
        all_preds, all_targets, all_probs, all_case_ids = [], [], [], []

        with torch.no_grad():
            for images, labels, case_ids in test_loader:
                images = images.to(Config.device, non_blocking=True)
                labels = labels.to(Config.device, non_blocking=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)

                prob_class1 = probs[:, 1].cpu().numpy()
                preds = (prob_class1 >= self.best_threshold).astype(int)

                all_preds.extend(preds.tolist())
                all_targets.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_case_ids.extend(list(case_ids))
                test_loss += loss.item()

        test_loss /= max(len(test_loader), 1)
        test_metrics = self.compute_metrics(all_preds, all_targets, all_probs, threshold=self.best_threshold)

    
        save_name = f"{test_name}_test_case_level_results.csv" if test_name else "test_case_level_results.csv"
        save_path = self.save_case_level_results(
            case_ids=all_case_ids, targets=all_targets, preds=all_preds, probs=all_probs,
            save_name=save_name
    )

        test_summary = {
            "loss": float(test_loss),
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "specificity": float(test_metrics["specificity"]),
            "youden": float(test_metrics["youden"]) if test_metrics["youden"] is not None else None,
            "roc_auc": float(test_metrics["roc_auc"]) if test_metrics["roc_auc"] is not None else None,
            "threshold_used": self.best_threshold,
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "classification_report": test_metrics["classification_report"],
            "case_level_results_csv": str(save_path),
        }

        summary_path = self.results_dir / f"test_summary_{test_name}.json" if test_name else self.results_dir / "test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)

        print("\n" + "=" * 70)
        print(f"Test Evaluation ({test_name if test_name else 'combined'})")
        print("=" * 70)
        print(f"Threshold    : {self.best_threshold:.4f}")
        print(f"Loss         : {test_loss:.4f}")
        print(f"Accuracy     : {test_metrics['accuracy']:.4f}")
        print(f"Precision    : {test_metrics['precision']:.4f}")
        print(f"Recall       : {test_metrics['recall']:.4f}")
        print(f"F1           : {test_metrics['f1']:.4f}")
        print(f"Specificity  : {test_metrics['specificity']:.4f}")
        print(f"Youden       : {test_metrics['youden'] if test_metrics['youden'] is not None else 'NA'}")
        print(f"AUC          : {test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else 'NA'}")
        print(f"Summary path : {summary_path}")
        print("=" * 70)

        return test_summary