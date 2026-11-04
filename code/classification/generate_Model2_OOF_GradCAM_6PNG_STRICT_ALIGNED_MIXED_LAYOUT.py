# -*- coding: utf-8 -*-
"""FINAL OOF Model 2 / audited 3D Grad-CAM / 2 slices x 6 PNGs.

Each patient:
  - highest nonempty GT/Pred 2D Dice slice
  - largest GT tumor area slice
Each slice: T2WI, GT-filled, Pred-filled, full CAM-only, semi-transparent
CAM overlay, semi-transparent CAM with GT+Pred contours.

Scientific constraints: frozen selected OOF classifier, exact independent-test
T2WI + original *_pred.nii.gz inputs, unchanged NIfTI voxel orientation,
original selected CAM target backbone.class_layers.relu, per-input spatial audit,
per-patient probability reproduction audit, and fixed image footprint in every
output PNG. No rotation, flip, crop, translated overlay or extra resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# -------------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import Config
from data_loader import RectalMRIClassification3DDataset, load_samples_from_test_csv
from models import create_model

# -------------------------------------------------------------------------
# Fixed paths  (anonymized)
# -------------------------------------------------------------------------
MODEL_ARCH = "densenet121_3d"
EXPECTED_N = {"IT1": 47, "IT2": 31, "ET": 83}
SEED = 42

OOF_MODEL2_SELECTION_JSON = Path(
    r"RECTAL_STAS_DATA\DenseNet121_Models1_2_3_Shared5Fold_OOF_v1"
    r"\Model2\selected_model\TRAINING_COMPLETE_selected_model.json"
)

IT1_CASES_DIR = Path(
    r"RECTAL_STAS_DATA\split_dataset_tstage_fixedtest10\test\cases"
)
IT1_FIXED_LABELS = Path(
    r"RECTAL_STAS_DATA\test_cohort_A\fixed_test_labels.csv"
)

IT2_CASES_DIR = Path(
    r"RECTAL_STAS_DATA\test_cohort_B\cases"
)
IT2_LABEL_XLSX = Path(
    r"RECTAL_STAS_DATA\test_cohort_B\labels.xlsx"
)

ET_CASES_DIR = Path(
    r"RECTAL_STAS_DATA\test_cohort_C\cases"
)
ET_LABEL_XLSX = Path(
    r"RECTAL_STAS_DATA\test_cohort_C\labels.xlsx"
)

FINAL_ANALYSIS_ROOT = Path(
    r"RECTAL_STAS_DATA\DenseNet121_Models1_2_3_Shared5Fold_OOF_v1"
    r"\paper_style_test_results_IT1_IT2_ET_Models1_2_3"
)
FROZEN_PRED_ROOT = FINAL_ANALYSIS_ROOT  # fallback directory; not used if VALIDATED_PRED_CSV exists

VALIDATED_PRED_CSV = {
    "IT1": FINAL_ANALYSIS_ROOT / "IT1" / "predictions" / "IT1_Model_2_predictions.csv",
    "IT2": FINAL_ANALYSIS_ROOT / "IT2" / "predictions" / "IT2_Model_2_predictions.csv",
    "ET":  FINAL_ANALYSIS_ROOT / "ET"  / "predictions" / "ET_Model_2_predictions.csv",
}

# IMPORTANT: Keep Windows output paths short. The previous long root together
# with descriptive filenames reached 260 characters and failed in Pillow/Matplotlib.
# This version uses an independent output directory; existing files stay intact.
OUTPUT_ROOT = Path(r"RECTAL_STAS_DATA\Model2_CAM_6PNG_AUDITED_v1016")

# -------------------------------------------------------------------------
# ROI contour colors (retain the user's original actual color choices)
# GT mask   = vivid magenta
# Pred mask = vivid blue
# -------------------------------------------------------------------------
GT_COLOR = "#DC07F0"       # magenta
PRED_COLOR = "#0515F8"     # vivid blue
MASK_ALPHA = 0.38
CONTOUR_LINEWIDTH = 1.8
HEATMAP_ALPHA = 0.45

# -------------------------------------------------------------------------
# Standardized PNG export settings
# - all PNGs share the same canvas size and DPI
# - no titles are drawn on figures
# - titles are instead encoded into detailed filenames
# -------------------------------------------------------------------------
PNG_FIGSIZE = (8, 8)
PNG_DPI = 400
PNG_FACE_COLOR = "white"
PNG_EDGE_COLOR = "white"
PNG_TIGHT_PAD = 0.2
# Conservative guard for the default Windows path-length restriction.
# Actual Windows long-path support varies by system and installed libraries.
MAX_SAFE_WINDOWS_PATH_LENGTH = 240


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def require_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def configure_deterministic_torch():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_case_id_4digit(raw_id) -> str:
    if pd.isna(raw_id):
        raise ValueError("Missing test-cohort A case ID")
    s = str(raw_id).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = str(int(float(s)))
    elif re.fullmatch(r"\d+", s):
        s = str(int(s))
    else:
        raise ValueError(f"Unexpected test-cohort A case ID: {raw_id}")
    return s.zfill(4)


def extract_t_stage_number(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float) and np.isfinite(val):
        return int(val)
    m = re.search(r"(\d+)", str(val).strip())
    return int(m.group(1)) if m else None


def map_stage_to_binary(val):
    n = extract_t_stage_number(val)
    if n in (1, 2):
        return 0
    if n in (3, 4):
        return 1
    return None


def load_model2_info() -> dict:
    require_file(OOF_MODEL2_SELECTION_JSON, "OOF Model 2 selection JSON")
    with OOF_MODEL2_SELECTION_JSON.open("r", encoding="utf-8") as f:
        d = json.load(f)

    required = [
        "selected_fold",
        "selected_threshold",
        "selected_best_model_path",
        "selected_validation_auc",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"OOF selection JSON missing keys: {missing}")

    ckpt = Path(d["selected_best_model_path"])
    require_file(ckpt, "OOF Model 2 checkpoint")

    return {
        "fold": int(d["selected_fold"]),
        "threshold": float(d["selected_threshold"]),
        "validation_auc": float(d["selected_validation_auc"]),
        "checkpoint": ckpt,
    }


# -------------------------------------------------------------------------
# Build manifests
# -------------------------------------------------------------------------
def build_it1_manifest() -> pd.DataFrame:
    require_file(IT1_FIXED_LABELS, "IT1 fixed labels")
    df = pd.read_csv(IT1_FIXED_LABELS, encoding="utf-8-sig")
    for c in ["case_id_norm", "binary_label"]:
        if c not in df.columns:
            raise KeyError(f"IT1 fixed labels missing column: {c}")

    case_ids = [normalize_case_id_4digit(x) for x in df["case_id_norm"].tolist()]
    labels = pd.to_numeric(df["binary_label"], errors="raise").astype(int).tolist()

    rows = []
    for cid, label in zip(case_ids, labels):
        case_dir = IT1_CASES_DIR / cid
        rows.append({
            "cohort": "IT1",
            "case_id_folder": cid,
            "label": int(label),
            "case_dir": str(case_dir),
            "image_path": str(case_dir / f"{cid}_image.nii.gz"),
            "gt_mask_path": str(case_dir / f"{cid}_mask.nii.gz"),
            "pred_mask_path": str(case_dir / f"{cid}_pred.nii.gz"),
        })

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_N["IT1"]:
        raise RuntimeError(f"IT1 expected 47 cases, got {len(out)}")
    return out


def build_excel_manifest(cohort: str, label_xlsx: Path, cases_dir: Path) -> pd.DataFrame:
    require_file(label_xlsx, f"{cohort} label XLSX")
    df = pd.read_excel(label_xlsx)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["FolderName", "Histology results _Tstage"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"{cohort} labels missing column: {c}")

    df = df.dropna(subset=["Histology results _Tstage"]).copy()
    df["label"] = df["Histology results _Tstage"].apply(map_stage_to_binary)
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "Histology results _Tstage"].tolist()
        raise ValueError(f"{cohort}: unmappable pathology stages: {bad[:10]}")

    rows = []
    for _, r in df.iterrows():
        cid = str(r["FolderName"]).strip()
        case_dir = cases_dir / cid
        rows.append({
            "cohort": cohort,
            "case_id_folder": cid,
            "label": int(r["label"]),
            "case_dir": str(case_dir),
            "image_path": str(case_dir / f"{cid}_image.nii.gz"),
            "gt_mask_path": str(case_dir / f"{cid}_mask.nii.gz"),
            "pred_mask_path": str(case_dir / f"{cid}_pred.nii.gz"),
        })

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_N[cohort]:
        raise RuntimeError(f"{cohort} expected {EXPECTED_N[cohort]} cases, got {len(out)}")
    return out


def build_manifest(cohort: str) -> pd.DataFrame:
    if cohort == "IT1":
        df = build_it1_manifest()
    elif cohort == "IT2":
        df = build_excel_manifest("IT2", IT2_LABEL_XLSX, IT2_CASES_DIR)
    elif cohort == "ET":
        df = build_excel_manifest("ET", ET_LABEL_XLSX, ET_CASES_DIR)
    else:
        raise ValueError(cohort)

    if df["case_id_folder"].duplicated().any():
        dup = df.loc[df["case_id_folder"].duplicated(keep=False), "case_id_folder"].tolist()
        raise RuntimeError(f"{cohort}: duplicate case IDs: {dup[:10]}")

    for _, r in df.iterrows():
        image_path = Path(r["image_path"])
        gt_mask_path = Path(r["gt_mask_path"])
        pred_mask_path = Path(r["pred_mask_path"])
        require_file(image_path, f"{cohort} T2WI {r['case_id_folder']}")
        require_file(gt_mask_path, f"{cohort} GT mask {r['case_id_folder']}")
        require_file(pred_mask_path, f"{cohort} predicted mask {r['case_id_folder']}")

        if not gt_mask_path.name.endswith("_mask.nii.gz"):
            raise RuntimeError(f"{cohort}/{r['case_id_folder']}: GT mask must be *_mask.nii.gz")
        if not pred_mask_path.name.endswith("_pred.nii.gz"):
            raise RuntimeError(f"{cohort}/{r['case_id_folder']}: predicted mask must be *_pred.nii.gz")
        if pred_mask_path.name.endswith("_pred_oof.nii.gz"):
            raise RuntimeError(f"{cohort}/{r['case_id_folder']}: development OOF mask cannot be used on test set")

    return df


# -------------------------------------------------------------------------
# Reference probabilities
# -------------------------------------------------------------------------
def frozen_prediction_path(cohort: str, fold: int) -> Path:
    if cohort == "IT1":
        return FROZEN_PRED_ROOT / f"densenet121_3d_image_plus_pred_mask_fold{fold}_test_cohort_A.csv"
    if cohort == "IT2":
        return FROZEN_PRED_ROOT / f"densenet121_3d_image_plus_pred_mask_fold{fold}_test_cohort_B.csv"
    if cohort == "ET":
        return FROZEN_PRED_ROOT / f"densenet121_3d_image_plus_pred_mask_fold{fold}_test_cohort_C.csv"
    raise ValueError(cohort)


def load_reference_probabilities(cohort: str, fold: int) -> Optional[Dict[str, float]]:
    candidates = [frozen_prediction_path(cohort, fold), VALIDATED_PRED_CSV[cohort]]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"case_id": str})
        if not {"case_id", "prob_class1"}.issubset(df.columns):
            continue
        df["case_id"] = df["case_id"].astype(str).str.strip()
        if cohort == "IT1":
            df["case_id"] = df["case_id"].map(normalize_case_id_4digit)
        if df["case_id"].duplicated().any():
            raise RuntimeError(f"Duplicate case IDs in reference prediction CSV: {p}")
        print(f"[{cohort}] Probability audit source:\n  {p}")
        return dict(zip(df["case_id"], df["prob_class1"].astype(float)))
    print(f"[{cohort}] WARNING: no validated prediction CSV found; probability equality audit skipped.")
    return None


# -------------------------------------------------------------------------
# NIfTI helpers
# -------------------------------------------------------------------------
def nifti_native_to_dhw(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {arr.shape}")
    depth_axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, depth_axis, 0), depth_axis


def dhw_to_nifti_native(arr_dhw: np.ndarray, depth_axis: int) -> np.ndarray:
    return np.moveaxis(arr_dhw, 0, depth_axis)


def validate_same_geometry(image_nii, mask_nii, tag: str):
    if tuple(image_nii.shape) != tuple(mask_nii.shape):
        raise RuntimeError(f"{tag}: image/mask shape mismatch: {image_nii.shape} vs {mask_nii.shape}")
    if not np.allclose(
        np.asarray(image_nii.affine, dtype=np.float64),
        np.asarray(mask_nii.affine, dtype=np.float64),
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError(f"{tag}: image/mask affine mismatch")


def load_raw_image_and_masks(image_path: Path, gt_mask_path: Path, pred_mask_path: Path):
    img_nii = nib.load(str(image_path))
    gt_nii = nib.load(str(gt_mask_path))
    pred_nii = nib.load(str(pred_mask_path))

    validate_same_geometry(img_nii, gt_nii, f"{image_path.name} vs GT")
    validate_same_geometry(img_nii, pred_nii, f"{image_path.name} vs PRED")

    image_native = img_nii.get_fdata(dtype=np.float32)
    gt_native = gt_nii.get_fdata(dtype=np.float32)
    pred_native = pred_nii.get_fdata(dtype=np.float32)

    image_dhw, depth_axis = nifti_native_to_dhw(image_native)
    gt_dhw, depth_axis_gt = nifti_native_to_dhw(gt_native)
    pred_dhw, depth_axis_pred = nifti_native_to_dhw(pred_native)

    if depth_axis != depth_axis_gt or depth_axis != depth_axis_pred:
        raise RuntimeError("Image/GT/Pred depth-axis interpretation mismatch")

    gt_dhw = (gt_dhw > 0).astype(np.uint8)
    pred_dhw = (pred_dhw > 0).astype(np.uint8)

    return img_nii, image_dhw, gt_dhw, pred_dhw, depth_axis


# -------------------------------------------------------------------------
# Model + target layer
# -------------------------------------------------------------------------
def load_model(model2_info: dict, device: torch.device) -> nn.Module:
    Config.input_mode = "image_plus_pred_mask"
    Config.model_name = MODEL_ARCH
    Config.label_mode = "binary"

    model = create_model(
        model_name=MODEL_ARCH,
        num_classes=2,
        in_channels=Config.get_num_input_channels(),
    )

    try:
        checkpoint = torch.load(
            model2_info["checkpoint"],
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model2_info["checkpoint"],
            map_location=device,
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("model_state_dict missing in checkpoint")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def list_conv3d_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv3d)]


def resolve_target_layer(
    model: nn.Module,
    requested_name: Optional[str] = None,
) -> Tuple[str, nn.Module]:
    """
    Resolve a Grad-CAM target module.

    AUDITED default preference for the FINAL OOF Model 2:
      1) backbone.class_layers.relu
         -> empirically verified 5D post-activation spatial tensor
            [B, C, D, H, W] immediately before AdaptiveAvgPool3d.
      2) backbone.features
         -> numerically identical to the tensor entering class_layers,
            but before class_layers.relu.
      3) backbone.features.norm5
      4) backbone.features.denseblock4
      5) deepest / last Conv3d layer (fallback only)

    The default is intentionally NOT the final denselayer conv2 because
    DenseNet concatenates features across dense layers. The audited ReLU
    target most directly represents the final spatial tensor actually used
    by the classification head before global pooling.
    """
    named = dict(model.named_modules())

    if requested_name:
        if requested_name not in named:
            candidates = [
                n for n in named.keys()
                if (
                    n == "backbone.class_layers.relu"
                    or n == "backbone.features"
                    or n.endswith(".class_layers.relu")
                    or n.endswith(".features")
                    or n.endswith(".features.norm5")
                    or n.endswith(".features.denseblock4")
                    or isinstance(named[n], nn.Conv3d)
                )
            ]
            raise KeyError(
                f"Requested target layer not found: {requested_name}\n"
                f"Relevant available modules (last 50):\n"
                + "\n".join(candidates[-50:])
            )
        return requested_name, named[requested_name]

    preferred = [
        "backbone.class_layers.relu",
        "backbone.features",
        "backbone.features.norm5",
        "backbone.features.denseblock4",
    ]
    for name in preferred:
        if name in named:
            return name, named[name]

    # More general suffix fallback in case the model wrapper name differs.
    for suffix in (
        ".class_layers.relu",
        ".features",
        ".features.norm5",
        ".features.denseblock4",
    ):
        matches = [name for name in named if name.endswith(suffix)]
        if matches:
            return matches[-1], named[matches[-1]]

    convs = list_conv3d_layers(model)
    if not convs:
        raise RuntimeError(
            "Could not find the audited final spatial feature module or any Conv3d layer."
        )
    return convs[-1]

# -------------------------------------------------------------------------
# 3D Grad-CAM
# -------------------------------------------------------------------------
class GradCAM3D:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        if not torch.is_tensor(output):
            raise TypeError(f"Grad-CAM target layer output must be a Tensor, got {type(output)}")
        self.activations = output
        if output.requires_grad:
            def save_gradient(grad):
                self.gradients = grad
            output.register_hook(save_gradient)
        else:
            self.gradients = None

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __call__(self, x: torch.Tensor, target_class: int) -> Tuple[np.ndarray, torch.Tensor]:
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(f"Expected x=[1,C,D,H,W], got {tuple(x.shape)}")

        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

        score = logits[0, int(target_class)]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook did not capture activations/gradients")

        acts = self.activations     # expected [1,C,d,h,w]
        grads = self.gradients      # expected [1,C,d,h,w]

        # Scientific safety checks: the target must still carry 3D spatial
        # information, and the captured gradients must match its shape.
        if acts.ndim != 5:
            raise RuntimeError(
                f"Grad-CAM target output must be 5D [B,C,D,H,W], got {tuple(acts.shape)}"
            )
        if tuple(grads.shape) != tuple(acts.shape):
            raise RuntimeError(
                f"Grad-CAM gradient/activation shape mismatch: "
                f"acts={tuple(acts.shape)}, grads={tuple(grads.shape)}"
            )
        if tuple(acts.shape[2:]) == (1, 1, 1):
            raise RuntimeError(
                "Grad-CAM target is already globally pooled to 1x1x1 and no longer "
                "contains useful spatial localization information."
            )
        if int(torch.count_nonzero(grads).item()) == 0:
            raise RuntimeError(
                "Grad-CAM target gradients are all zero for this case/target class."
            )

        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=tuple(x.shape[2:]),
            mode="trilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if float(cam_max - cam_min) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.detach().cpu().numpy().astype(np.float32), logits.detach()


# -------------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------------
def robust_image_display(image_dhw: np.ndarray) -> Tuple[float, float]:
    finite = image_dhw[np.isfinite(image_dhw)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size > 100 else finite
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    sa = int(a.sum())
    sb = int(b.sum())
    if sa + sb == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / float(sa + sb)


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(a, b).sum())
    return float(inter) / float(union)


def choose_best_dice_slice(image_dhw: np.ndarray, gt_dhw: np.ndarray, pred_dhw: np.ndarray, cam: np.ndarray):
    """
    Deterministic representative-slice selection:
      1) maximize 2D Dice(gt, pred)
      2) maximize IoU(gt, pred)
      3) maximize GT tumor area
      4) maximize union area
      5) fallback to lower slice index for full determinism

    Importantly, Grad-CAM intensity is NOT used for slice selection.
    If no slice contains GT or predicted tumor pixels at all,
    choose the slice with the highest total Grad-CAM activation.
    """
    if not (image_dhw.shape == gt_dhw.shape == pred_dhw.shape == cam.shape):
        raise RuntimeError("Shape mismatch while choosing representative slice")

    D = image_dhw.shape[0]
    has_any_tumor = bool(np.any(gt_dhw > 0) or np.any(pred_dhw > 0))

    if not has_any_tumor:
        total_scores = np.sum(cam, axis=(1, 2))
        idx = int(np.argmax(total_scores))
        return idx, {
            "selection_mode": "fallback_max_total_gradcam",
            "slice_dice_2d": np.nan,
            "slice_iou_2d": np.nan,
            "slice_union_area": 0,
            "slice_cam_in_union": 0.0,
            "slice_cam_total": float(total_scores[idx]),
        }

    best_key = None
    best_idx = None
    best_info = None

    for z in range(D):
        gt = gt_dhw[z].astype(bool)
        pred = pred_dhw[z].astype(bool)
        union = np.logical_or(gt, pred)

        if union.sum() == 0:
            continue

        dsc = dice_binary(gt, pred)
        iou = iou_binary(gt, pred)
        union_area = int(union.sum())
        cam_in_union = float((cam[z] * union.astype(np.float32)).sum())
        cam_total = float(cam[z].sum())

        # Selection is independent of Grad-CAM intensity.
        # Final tiebreaker: smaller z is preferred for determinism.
        gt_area = int(gt.sum())
        key = (
            dsc,
            iou,
            gt_area,
            union_area,
            -z,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = z
            best_info = {
                "selection_mode": "best_prediction_effect_slice",
                "slice_dice_2d": dsc,
                "slice_iou_2d": iou,
                "slice_gt_area": gt_area,
                "slice_union_area": union_area,
                "slice_cam_in_union": cam_in_union,
                "slice_cam_total": cam_total,
            }

    if best_idx is None:
        total_scores = np.sum(cam, axis=(1, 2))
        idx = int(np.argmax(total_scores))
        return idx, {
            "selection_mode": "fallback_max_total_gradcam",
            "slice_dice_2d": np.nan,
            "slice_iou_2d": np.nan,
            "slice_union_area": 0,
            "slice_cam_in_union": 0.0,
            "slice_cam_total": float(total_scores[idx]),
        }

    return int(best_idx), best_info


def choose_max_gt_tumor_area_slice(
    gt_dhw: np.ndarray,
    pred_dhw: np.ndarray,
    cam: np.ndarray,
):
    """
    Select the axial slice with the largest MANUAL/GT tumor area.

    This selection is independent of segmentation agreement and therefore
    avoids choosing a display slice because the prediction looks especially
    good. If the GT mask is unexpectedly empty, use the largest predicted-mask
    area as fallback; if both are empty, use maximal total Grad-CAM activation.
    """
    if not (gt_dhw.shape == pred_dhw.shape == cam.shape):
        raise RuntimeError("Shape mismatch while choosing max-GT-area slice")

    gt_areas = gt_dhw.sum(axis=(1, 2)).astype(np.int64)
    if int(gt_areas.max()) > 0:
        idx = int(np.argmax(gt_areas))
        mode = "max_gt_tumor_area_slice"
    else:
        pred_areas = pred_dhw.sum(axis=(1, 2)).astype(np.int64)
        if int(pred_areas.max()) > 0:
            idx = int(np.argmax(pred_areas))
            mode = "fallback_max_pred_tumor_area_slice"
        else:
            total_cam = cam.sum(axis=(1, 2))
            idx = int(np.argmax(total_cam))
            mode = "fallback_max_total_gradcam"

    gt = gt_dhw[idx].astype(bool)
    pred = pred_dhw[idx].astype(bool)
    union = np.logical_or(gt, pred)

    return idx, {
        "selection_mode": mode,
        "slice_dice_2d": dice_binary(gt, pred),
        "slice_iou_2d": iou_binary(gt, pred),
        "slice_gt_area": int(gt.sum()),
        "slice_pred_area": int(pred.sum()),
        "slice_union_area": int(union.sum()),
        "slice_cam_in_union": float((cam[idx] * union.astype(np.float32)).sum()),
        "slice_cam_total": float(cam[idx].sum()),
    }


def stage_name(binary_label: int) -> str:
    return "pT3-4" if int(binary_label) == 1 else "pT1-2"



# ============================================================================
# Model 2 Grad-CAM publication panel exporter -- V1 SCIENTIFIC AUDIT
# User-defined output: two slices x six PNGs. Original NIfTI orientation;
# no rotations, flips, crops, translated extents, or display-space resizing.
# Matching image footprint is fixed at 3200 x 3200 in EVERY 3500 x 3200 PNG.
# All data/model/Grad-CAM routines above are inherited from the original script.
# ============================================================================
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from data_loader import zscore_nonzero, percentile_normalize

PNG_DPI = 400
DATA_PIXEL_N = 3200
RIGHT_PIXEL_N = 850
SQUARE_CANVAS_WIDTH_PX = DATA_PIXEL_N
CAM_CANVAS_WIDTH_PX = DATA_PIXEL_N + RIGHT_PIXEL_N
CANVAS_WIDTH_PX = CAM_CANVAS_WIDTH_PX
CANVAS_HEIGHT_PX = DATA_PIXEL_N
OVERLAY_CAM_ALPHA = 0.45
CAM_CMAP = "jet"
CAM_VMIN = 0.0
CAM_VMAX = 1.0
# No arbitrary low-activation cutoff: the full CAM 0-1 is displayed consistently.
# Overlay alpha is deliberately lower than CAM-only; bar is ALWAYS opaque.
CAM_VISUAL_THRESHOLD = None
DISPLAY_INTERPOLATION = "nearest"
EXPECTED_SPATIAL = (24, 400, 400)
EXPECTED_TARGET_MODULE = "backbone.class_layers.relu"


def strict_loader_voxel_audit(image_tensor, raw_image_dhw, pred_dhw, case_key):
    """Fail closed if either real model input channel differs from NIfTI voxel grid.

    Mirrors the uploaded data_loader.py: to_dhw + normalization, no val/test
    augmentation; this check detects axis flips, transposes, shifted masks, etc.
    It is NOT a proof of clinical left/right orientation (which is in affine).
    """
    if image_tensor.ndim != 4 or tuple(image_tensor.shape[1:]) != tuple(raw_image_dhw.shape):
        raise RuntimeError(f"{case_key}: input/raw spatial shape mismatch")
    if image_tensor.shape[0] != 2:
        raise RuntimeError(f"{case_key}: two input channels required")
    if tuple(raw_image_dhw.shape) != tuple(pred_dhw.shape):
        raise RuntimeError(f"{case_key}: original image/pred shape mismatch")
    if tuple(raw_image_dhw.shape) != EXPECTED_SPATIAL:
        raise RuntimeError(f"{case_key}: nonstandard DHW {raw_image_dhw.shape}")

    tensor_np = image_tensor.detach().cpu().numpy()
    if Config.intensity_normalization == "zscore_nonzero":
        expected_image = zscore_nonzero(raw_image_dhw.copy())
    elif Config.intensity_normalization == "percentile":
        expected_image = percentile_normalize(raw_image_dhw.copy())
    else:
        expected_image = raw_image_dhw.astype(np.float32)
    if not np.all(np.isfinite(expected_image)) or not np.all(np.isfinite(tensor_np)):
        raise RuntimeError(f"{case_key}: nonfinite raw/model voxels")

    diff_image = np.abs(tensor_np[0].astype(np.float64) - expected_image.astype(np.float64))
    max_image_err = float(diff_image.max())
    image_match = bool(np.allclose(tensor_np[0], expected_image, atol=1e-5, rtol=1e-5))
    mask_match = bool(np.array_equal(tensor_np[1], (pred_dhw > 0).astype(np.float32)))
    mask_mismatches = int(np.count_nonzero(tensor_np[1] != (pred_dhw > 0).astype(np.float32)))

    if not image_match or not mask_match:
        raise RuntimeError(
            f"{case_key}: STOP -- Model 2 input not voxel-aligned with NIfTI. "
            f"T2WI_normalized_max_error={max_image_err:.8g}, "
            f"Pred_mask_mismatched_voxels={mask_mismatches}. "
            "Do NOT reorient or resize the PNG to conceal this."
        )
    return {
        "input_t2wi_matches_normalized_original": True,
        "input_pred_matches_original_binary_mask": True,
        "max_image_voxel_abs_error": max_image_err,
        "pred_mask_mismatched_voxels": 0,
        "model_spatial_shape_dhw": list(raw_image_dhw.shape),
        "test_augmentation_applied": False,
    }


def create_square_canvas(height, width):
    """Square canvas for T2WI-only / mask-filled panels.

    The displayed MRI footprint remains 3200 x 3200 px and fills the whole PNG.
    No right gutter is reserved for these non-GradCAM panels.
    """
    if (height, width) != (400, 400):
        raise RuntimeError(f"Expected 400x400 2D display, got {(height, width)}")
    fig = plt.figure(
        figsize=(SQUARE_CANVAS_WIDTH_PX / PNG_DPI, CANVAS_HEIGHT_PX / PNG_DPI),
        dpi=PNG_DPI, facecolor="white"
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    x0, x1 = -0.5, float(width) - 0.5
    y0, y1 = float(height) - 0.5, -0.5
    extent = (x0, x1, y0, y1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return fig, ax, extent, (SQUARE_CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX)


def create_cam_canvas(height, width):
    """Canvas for GradCAM-containing panels with a right annotation gutter."""
    if (height, width) != (400, 400):
        raise RuntimeError(f"Expected 400x400 2D display, got {(height, width)}")
    fig = plt.figure(
        figsize=(CAM_CANVAS_WIDTH_PX / PNG_DPI, CANVAS_HEIGHT_PX / PNG_DPI),
        dpi=PNG_DPI, facecolor="white"
    )
    ax = fig.add_axes([0.0, 0.0, DATA_PIXEL_N / CAM_CANVAS_WIDTH_PX, 1.0])
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    x0, x1 = -0.5, float(width) - 0.5
    y0, y1 = float(height) - 0.5, -0.5
    extent = (x0, x1, y0, y1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return fig, ax, extent, (CAM_CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX)


def show_image_exact(ax, pixels, extent, *, cmap=None, vmin=None, vmax=None, alpha=None):
    if pixels.ndim not in (2, 3):
        raise RuntimeError(f"Non-2D display array: {pixels.shape}")
    h, w = pixels.shape[:2]
    if (h, w) != (400, 400):
        raise RuntimeError(f"Invalid plane dimensions: {pixels.shape}")
    return ax.imshow(
        pixels, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
        origin="upper", extent=extent, interpolation=DISPLAY_INTERPOLATION,
        zorder=1 if cmap == "gray" else 2,
    )


def rgba_mask_exact(mask, color, alpha):
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[..., :3] = plt.matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.asarray(mask > 0, dtype=np.float32) * float(alpha)
    return rgba


def draw_contour_index_coordinates(ax, mask2d, color, lw):
    """Explicit x=column, y=row index coordinates; avoids origin ambiguity."""
    if not np.any(mask2d > 0):
        return
    h, w = mask2d.shape
    ax.contour(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        mask2d.astype(np.float32),
        levels=[0.5], colors=[color], linewidths=lw, zorder=4,
    )


def fixed_colorbar(fig):
    # Independent ScalarMappable guarantees the same opaque 0-1 jet colorbar
    # in all three CAM-related PNGs. The right gutter is widened so the bar,
    # ticks and label fit without encroaching on the MRI image footprint.
    cax = fig.add_axes([3340 / CANVAS_WIDTH_PX, 560 / CANVAS_HEIGHT_PX,
                        100 / CANVAS_WIDTH_PX, 2240 / CANVAS_HEIGHT_PX])
    mappable = ScalarMappable(norm=Normalize(vmin=CAM_VMIN, vmax=CAM_VMAX), cmap=CAM_CMAP)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=22, length=9,width=1.5,pad=2.5)
    cb.set_label("Normalized Grad-CAM activation map", fontsize=28, labelpad=18)


def draw_mask_labels(fig, names):
    # Legend is always OUTSIDE the MRI image; it never covers anatomy.
    # To avoid clipping, each legend item is drawn on two rows:
    # a colored line above and the corresponding text centered below it.
    if not names:
        return
    la = fig.add_axes([3260 / CANVAS_WIDTH_PX, 0.010,
                       820 / CANVAS_WIDTH_PX, 0.135])
    la.set_xlim(0, 1)
    la.set_ylim(0, 1)
    la.axis("off")

    if len(names) == 1:
        centers = [0.56]
    else:
        centers = [0.74, 0.24]

    for (label, color), cy in zip(names, centers):
        la.plot([0.18, 0.82], [cy + 0.10, cy + 0.10], color=color, linewidth=6.0)
        la.text(0.50, cy - 0.02, label, fontsize=20, fontweight="bold",ha="center", va="top")


def draw_mask_labels_in_image(ax, names):
    """Compact legend placed inside the lower-right corner of the MRI image.

    Used only for non-GradCAM GT/Pred filled panels, per user request.
    """
    if not names:
        return
    x_line0, x_line1 = 0.56, 0.68
    x_text = 0.70
    if len(names) == 1:
        ys = [0.06]
    else:
        ys = [0.10, 0.04]
    for (label, color), y in zip(names, ys):
        ax.plot([x_line0, x_line1], [y, y], transform=ax.transAxes,
                color=color, linewidth=8.0, solid_capstyle='butt', zorder=6)
        ax.text(x_text, y, label, transform=ax.transAxes,
                fontsize=32, color='white', ha='left', va='center', zorder=6,
                bbox=dict(facecolor=(0, 0, 0, 0.55), edgecolor='none', pad=2.5))


def write_panel_png(fig, path, *, allow_overwrite, expected_size):
    path = Path(path)
    if len(str(path)) >= MAX_SAFE_WINDOWS_PATH_LENGTH:
        plt.close(fig)
        raise RuntimeError(f"Windows path too long ({len(str(path))}): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_overwrite:
        plt.close(fig)
        raise FileExistsError(
            f"Existing result must not be silently reused: {path}. "
            "Set --overwrite to intentionally regenerate."
        )
    try:
        fig.savefig(path, dpi=PNG_DPI, facecolor="white", pad_inches=0)
    finally:
        plt.close(fig)
    # Pillow is optional in normal Matplotlib, but is available in the
    # expected environment (Matplotlib's PNG backend uses it).
    from PIL import Image
    with Image.open(path) as png:
        if png.size != tuple(expected_size):
            raise RuntimeError(f"Incorrect PNG dimensions {png.size}: {path}; expected {tuple(expected_size)}")


def write_five_panel_summary_png(png_dict, out_path, *, allow_overwrite):
    """Create one additional high-resolution montage WITHOUT changing the
    individual panel renderings.

    Order required by the user:
      1) T2WI
      2) T2WI + GT mask (filled)
      3) T2WI + Pred mask (filled)
      4) Grad-CAM only
      5) T2WI + Grad-CAM + GT/Pred contours
    """
    from PIL import Image

    order = [
        "01_T2WI",
        "02_T2WI_GT_filled",
        "03_T2WI_Pred_filled",
        "04_GradCAM_only",
        "06_T2WI_GradCAM_GT_Pred_contours",
    ]
    paths = [Path(png_dict[k]) for k in order]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Panel missing for 5-panel summary: {p}")

    out_path = Path(out_path)
    if len(str(out_path)) >= MAX_SAFE_WINDOWS_PATH_LENGTH:
        raise RuntimeError(f"Windows path too long ({len(str(out_path))}): {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Existing 5-panel summary must not be silently reused: {out_path}. "
            "Set --overwrite to intentionally regenerate."
        )

    gap_px = 24
    images = []
    try:
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(img)
        widths = [im.size[0] for im in images]
        heights = [im.size[1] for im in images]
        if len(set(heights)) != 1:
            raise RuntimeError(f"5-panel summary input heights differ: {heights}")
        h = heights[0]
        total_w = sum(widths) + gap_px * (len(images) - 1)
        canvas = Image.new("RGB", (total_w, h), color=(255, 255, 255))
        x = 0
        for im in images:
            canvas.paste(im, (x, 0))
            x += im.size[0] + gap_px
        canvas.save(out_path, format="PNG")
    finally:
        for im in images:
            try:
                im.close()
            except Exception:
                pass

    with Image.open(out_path) as png:
        expected_w = (3 * SQUARE_CANVAS_WIDTH_PX) + (2 * CAM_CANVAS_WIDTH_PX) + gap_px * (len(order) - 1)
        if png.size != (expected_w, CANVAS_HEIGHT_PX):
            raise RuntimeError(f"Incorrect 5-panel PNG dimensions {png.size}: {out_path}")
    return str(out_path)


def export_six_pngs(out_dir, cohort, cid, slice_name, z, raw, gt, pred, cam, *, overwrite):
    if not (raw.shape == gt.shape == pred.shape == cam.shape):
        raise RuntimeError(f"{cohort}/{cid}: volume alignment error")
    img2d, gt2d, pred2d, cam2d = [a[z] for a in (raw, gt, pred, cam)]
    if not all(a.shape == (400, 400) for a in (img2d, gt2d, pred2d, cam2d)):
        raise RuntimeError(f"{cohort}/{cid}: 2D dimension mismatch")
    if not np.all(np.isfinite(img2d)) or not np.all(np.isfinite(cam2d)):
        raise RuntimeError(f"{cohort}/{cid}: nonfinite display pixels")
    if float(np.min(cam2d)) < -1e-6 or float(np.max(cam2d)) > 1+1e-6:
        raise RuntimeError(f"{cohort}/{cid}: Grad-CAM expected in [0,1]")

    vmin, vmax = robust_image_display(raw)
    base_name = f"{cohort}_{cid}_{slice_name}_z{z:02d}"
    names = (
        "01_T2WI",
        "02_T2WI_GT_filled",
        "03_T2WI_Pred_filled",
        "04_GradCAM_only",
        "05_T2WI_GradCAM_overlay",
        "06_T2WI_GradCAM_GT_Pred_contours",
    )
    outpaths = []
    for panel_type, stem in enumerate(names, start=1):
        if panel_type in (1, 2, 3):
            fig, ax, extent, expected_size = create_square_canvas(*img2d.shape)
        else:
            fig, ax, extent, expected_size = create_cam_canvas(*img2d.shape)

        if panel_type == 4:
            show_image_exact(ax, cam2d, extent, cmap=CAM_CMAP, vmin=CAM_VMIN, vmax=CAM_VMAX)
        else:
            show_image_exact(ax, img2d, extent, cmap="gray", vmin=vmin, vmax=vmax)

        if panel_type == 2:
            show_image_exact(ax, rgba_mask_exact(gt2d, GT_COLOR, MASK_ALPHA), extent)
            draw_mask_labels_in_image(ax, [("GT mask", GT_COLOR)])
        elif panel_type == 3:
            show_image_exact(ax, rgba_mask_exact(pred2d, PRED_COLOR, MASK_ALPHA), extent)
            draw_mask_labels_in_image(ax, [("Pred mask", PRED_COLOR)])
        elif panel_type in (5, 6):
            show_image_exact(
                ax, cam2d, extent, cmap=CAM_CMAP, vmin=CAM_VMIN,
                vmax=CAM_VMAX, alpha=OVERLAY_CAM_ALPHA
            )
            if panel_type == 6:
                draw_contour_index_coordinates(ax, gt2d, GT_COLOR, CONTOUR_LINEWIDTH)
                draw_contour_index_coordinates(ax, pred2d, PRED_COLOR, CONTOUR_LINEWIDTH)
                draw_mask_labels_in_image(ax, [("GT mask", GT_COLOR), ("Pred mask", PRED_COLOR)])
        if panel_type in (4, 5, 6):
            fixed_colorbar(fig)

        path = out_dir / f"{base_name}_{stem}.png"
        write_panel_png(fig, path, allow_overwrite=overwrite, expected_size=expected_size)
        outpaths.append(str(path))
    return dict(zip(names, outpaths))


def draw_mask_labels_in_image_bottom_right(ax, names):
    if not names:
        return

    x_line0, x_line1 = 0.63, 0.77
    x_text = 0.79
    ys = [0.16, 0.08]

    for (label, color), y in zip(names, ys):
        ax.plot(
            [x_line0, x_line1], [y, y],
            transform=ax.transAxes,
            color=color,
            linewidth=6.0,
            solid_capstyle="butt",
            zorder=8
        )
        ax.text(
            x_text, y, label,
            transform=ax.transAxes,
            fontsize=32,
            fontweight="bold",
            color="white",
            ha="left",
            va="center",
            zorder=8,
            bbox=dict(facecolor=(0, 0, 0, 0.50), edgecolor="none", pad=2.0)
        )


def save_cam_nifti(cam_dhw, source_nii, depth_axis, filename):
    cam_native = dhw_to_nifti_native(cam_dhw, depth_axis)
    if cam_native.shape != source_nii.shape:
        raise RuntimeError(f"CAM NIfTI grid mismatch {cam_native.shape} != {source_nii.shape}")
    h = source_nii.header.copy()
    h.set_data_dtype(np.float32)
    cam_nii = nib.Nifti1Image(cam_native.astype(np.float32), source_nii.affine, header=h)
    # Preserve sform/qform codes as well as the primary affine.
    qform, qcode = source_nii.get_qform(coded=True)
    sform, scode = source_nii.get_sform(coded=True)
    if qform is not None:
        cam_nii.set_qform(qform, int(qcode))
    if sform is not None:
        cam_nii.set_sform(sform, int(scode))
    nib.save(cam_nii, str(filename))


def process_cohort(
    cohort, manifest, model, gradcam, model_info, device, *,
    target_mode, target_name, tol, skip_probability_audit, max_cases, case_ids, overwrite
):
    cohort_dir = OUTPUT_ROOT / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)
    # Strictly reuse original classifier's test CSV pathway, NOT a new
    # normalization, resampling, reorientation or mask source.
    temp = manifest[["case_id_folder", "label", "image_path", "pred_mask_path"]].copy()
    temp = temp.rename(columns={"pred_mask_path": "mask_path"})
    temp_csv = cohort_dir / f"{cohort}_Model2_GradCAM_input_manifest.csv"
    temp.to_csv(temp_csv, index=False, encoding="utf-8-sig")
    samples = load_samples_from_test_csv(str(temp_csv), input_mode="image_plus_pred_mask")
    if len(samples) != len(manifest):
        raise RuntimeError(f"{cohort}: loaded {len(samples)}, expected {len(manifest)}")
    dataset = RectalMRIClassification3DDataset(samples=samples, is_train=False)
    reference = None if skip_probability_audit else load_reference_probabilities(cohort, model_info["fold"])
    if not skip_probability_audit and reference is None:
        raise RuntimeError(f"{cohort}: no validated Model 2 prediction reference found")

    keep_indices = list(range(len(manifest)))
    if case_ids:
        requested = {normalize_case_id_4digit(x) if cohort == "IT1" else str(x).strip() for x in case_ids}
        keep_indices = [i for i in keep_indices if str(manifest.iloc[i]["case_id_folder"]) in requested]
    if max_cases is not None:
        keep_indices = keep_indices[:max_cases]
    rows = []
    for number, i in enumerate(keep_indices, start=1):
        row = manifest.iloc[i]
        cid = str(row["case_id_folder"])
        label = int(row["label"])
        tensor, label_tensor, loader_case_id = dataset[i]
        actual_id = normalize_case_id_4digit(loader_case_id) if cohort == "IT1" else str(loader_case_id).strip()
        if actual_id != cid or int(label_tensor.item()) != label:
            raise RuntimeError(f"{cohort}/{cid}: sample ID or label mismatch")

        image_path = Path(row["image_path"])
        img_nii, image_dhw, gt_dhw, pred_dhw, depth_axis = load_raw_image_and_masks(
            image_path, Path(row["gt_mask_path"]), Path(row["pred_mask_path"])
        )
        spatial_audit = strict_loader_voxel_audit(tensor, image_dhw, pred_dhw, f"{cohort}/{cid}")
        x = tensor.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits0 = model(x)
            p0 = float(F.softmax(logits0, dim=1)[0, 1].item())
        yp = int(p0 >= model_info["threshold"])
        target_class = yp if target_mode == "decision" else (1 if target_mode == "class1" else 0)
        cam, logits1 = gradcam(x, target_class)
        p1 = float(F.softmax(logits1, dim=1)[0, 1].item())
        if abs(p1-p0) > 1e-7:
            raise RuntimeError(f"{cohort}/{cid}: repeated forward mismatch {p0} != {p1}")
        if reference is not None:
            if cid not in reference or abs(p1-reference[cid]) > tol:
                raise RuntimeError(
                    f"{cohort}/{cid}: Model 2 probability reproduction FAILED: "
                    f"observed={p1:.9f}, reference={reference.get(cid)}, tolerance={tol}"
                )
        if cam.shape != image_dhw.shape or cam.shape != pred_dhw.shape or cam.shape != gt_dhw.shape:
            raise RuntimeError(f"{cohort}/{cid}: CAM and image/mask voxel grids differ")
        if int(np.count_nonzero(gt_dhw)) == 0:
            raise RuntimeError(
                f"{cohort}/{cid}: GT tumor mask is empty: cannot scientifically "
                "define the maximum GT tumor area slice"
            )

        z_dice, info_dice = choose_best_dice_slice(image_dhw, gt_dhw, pred_dhw, cam)
        z_gt, info_gt = choose_max_gt_tumor_area_slice(gt_dhw, pred_dhw, cam)
        case_dir = cohort_dir / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        groups = [
            ("01_best_GT_Pred_Dice_slice", "bestDice", z_dice, info_dice),
            ("02_max_GT_tumor_area_slice", "maxGTarea", z_gt, info_gt),
        ]
        group_meta = {}
        for folder, short_tag, z, selection_info in groups:
            six = export_six_pngs(case_dir / folder, cohort, cid, short_tag, z,
                                  image_dhw, gt_dhw, pred_dhw, cam, overwrite=overwrite)
            base_name = f"{cohort}_{cid}_{short_tag}_z{int(z):02d}"
            five_panel = write_five_panel_summary_png(
                six,
                case_dir / folder / f"{base_name}_07_five_panel_summary.png",
                allow_overwrite=overwrite,
            )
            group_meta[short_tag] = {
                "slice_index_dhw_zero_based": int(z),
                "selection": selection_info,
                "png_files": six,
                "five_panel_summary_png": five_panel,
            }
        cam_file = case_dir / f"{cid}_Model2_GradCAM_3D.nii.gz"
        if cam_file.exists() and not overwrite:
            raise FileExistsError(f"Existing Grad-CAM NIfTI: {cam_file}")
        save_cam_nifti(cam, img_nii, depth_axis, cam_file)

        meta = {
            "cohort": cohort, "case_id": cid, "source_image": str(image_path),
            "source_gt_mask": str(row["gt_mask_path"]),
            "source_pred_mask": str(row["pred_mask_path"]),
            "ground_truth_class": label, "predicted_class": yp,
            "p_class1": p1, "reference_p_class1": None if reference is None else reference[cid],
            "decision_threshold": model_info["threshold"],
            "gradcam_target_class": target_class, "gradcam_target_layer": target_name,
            "gradcam_target_activation_shape": list(gradcam.activations.shape),
            "gradcam_normalization": "per-patient full 3D min-max after upsampling",
            "gradcam_colormap": CAM_CMAP, "gradcam_vmin": CAM_VMIN, "gradcam_vmax": CAM_VMAX,
            "gradcam_overlay_alpha": OVERLAY_CAM_ALPHA,
            "gradcam_display_threshold": CAM_VISUAL_THRESHOLD,
            "imaging_display": "original NIfTI voxel order after shared to_dhw, NO rotate/flip",
            "original_nifti_shape": list(img_nii.shape),
            "original_nifti_affine": img_nii.affine.tolist(),
            "nifti_depth_axis_in_native_storage": int(depth_axis),
            "model_input_t2wi_pred_voxel_audit": spatial_audit,
            "png_size_px": [CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX],
            "image_footprint_px": [DATA_PIXEL_N, DATA_PIXEL_N],
            "t2wi_display_vmin_vmax": list(robust_image_display(image_dhw)),
            "gradcam_nifti": str(cam_file), "two_selected_slices": group_meta,
        }
        save_json(meta, case_dir / f"{cid}_Model2_GradCAM_AUDIT.json")
        rows.append({
            "cohort": cohort, "case_id": cid, "y_true": label, "y_pred": yp,
            "p_class1": p1, "target_class": target_class,
            "best_dice_z": z_dice, "best_dice_2d": info_dice.get("slice_dice_2d"),
            "max_gt_area_z": z_gt, "max_gt_tumor_area_voxels": info_gt.get("slice_gt_area"),
            "input_t2wi_grid_audit": True, "input_pred_grid_audit": True,
            "gradcam_nii": str(cam_file),
            "best_dice_png_dir": str(case_dir / groups[0][0]),
            "max_gt_area_png_dir": str(case_dir / groups[1][0]),
        })
        print(f"[{cohort}] {number}/{len(keep_indices)}: "
              f"p1={p1:.6f} pred={yp} bestDiceZ={z_dice} maxGTareaZ={z_gt} "
              "| 12 PNGs, input-grid AUDIT PASSED")
        del x, logits0, logits1, cam, tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(cohort_dir / f"{cohort}_Model2_GradCAM_6PNG_manifest.csv",
               index=False, encoding="utf-8-sig")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Audited Final OOF Model 2 Grad-CAM: 2 slices x 6 equal-sized 400-DPI PNGs"
    )
    parser.add_argument("--cohorts", nargs="+", choices=["IT1", "IT2", "ET"],
                        default=["IT1", "IT2", "ET"])
    parser.add_argument("--case-ids", nargs="+", default=None,
                        help="Optional anonymized case IDs for spot-check, e.g. --case-ids RCM_0001")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--target", choices=["decision", "class1", "class0"], default="decision")
    parser.add_argument("--prob-tolerance", type=float, default=1e-5)
    parser.add_argument("--skip-probability-audit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >0")
    configure_deterministic_torch()
    Config.input_mode = "image_plus_pred_mask"
    Config.model_name = "densenet121_3d"
    Config.label_mode = "binary"
    if Config.input_size_3d != EXPECTED_SPATIAL:
        raise RuntimeError("Model2 Config input_size_3d mismatch")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    info = load_model2_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(info, device)
    target_name, target_layer = resolve_target_layer(model)
    # Do not silently select another architecture layer.
    if target_name != EXPECTED_TARGET_MODULE:
        raise RuntimeError(f"Audited layer {EXPECTED_TARGET_MODULE} missing; resolved {target_name}")
    engine = GradCAM3D(model, target_layer)
    records = []
    try:
        for c in args.cohorts:
            manifest = build_manifest(c)
            df = process_cohort(
                c, manifest, model, engine, info, device,
                target_mode=args.target, target_name=target_name,
                tol=args.prob_tolerance, skip_probability_audit=args.skip_probability_audit,
                max_cases=args.max_cases, case_ids=args.case_ids, overwrite=args.overwrite
            )
            records.append(df)
    finally:
        engine.remove()
    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    combined.to_csv(OUTPUT_ROOT / "ALL_Model2_GradCAM_6PNG_manifest.csv", index=False, encoding="utf-8-sig")
    provenance = {
        "created_at": datetime.now().isoformat(),
        "selected_model_json": str(OOF_MODEL2_SELECTION_JSON),
        "selected_model_json_sha256": sha256_file(OOF_MODEL2_SELECTION_JSON),
        "checkpoint": str(info["checkpoint"]),
        "checkpoint_sha256": sha256_file(info["checkpoint"]),
        "selected_fold": info["fold"], "threshold": info["threshold"],
        "validation_auc": info["validation_auc"],
        "model_input_mode": Config.input_mode,
        "cohorts": args.cohorts, "case_ids": args.case_ids, "max_cases": args.max_cases,
        "target_mode": args.target, "target_module": target_name,
        "per_case_full_3D_CAM_normalization": "min-max to [0,1] after trilinear interpolation",
        "png_dpi": PNG_DPI, "png_size": [CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX],
        "identical_image_footprint_px_in_all_six": [DATA_PIXEL_N, DATA_PIXEL_N],
        "plane_voxel_shape": [400, 400],
        "display_no_rotation_no_flip_no_crop": True,
        "cam_overlay_alpha": OVERLAY_CAM_ALPHA, "cam_display_threshold": CAM_VISUAL_THRESHOLD,
        "colorbar_uniform_opaque_jet_0_to_1": True,
        "probability_audit_skipped": bool(args.skip_probability_audit),
        "outputs_per_patient": "2 selected slices x 6 PNGs, plus 3D Grad-CAM NIfTI and audit JSON",
    }
    save_json(provenance, OUTPUT_ROOT / "Model2_GradCAM_6PNG_RUN_PROVENANCE.json")
    print(f"COMPLETED: {len(combined)} patients, {len(combined)*12} PNGs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()