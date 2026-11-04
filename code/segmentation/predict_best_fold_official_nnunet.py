import os
import re
import json
import math
import time
import shutil
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch

from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

# =========================================================

# =========================================================
from config import Config



os.environ["nnUNet_raw"] = str(Config.nnunet_raw)
os.environ["nnUNet_preprocessed"] = str(Config.nnunet_preprocessed)
os.environ["nnUNet_results"] = str(Config.nnunet_results)
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.gpu_id)
os.environ["nnUNet_n_proc_DA"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor




MODEL_ROOT = Path(
    r"RECTAL_STAS_DATA\nnUNet_results"
    r"\Dataset304_RectalCancerMRI_TwoCenters_TrainvalMerged_XY400Z24"
    r"\nnUNetTrainer__nnUNetPlans__3d_fullres"
)




#   "ema_pseudo_dice"
FOLD_SELECTION_MODE = "validation_dice_mean"


MANUAL_FOLD_OVERRIDE = None
# MANUAL_FOLD_OVERRIDE = 3

RUN_INFERENCE = True
WRITE_BACK_PRED_TO_CASE_DIR = False


GLOBAL_OUTPUT_ROOT = Path(
    r"RECTAL_STAS_DATA\nnUNet_bestfold_prediction_and_metrics"
)



# Development cohort (617) is evaluated with OOF masks, NOT with the
# fixed segmentation model. Do not include Merged_trainval here.
TARGET_DATASETS = [
    {
        "name": "Internal test set 1",
        "cases_dir": Path(
            r"RECTAL_STAS_DATA\split_dataset_tstage_fixedtest10\test\cases"
        ),
        "enabled": True,
    },
    {
        "name": "Internal test set 2",
        "cases_dir": Path(
            r"RECTAL_STAS_DATA\test_cohort_B\cases"
        ),
        "enabled": True,
    },
]


# =========================================================

SELECTED_FOLD = None
BEST_FOLD_SCORE = None
BEST_SCORE_SOURCE = None
BEST_SCORE_TYPE = None
BEST_CHECKPOINT_PATH = None



def add_timestamp_to_path(path: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def safe_to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        alt = add_timestamp_to_path(path)
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"⚠️ CSV is locked, saving to: {alt}")
        return alt


def safe_to_excel(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        alt = add_timestamp_to_path(path)
        df.to_excel(alt, index=False)
        print(f"⚠️ Excel is locked, saving to: {alt}")
        return alt


def safe_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        return path
    except PermissionError:
        alt = add_timestamp_to_path(path)
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        print(f"⚠️ JSON is locked, saving to: {alt}")
        return alt


def fold_dir(fold: int):
    return MODEL_ROOT / f"fold_{fold}"


def checkpoint_best_path(fold: int):
    return fold_dir(fold) / "checkpoint_best.pth"


def checkpoint_latest_path(fold: int):
    return fold_dir(fold) / "checkpoint_latest.pth"


def checkpoint_final_path(fold: int):
    return fold_dir(fold) / "checkpoint_final.pth"


def list_possible_log_files(fold_path: Path):
    files = []
    for pat in ["*.txt", "*.log"]:
        files.extend(fold_path.rglob(pat))
    return sorted(set(files))


def extract_all_dice_values_from_summary(obj):
    vals = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == "dice" and isinstance(v, (int, float)):
                vals.append(float(v))
            else:
                vals.extend(extract_all_dice_values_from_summary(v))
    elif isinstance(obj, list):
        for item in obj:
            vals.extend(extract_all_dice_values_from_summary(item))

    return vals


def parse_validation_dice_mean_from_summary(fold_path: Path):
    summary_path = fold_path / "validation" / "summary.json"
    if not summary_path.exists():
        return None, None, None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None

    dice_vals = extract_all_dice_values_from_summary(data)
    if len(dice_vals) == 0:
        return None, None, None

    return float(np.mean(dice_vals)), "validation_dice_mean", str(summary_path)


def parse_best_ema_pseudo_dice_from_logs(fold_path: Path):
    log_files = list_possible_log_files(fold_path)

    best_score = None
    best_source = None
    pattern = re.compile(r"New best EMA pseudo Dice:\s*([0-9]*\.?[0-9]+)")

    for log_file in log_files:
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        vals = [float(m.group(1)) for m in pattern.finditer(text)]
        if len(vals) > 0:
            cur = max(vals)
            if best_score is None or cur > best_score:
                best_score = cur
                best_source = str(log_file)

    if best_score is None:
        return None, None, None

    return float(best_score), "ema_pseudo_dice", best_source


def try_parse_fold_score(fold: int):
    fdir = fold_dir(fold)

    if not fdir.exists():
        return {
            "fold": fold,
            "fold_dir": str(fdir),
            "fold_dir_exists": False,
            "checkpoint_best_exists": False,
            "checkpoint_latest_exists": False,
            "checkpoint_final_exists": False,
            "score": None,
            "score_type": "",
            "score_source": "",
        }

    best_ckpt = checkpoint_best_path(fold)
    latest_ckpt = checkpoint_latest_path(fold)
    final_ckpt = checkpoint_final_path(fold)

    if FOLD_SELECTION_MODE == "validation_dice_mean":
        score, score_type, score_source = parse_validation_dice_mean_from_summary(fdir)
    elif FOLD_SELECTION_MODE == "ema_pseudo_dice":
        score, score_type, score_source = parse_best_ema_pseudo_dice_from_logs(fdir)
    else:
        raise ValueError(f"Unknown FOLD_SELECTION_MODE: {FOLD_SELECTION_MODE}")

    return {
        "fold": fold,
        "fold_dir": str(fdir),
        "fold_dir_exists": True,
        "checkpoint_best_exists": best_ckpt.exists(),
        "checkpoint_latest_exists": latest_ckpt.exists(),
        "checkpoint_final_exists": final_ckpt.exists(),
        "score": score,
        "score_type": score_type,
        "score_source": score_source,
    }


def select_best_fold():
    global SELECTED_FOLD, BEST_FOLD_SCORE, BEST_SCORE_SOURCE, BEST_SCORE_TYPE, BEST_CHECKPOINT_PATH

    if MANUAL_FOLD_OVERRIDE is not None:
        if MANUAL_FOLD_OVERRIDE not in Config.folds:
            raise ValueError(f"MANUAL_FOLD_OVERRIDE={MANUAL_FOLD_OVERRIDE} not in Config.folds={Config.folds}")

        ckpt = checkpoint_best_path(MANUAL_FOLD_OVERRIDE)
        if not ckpt.exists():
            raise FileNotFoundError(f"fold {MANUAL_FOLD_OVERRIDE} has no checkpoint_best.pth: {ckpt}")

        SELECTED_FOLD = MANUAL_FOLD_OVERRIDE
        BEST_FOLD_SCORE = None
        BEST_SCORE_SOURCE = "manual_override"
        BEST_SCORE_TYPE = "manual_override"
        BEST_CHECKPOINT_PATH = ckpt
        print(f"🎯 Manually specified fold = {MANUAL_FOLD_OVERRIDE}")
        return

    rows = [try_parse_fold_score(f) for f in Config.folds]
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("Fold Selection Table")
    print("=" * 80)
    print(df.to_string(index=False))

    valid = df[
        (df["checkpoint_best_exists"] == True) &
        (df["score"].notna())
    ].copy()

    if len(valid) == 0:
        raise RuntimeError(
            "Unable to automatically select best fold.\nPlease check checkpoint_best.pth, validation summary or training logs."
        )

    valid = valid.sort_values(["score", "fold"], ascending=[False, True]).reset_index(drop=True)
    best_row = valid.iloc[0]

    SELECTED_FOLD = int(best_row["fold"])
    BEST_FOLD_SCORE = float(best_row["score"])
    BEST_SCORE_SOURCE = str(best_row["score_source"])
    BEST_SCORE_TYPE = str(best_row["score_type"])
    BEST_CHECKPOINT_PATH = checkpoint_best_path(SELECTED_FOLD)

    print("\n" + "=" * 80)
    print("Best Fold Selected")
    print("=" * 80)
    print(f"selected fold   : {SELECTED_FOLD}")
    print(f"selected score  : {BEST_FOLD_SCORE}")
    print(f"score type      : {BEST_SCORE_TYPE}")
    print(f"score source    : {BEST_SCORE_SOURCE}")
    print(f"checkpoint_best : {BEST_CHECKPOINT_PATH}")
    print("=" * 80)

    fold_sel_dir = GLOBAL_OUTPUT_ROOT / "fold_selection"
    fold_sel_dir.mkdir(parents=True, exist_ok=True)
    safe_to_csv(df, fold_sel_dir / "fold_selection_table.csv")
    safe_write_json(
        {
            "selection_mode": FOLD_SELECTION_MODE,
            "selected_fold": SELECTED_FOLD,
            "selected_score": BEST_FOLD_SCORE,
            "selected_score_type": BEST_SCORE_TYPE,
            "selected_score_source": BEST_SCORE_SOURCE,
            "selected_checkpoint_best_path": str(BEST_CHECKPOINT_PATH),
            "all_folds": df.to_dict(orient="records"),
        },
        fold_sel_dir / "fold_selection_summary.json"
    )


def parse_checkpoint_path(checkpoint_path: Path):
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    fdir = checkpoint_path.parent
    model_training_output_dir = fdir.parent
    checkpoint_name = checkpoint_path.name

    if not fdir.name.startswith("fold_"):
        raise ValueError(f"Checkpoint directory is not in fold_x format: {fdir.name}")

    fold = int(fdir.name.split("_")[1])
    return model_training_output_dir, fold, checkpoint_name


def strip_nii_gz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def find_image_mask(case_dir: Path):
    image_files = sorted(case_dir.glob("*_image.nii.gz"))
    mask_files = sorted(case_dir.glob("*_mask.nii.gz"))

    if len(image_files) != 1 or len(mask_files) != 1:
        raise FileNotFoundError(
            f"Files are not unique or missing in {case_dir.name}: image={len(image_files)}, mask={len(mask_files)}"
        )
    return image_files[0], mask_files[0]


def load_nifti(path: Path):
    nii = nib.load(str(path))
    data = nii.get_fdata()
    spacing = nii.header.get_zooms()[:3]
    return data, spacing, nii


def load_mask_as_binary(mask_path: Path):
    nii = nib.load(str(mask_path))
    data = nii.get_fdata()
    spacing = nii.header.get_zooms()[:3]
    binary = (data > 0).astype(np.uint8)
    return binary, spacing


def to_zyx(vol: np.ndarray) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")
    z_axis = int(np.argmin(vol.shape))
    return np.moveaxis(vol, z_axis, 0)


def normalize_image_for_display(vol_zyx: np.ndarray) -> np.ndarray:
    vol = vol_zyx.astype(np.float32)
    nonzero = vol[vol > 0]
    if len(nonzero) > 0:
        p1 = np.percentile(nonzero, 1)
        p99 = np.percentile(nonzero, 99)
    else:
        p1 = np.percentile(vol, 1)
        p99 = np.percentile(vol, 99)

    if p99 <= p1:
        p1 = float(vol.min())
        p99 = float(vol.max()) if vol.max() > vol.min() else float(vol.min() + 1.0)

    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol



def compute_basic_metrics(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_surface_distances(gt: np.ndarray, pred: np.ndarray, spacing):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if not gt.any() and not pred.any():
        return np.array([0.0], dtype=np.float32)

    if not gt.any() or not pred.any():
        return np.array([np.inf], dtype=np.float32)

    structure = generate_binary_structure(gt.ndim, 1)

    gt_surface = np.logical_xor(gt, binary_erosion(gt, structure=structure, border_value=0))
    pred_surface = np.logical_xor(pred, binary_erosion(pred, structure=structure, border_value=0))

    if not gt_surface.any() or not pred_surface.any():
        return np.array([np.inf], dtype=np.float32)

    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    dist_gt_to_pred = dt_pred[gt_surface]
    dist_pred_to_gt = dt_gt[pred_surface]

    return np.concatenate([dist_gt_to_pred, dist_pred_to_gt]).astype(np.float32)


def compute_hd95_and_asd(gt: np.ndarray, pred: np.ndarray, spacing):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if not gt.any() and not pred.any():
        return {"hd95": 0.0, "asd": 0.0}

    if not gt.any() or not pred.any():
        return {"hd95": float("inf"), "asd": float("inf")}

    distances = compute_surface_distances(gt, pred, spacing)

    if len(distances) == 0:
        return {"hd95": float("inf"), "asd": float("inf")}

    return {
        "hd95": float(np.percentile(distances, 95)),
        "asd": float(np.mean(distances)),
    }



def choose_representative_slice(gt_zyx: np.ndarray, pred_zyx: np.ndarray) -> int:
    union_mask = (gt_zyx > 0) | (pred_zyx > 0)
    areas = union_mask.reshape(union_mask.shape[0], -1).sum(axis=1)
    if np.max(areas) > 0:
        return int(np.argmax(areas))
    return int(union_mask.shape[0] // 2)


def save_representative_comparison_figure(
    image_zyx_norm: np.ndarray,
    pred_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    z: int,
    case_id: str,
    metrics_row: dict,
    out_png: Path,
):
    img = image_zyx_norm[z]
    pred = pred_zyx[z] > 0
    gt = gt_zyx[z] > 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"{case_id}\nOriginal T2WI\nz={z}")
    axes[0].axis("off")

    axes[1].imshow(gt.astype(np.uint8), cmap="Greens")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred.astype(np.uint8), cmap="Reds")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(img, cmap="gray")
    axes[3].imshow(np.ma.masked_where(~gt, gt), cmap="Greens", alpha=0.45)
    axes[3].set_title("T2WI + GT Overlay")
    axes[3].axis("off")

    axes[4].imshow(img, cmap="gray")
    axes[4].imshow(np.ma.masked_where(~pred, pred), cmap="autumn", alpha=0.45)
    axes[4].set_title("T2WI + Pred Overlay")
    axes[4].axis("off")

    axes[5].imshow(img, cmap="gray")
    axes[5].imshow(np.ma.masked_where(~pred, pred), cmap="autumn", alpha=0.45)
    if np.any(gt):
        axes[5].contour(gt.astype(np.uint8), levels=[0.5], colors="lime", linewidths=1.5)
    axes[5].set_title(
        "T2WI + Pred + GT contour\n"
        f"Dice={metrics_row['dice']:.4f}, IoU={metrics_row['iou']:.4f}, "
        f"HD95={metrics_row['hd95']:.4f}, ASD={metrics_row['asd']:.4f}"
    )
    axes[5].axis("off")

    plt.suptitle(
        f"{case_id} | representative slice comparison | red=Pred, green/lime=GT",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================

# =========================================================
def prepare_temp_input_from_cases(cases_dir: Path, temp_input_dir: Path):
    temp_input_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    case_dirs = sorted([p for p in cases_dir.iterdir() if p.is_dir()])

    for case_dir in case_dirs:
        case_id = case_dir.name
        image_path, mask_path = find_image_mask(case_dir)

       
        nnunet_input = temp_input_dir / f"{case_id}_0000.nii.gz"
        shutil.copy2(image_path, nnunet_input)

        rows.append({
            "case_id": case_id,
            "case_dir": str(case_dir),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "nnunet_input_path": str(nnunet_input),
        })

    return pd.DataFrame(rows)


def run_one_dataset_inference(dataset_name: str, cases_dir: Path):
    if not cases_dir.exists():
        raise FileNotFoundError(f"cases_dir does not exist: {cases_dir}")

    dataset_out_root = GLOBAL_OUTPUT_ROOT / dataset_name
    temp_input_dir = dataset_out_root / "temp_nnunet_input"
    pred_output_dir = dataset_out_root / "pred_from_best_fold"
    vis_dir = dataset_out_root / "visualizations"

    if temp_input_dir.exists():
        shutil.rmtree(temp_input_dir)
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"📦 Starting dataset: {dataset_name}")
    print("=" * 80)
    print(f"cases_dir      : {cases_dir}")
    print(f"temp_input_dir : {temp_input_dir}")
    print(f"pred_output_dir: {pred_output_dir}")
    print("=" * 80)

    mapping_df = prepare_temp_input_from_cases(cases_dir, temp_input_dir)
    safe_to_csv(mapping_df, dataset_out_root / "temp_input_mapping.csv")

    if RUN_INFERENCE:
        model_training_output_dir, fold, checkpoint_name = parse_checkpoint_path(BEST_CHECKPOINT_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=torch.cuda.is_available(),
            device=device,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True,
        )

        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=str(model_training_output_dir),
            use_folds=(fold,),
            checkpoint_name=checkpoint_name,
        )

        predictor.predict_from_files(
            list_of_lists_or_source_folder=str(temp_input_dir),
            output_folder_or_list_of_truncated_output_files=str(pred_output_dir),
            save_probabilities=False,
            overwrite=Config.overwrite,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    pred_files = sorted(pred_output_dir.glob("*.nii.gz"))
    if len(pred_files) == 0:
        raise RuntimeError(f"Prediction directory is empty: {pred_output_dir}")

    result_rows = []

    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=f"Evaluating {dataset_name}"):
        case_id = row["case_id"]
        case_dir = Path(row["case_dir"])
        image_path = Path(row["image_path"])
        gt_path = Path(row["mask_path"])
        pred_path = pred_output_dir / f"{case_id}.nii.gz"

        if not pred_path.exists():
            print(f"⚠️ Prediction file not found, skipping: {pred_path}")
            continue

        
        # 1) optionally copy pred back to the case folder
        if WRITE_BACK_PRED_TO_CASE_DIR:
            dst_pred_path = case_dir / f"{case_id}_pred.nii.gz"
            shutil.copy2(pred_path, dst_pred_path)
        else:
            dst_pred_path = pred_path


        pred_mask, _ = load_mask_as_binary(pred_path)
        gt_mask, gt_spacing = load_mask_as_binary(gt_path)

        if pred_mask.shape != gt_mask.shape:
                raise ValueError(f"{case_id} shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}")

        basic = compute_basic_metrics(gt_mask, pred_mask)
        surface = compute_hd95_and_asd(gt_mask, pred_mask, gt_spacing)

        metrics_row = {
            "dataset_name": dataset_name,
            "case_id": case_id,
            "case_dir": str(case_dir),
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "pred_path": str(dst_pred_path),
            **basic,
            **surface,
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
        }
        result_rows.append(metrics_row)

        
        try:
            image, _, _ = load_nifti(image_path)
            pred, _, _ = load_nifti(pred_path)
            gt, _, _ = load_nifti(gt_path)

            image_zyx = to_zyx(image)
            pred_zyx = to_zyx(pred)
            gt_zyx = to_zyx(gt)

            if image_zyx.shape == pred_zyx.shape == gt_zyx.shape:
                image_zyx_norm = normalize_image_for_display(image_zyx)
                rep_z = choose_representative_slice(gt_zyx, pred_zyx)

                case_vis_dir = vis_dir / case_id
                case_vis_dir.mkdir(parents=True, exist_ok=True)

                save_representative_comparison_figure(
                    image_zyx_norm=image_zyx_norm,
                    pred_zyx=pred_zyx,
                    gt_zyx=gt_zyx,
                    z=rep_z,
                    case_id=case_id,
                    metrics_row=metrics_row,
                    out_png=case_vis_dir / f"{case_id}_representative_slice_comparison.png",
                )

        except Exception as e:
            print(f"⚠️ {case_id} visualization failed: {e}")

    if len(result_rows) == 0:
        raise RuntimeError(f"{dataset_name} did not successfully process any case")

    df = pd.DataFrame(result_rows).sort_values("dice", ascending=False).reset_index(drop=True)

    csv_path = safe_to_csv(df, dataset_out_root / f"{dataset_name}_metrics.csv")
    xlsx_path = safe_to_excel(df, dataset_out_root / f"{dataset_name}_metrics.xlsx")

    summary = {
        "dataset_name": dataset_name,
        "cases_dir": str(cases_dir),
        "num_cases": int(len(df)),
        "mean_dice": float(df["dice"].mean()),
        "mean_iou": float(df["iou"].mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "mean_specificity": float(df["specificity"].mean()),
        "mean_hd95": float(df["hd95"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(df) > 0 else None,
        "mean_asd": float(df["asd"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(df) > 0 else None,
        "metrics_csv": str(csv_path),
        "metrics_xlsx": str(xlsx_path),
        "pred_output_dir": str(pred_output_dir),
        "visualization_dir": str(vis_dir),
    }
    safe_write_json(summary, dataset_out_root / f"{dataset_name}_summary.json")

    print(f"\n✅ {dataset_name} completed")
    print(f"📄 Metrics CSV : {csv_path}")
    print(f"📄 Metrics Excel: {xlsx_path}")
    print(f"📂 pred directory : {pred_output_dir}")
    print(f"📂 vis directory : {vis_dir}")

    return df, summary



def main():
    GLOBAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    select_best_fold()

    all_summaries = []
    all_dfs = []

    for ds in TARGET_DATASETS:
        if not ds.get("enabled", True):
            continue

        dataset_name = ds["name"]
        cases_dir = Path(ds["cases_dir"])

        df, summary = run_one_dataset_inference(dataset_name, cases_dir)
        all_dfs.append(df)
        all_summaries.append(summary)

    if len(all_dfs) > 0:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        safe_to_csv(merged_df, GLOBAL_OUTPUT_ROOT / "all_datasets_metrics_merged.csv")
        safe_to_excel(merged_df, GLOBAL_OUTPUT_ROOT / "all_datasets_metrics_merged.xlsx")

    final_summary = {
        "model_root": str(MODEL_ROOT),
        "selection_mode": FOLD_SELECTION_MODE,
        "manual_fold_override": MANUAL_FOLD_OVERRIDE,
        "selected_fold": SELECTED_FOLD,
        "selected_score": BEST_FOLD_SCORE,
        "selected_score_type": BEST_SCORE_TYPE,
        "selected_score_source": BEST_SCORE_SOURCE,
        "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
        "datasets": all_summaries,
    }
    safe_write_json(final_summary, GLOBAL_OUTPUT_ROOT / "final_run_summary.json")

    print("\n" + "=" * 80)
    print("All Done ✅")
    print("=" * 80)
    print(f"selected fold   : {SELECTED_FOLD}")
    print(f"selected score  : {BEST_FOLD_SCORE}")
    print(f"score type      : {BEST_SCORE_TYPE}")
    print(f"checkpoint_best : {BEST_CHECKPOINT_PATH}")
    print(f"global output   : {GLOBAL_OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()