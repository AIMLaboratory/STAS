# -*- coding: utf-8 -*-
"""
Generate leakage-free out-of-fold (OOF) nnU-Net predictions for the
617-patient development cohort.

IMPORTANT
---------
1) This is NOT a 5-fold ensemble.
2) Each development patient is predicted by ONE and only ONE model:
   the fold for which that patient was in the validation set.
3) The original experiment is READ-ONLY:
   - existing *_pred.nii.gz files are never modified/deleted;
   - existing Model 2 results are never touched;
   - all OOF outputs are written to a new, independent folder.
4) The script creates full provenance tables:
   source patient ID <-> RCM_xxxx <-> OOF fold <-> checkpoint <-> OOF mask.
"""

import os
import json
import csv
import shutil
import hashlib
import platform
import importlib.metadata
from pathlib import Path
from collections import Counter

import numpy as np
import SimpleITK as sitk
import torch


# ============================================================
# 0. USER CONFIGURATION
# ============================================================

from config import Config

DATASET_NAME = Config.dataset_folder_name()

# 617-patient development cohort after your study-specific preprocessing
# Each folder is expected to contain:
#   <source_case_id>_image.nii.gz
#   <source_case_id>_mask.nii.gz
CASES_DIR = Config.source_cases_dir

NNUNET_RAW = Config.nnunet_raw

NNUNET_PREPROCESSED = Config.nnunet_preprocessed

NNUNET_RESULTS = Config.nnunet_results

SPLITS_JSON = (
    Config.nnunet_preprocessed
    / Config.dataset_folder_name()
    / "splits_final.json"
)

CASE_MAPPING_JSON = Config.case_mapping_path()

MODEL_ROOT = (
    Config.nnunet_results
    / Config.dataset_folder_name()
    / f"{Config.trainer}__{Config.plans}__{Config.configuration}"
)


OUTPUT_ROOT = Path(
    os.environ.get(
        "OOF_OUTPUT_ROOT",
        str(Path(r"RECTAL_STAS_DATA\nnUNet_OOF_617_exploratory_v1")),
    )
)

# Keep the same checkpoint convention as the original experiment.
# NOTE: this means "checkpoint_best within each fold".
# It does NOT mean selecting one best-performing fold among the five.
CHECKPOINT_NAME = "checkpoint_best.pth"

FOLDS = Config.folds
EXPECTED_NUM_CASES = 617

# nnU-Net inference settings kept aligned with the previous prediction script
TILE_STEP_SIZE = 0.5
USE_GAUSSIAN = True
USE_MIRRORING = True
NUM_PROCESSES_PREPROCESSING = 2
NUM_PROCESSES_SEGMENTATION_EXPORT = 2

CUDA_VISIBLE_DEVICES = str(Config.gpu_id)

# Safety:
# False = never overwrite an already generated OOF prediction.
# Recommended for an exploratory/supplementary experiment.
OVERWRITE_EXISTING_OOF = False

# If True, calculate Dice against the manual mask for QC only.
# The GT mask is NEVER passed into nnU-Net inference.
CALCULATE_QC_DICE = True

# Optional heavy provenance step.
# Hashing large checkpoints can take extra time, so leave False unless desired.
HASH_CHECKPOINTS = False


# ============================================================
# 1. SET nnU-Net ENVIRONMENT BEFORE IMPORTING PREDICTOR
# ============================================================

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Reduce unnecessary CPU thread contention, consistent with previous workflow
os.environ.setdefault("nnUNet_n_proc_DA", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# ============================================================
# 2. GENERIC HELPERS
# ============================================================

def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found:\n{path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def path_is_same_or_inside(child: Path, parent: Path):
    child_r = child.resolve()
    parent_r = parent.resolve()
    return child_r == parent_r or parent_r in child_r.parents


def geometry_equal(a: sitk.Image, b: sitk.Image, atol=1e-5):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def read_binary_mask(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    binary = (arr > 0).astype(np.uint8)
    return binary, img


def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(gt, pred).sum())
    return float(2.0 * intersection / denom)


def write_csv_atomic(rows, path: Path, fieldnames):
    """
    Write via a temporary file and then replace, reducing the chance
    of leaving a partially-written manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


# ============================================================
# 3. LOAD AND VALIDATE CASE MAPPING
# ============================================================

def build_case_mapping(case_mapping_rows):
    """
    Expected fields from your case_mapping.json:
      source_case_id
      nnunet_case_id
      imageTr
      labelTr
      split
    """
    if not isinstance(case_mapping_rows, list):
        raise TypeError("case_mapping.json must contain a list.")

    nn_to_row = {}
    source_to_nn = {}

    for row in case_mapping_rows:
        if "source_case_id" not in row or "nnunet_case_id" not in row:
            raise KeyError(
                "Every case_mapping row must contain source_case_id and nnunet_case_id."
            )

        source_id = str(row["source_case_id"])
        nn_id = str(row["nnunet_case_id"])

        if nn_id in nn_to_row:
            raise RuntimeError(f"Duplicate nnunet_case_id in mapping: {nn_id}")
        if source_id in source_to_nn:
            raise RuntimeError(f"Duplicate source_case_id in mapping: {source_id}")

        nn_to_row[nn_id] = row
        source_to_nn[source_id] = nn_id

    if len(nn_to_row) != EXPECTED_NUM_CASES:
        raise RuntimeError(
            f"case_mapping.json contains {len(nn_to_row)} cases, "
            f"expected {EXPECTED_NUM_CASES}."
        )

    return nn_to_row, source_to_nn


# ============================================================
# 4. LOAD AND STRICTLY VALIDATE OOF SPLITS
# ============================================================

def validate_and_parse_splits(splits, all_nnunet_ids):
    """
    OOF rule:
      split[0]["val"] -> fold_0 model
      split[1]["val"] -> fold_1 model
      ...
      split[4]["val"] -> fold_4 model

    Returns:
      fold_val_ids: dict {fold: [RCM_xxxx, ...]}
      oof_fold_by_nn_id: dict {RCM_xxxx: fold}
    """
    if not isinstance(splits, list):
        raise TypeError("splits_final.json must contain a list.")

    if len(splits) != 5:
        raise RuntimeError(
            f"Expected exactly 5 folds, but splits_final.json has {len(splits)}."
        )

    all_nnunet_ids = set(all_nnunet_ids)
    fold_val_ids = {}
    oof_fold_by_nn_id = {}
    all_val_ids = []

    for fold in FOLDS:
        split = splits[fold]

        if "train" not in split or "val" not in split:
            raise KeyError(
                f"Fold {fold} must contain both 'train' and 'val' keys."
            )

        train_ids = [str(x) for x in split["train"]]
        val_ids = [str(x) for x in split["val"]]

        train_set = set(train_ids)
        val_set = set(val_ids)

        if train_set & val_set:
            overlap = sorted(train_set & val_set)
            raise RuntimeError(
                f"Fold {fold}: train/val overlap detected: {overlap[:10]}"
            )

        # Every fold should partition all 617 development cases
        fold_union = train_set | val_set
        missing = sorted(all_nnunet_ids - fold_union)
        unknown = sorted(fold_union - all_nnunet_ids)

        if missing:
            raise RuntimeError(
                f"Fold {fold}: {len(missing)} mapped cases are missing "
                f"from train+val. Examples: {missing[:10]}"
            )
        if unknown:
            raise RuntimeError(
                f"Fold {fold}: {len(unknown)} split IDs are absent from mapping. "
                f"Examples: {unknown[:10]}"
            )

        fold_val_ids[fold] = val_ids

        for nn_id in val_ids:
            if nn_id in oof_fold_by_nn_id:
                raise RuntimeError(
                    f"{nn_id} appears in validation for more than one fold: "
                    f"{oof_fold_by_nn_id[nn_id]} and {fold}"
                )
            oof_fold_by_nn_id[nn_id] = fold
            all_val_ids.append(nn_id)

    # Across all folds, every patient must occur in validation exactly once.
    counts = Counter(all_val_ids)
    bad_counts = {k: v for k, v in counts.items() if v != 1}

    if bad_counts:
        examples = list(bad_counts.items())[:10]
        raise RuntimeError(
            f"OOF validation membership is not unique. Examples: {examples}"
        )

    missing_from_all_val = sorted(all_nnunet_ids - set(all_val_ids))
    if missing_from_all_val:
        raise RuntimeError(
            f"{len(missing_from_all_val)} cases never appear in a validation fold. "
            f"Examples: {missing_from_all_val[:10]}"
        )

    if len(all_val_ids) != EXPECTED_NUM_CASES:
        raise RuntimeError(
            f"Total validation memberships = {len(all_val_ids)}, "
            f"expected {EXPECTED_NUM_CASES}."
        )

    return fold_val_ids, oof_fold_by_nn_id


# ============================================================
# 5. PATH AND MODEL PREFLIGHT
# ============================================================

def locate_source_image(source_case_id: str):
    case_dir = CASES_DIR / source_case_id
    expected = case_dir / f"{source_case_id}_image.nii.gz"

    if expected.exists():
        return expected

    # Strict fallback: allow exactly one *_image.nii.gz in the case folder
    candidates = sorted(case_dir.glob("*_image.nii.gz")) if case_dir.exists() else []
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot uniquely locate image.\n"
        f"Expected: {expected}\n"
        f"Candidates found: {len(candidates)}"
    )


def locate_gt_mask(source_case_id: str):
    case_dir = CASES_DIR / source_case_id
    expected = case_dir / f"{source_case_id}_mask.nii.gz"

    if expected.exists():
        return expected

    candidates = sorted(case_dir.glob("*_mask.nii.gz")) if case_dir.exists() else []
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot uniquely locate GT mask for {source_case_id}.\n"
        f"Expected: {expected}\n"
        f"Candidates found: {len(candidates)}"
    )


def checkpoint_path(fold: int):
    return MODEL_ROOT / f"fold_{fold}" / CHECKPOINT_NAME


def strict_preflight(nn_to_row, fold_val_ids):
    """
    No prediction starts until all 617 IDs, source images and checkpoints pass.
    """
    if not CASES_DIR.exists():
        raise FileNotFoundError(f"CASES_DIR not found:\n{CASES_DIR}")

    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"MODEL_ROOT not found:\n{MODEL_ROOT}")

    # Critical safety rule: output must not be the original case directory
    # and must not sit inside it.
    if path_is_same_or_inside(OUTPUT_ROOT, CASES_DIR):
        raise RuntimeError(
            "SAFETY STOP: OUTPUT_ROOT is equal to or inside CASES_DIR.\n"
            "OOF output must be stored in an independent folder."
        )

    # Confirm all checkpoints exist
    for fold in FOLDS:
        ckpt = checkpoint_path(fold)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fold}:\n{ckpt}"
            )

    # Confirm all source images and GT masks exist before starting.
    missing_images = []
    missing_masks = []

    for nn_id, row in nn_to_row.items():
        source_id = str(row["source_case_id"])
        try:
            locate_source_image(source_id)
        except Exception as e:
            # do not leak source_case_id; report anonymous case + reason
            missing_images.append((nn_id, str(e).splitlines()[0]))

        if CALCULATE_QC_DICE:
            try:
                locate_gt_mask(source_id)
            except Exception as e:
                missing_masks.append((source_id, str(e)))

    if missing_images:
        raise RuntimeError(
            f"Preflight failed: {len(missing_images)} source images missing/ambiguous.\n"
            f"First examples:\n{missing_images[:5]}"
        )

    if missing_masks:
        raise RuntimeError(
            f"Preflight failed: {len(missing_masks)} GT masks missing/ambiguous.\n"
            f"First examples:\n{missing_masks[:5]}"
        )

    # Confirm fold counts from the actual split file.
    fold_counts = {fold: len(fold_val_ids[fold]) for fold in FOLDS}

    if sum(fold_counts.values()) != EXPECTED_NUM_CASES:
        raise RuntimeError(
            f"Fold validation counts sum to {sum(fold_counts.values())}, "
            f"expected {EXPECTED_NUM_CASES}."
        )

    return fold_counts


# ============================================================
# 6. PROVENANCE / FIXED ASSIGNMENT TABLE
# ============================================================

ASSIGNMENT_FIELDS = [
    "anonymous_case_id",
    "oof_fold",
    "checkpoint_name",
    "checkpoint_path",
    "source_image",
    "gt_mask",
    "original_nnunet_imageTr",
    "original_nnunet_labelTr",
    "oof_mask",
]


def build_assignment_table(nn_to_row, oof_fold_by_nn_id):
    rows = []

    for nn_id in sorted(nn_to_row.keys()):
        row = nn_to_row[nn_id]
        source_id = str(row["source_case_id"])         
        fold = int(oof_fold_by_nn_id[nn_id])            

        source_img = locate_source_image(source_id)
        gt_mask = locate_gt_mask(source_id) if CALCULATE_QC_DICE else ""

        
        oof_mask = (
            OUTPUT_ROOT
            / "oof_masks_by_patient"
            / f"{nn_id}_pred_oof.nii.gz"
        )

        rows.append(
            {
                "anonymous_case_id": nn_id,
                "oof_fold": fold,
                "checkpoint_name": CHECKPOINT_NAME,
                "checkpoint_path": str(checkpoint_path(fold)),
                "source_image": str(source_img),
                "gt_mask": str(gt_mask) if gt_mask else "",
                "original_nnunet_imageTr": str(row.get("imageTr", "")),
                "original_nnunet_labelTr": str(row.get("labelTr", "")),
                "oof_mask": str(oof_mask),
            }
        )

    return rows


def save_provenance_snapshot(assignment_rows, fold_counts):
    provenance_dir = OUTPUT_ROOT / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    # Freeze exact split/mapping files used for this OOF run.
    shutil.copy2(
        SPLITS_JSON,
        provenance_dir / "splits_final_USED_FOR_OOF.json"
    )
   

    write_csv_atomic(
        assignment_rows,
        provenance_dir / "oof_fold_assignment_617.csv",
        ASSIGNMENT_FIELDS,
    )

    # Snapshot the original *_pred.nii.gz files.
    # This is READ-ONLY documentation; nothing is modified.
    original_pred_rows = []
    for p in sorted(CASES_DIR.glob("*/*_pred.nii.gz")):
        original_pred_rows.append(
            {
                "original_pred_path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_time_ns": p.stat().st_mtime_ns,
            }
        )

    if original_pred_rows:
        write_csv_atomic(
            original_pred_rows,
            provenance_dir / "original_pred_files_snapshot.csv",
            ["original_pred_path", "size_bytes", "modified_time_ns"],
        )

    nnunet_version = importlib.metadata.version("nnunetv2")

    run_info = {
        "experiment": "OOF segmentation predictions for Model 2 sensitivity/exploratory analysis",
        "not_an_ensemble": True,
        "oof_rule": (
            "Each development patient is predicted only by the fold "
            "for which that patient is in the validation partition."
        ),
        "expected_cases": EXPECTED_NUM_CASES,
        "fold_validation_counts": fold_counts,
        "checkpoint_name_within_each_fold": CHECKPOINT_NAME,
        "nnunetv2_version": nnunet_version,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "tile_step_size": TILE_STEP_SIZE,
        "use_gaussian": USE_GAUSSIAN,
        "use_mirroring": USE_MIRRORING,
        "save_probabilities": False,
        "overwrite_existing_oof": OVERWRITE_EXISTING_OOF,
        "splits_sha256": sha256_file(SPLITS_JSON),
        "case_mapping_sha256": sha256_file(CASE_MAPPING_JSON),
    }

    if HASH_CHECKPOINTS:
        run_info["checkpoint_sha256"] = {
            f"fold_{fold}": sha256_file(checkpoint_path(fold))
            for fold in FOLDS
        }
    else:
        run_info["checkpoint_metadata"] = {
            f"fold_{fold}": {
                "path": str(checkpoint_path(fold)),
                "size_bytes": checkpoint_path(fold).stat().st_size,
                "modified_time_ns": checkpoint_path(fold).stat().st_mtime_ns,
            }
            for fold in FOLDS
        }

    with (provenance_dir / "run_provenance.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)


# ============================================================
# 7. nnU-Net PREDICTOR: ONE FOLD ONLY
# ============================================================

def create_predictor_for_one_fold(fold: int):
    """
    IMPORTANT:
    use_folds=(fold,) contains exactly ONE fold.
    Therefore this is NOT a fold ensemble.
    """
    ckpt = checkpoint_path(fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=USE_GAUSSIAN,
        use_mirroring=USE_MIRRORING,
        perform_everything_on_device=torch.cuda.is_available(),
        device=device,
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_ROOT),
        use_folds=(fold,),
        checkpoint_name=ckpt.name,
    )

    return predictor


# ============================================================
# 8. PREDICT ONE OOF FOLD
# ============================================================

def predict_one_fold(fold, val_nn_ids, nn_to_row):
    fold_raw_dir = OUTPUT_ROOT / "raw_predictions_by_fold" / f"fold_{fold}"
    fold_raw_dir.mkdir(parents=True, exist_ok=True)

    input_list = []
    output_list = []

    for nn_id in val_nn_ids:
        source_id = str(nn_to_row[nn_id]["source_case_id"])
        source_img = locate_source_image(source_id)

        
        output_trunc = fold_raw_dir / nn_id

        input_list.append([str(source_img)])
        output_list.append(str(output_trunc))

    predictor = create_predictor_for_one_fold(fold)

    print("\n" + "=" * 100)
    print(
        f"OOF FOLD {fold}: predicting {len(val_nn_ids)} validation patients "
        f"with fold_{fold}/{CHECKPOINT_NAME}"
    )
    print("THIS IS SINGLE-FOLD OOF INFERENCE, NOT AN ENSEMBLE.")
    print("=" * 100)

    predictor.predict_from_files(
        input_list,
        output_list,
        save_probabilities=False,
        overwrite=OVERWRITE_EXISTING_OOF,
        num_processes_preprocessing=NUM_PROCESSES_PREPROCESSING,
        num_processes_segmentation_export=NUM_PROCESSES_SEGMENTATION_EXPORT,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

  
    missing = []
    for nn_id in val_nn_ids:
        expected = fold_raw_dir / f"{nn_id}.nii.gz"
        if not expected.exists():
            missing.append(str(expected))

    if missing:
        raise RuntimeError(
            f"Fold {fold}: {len(missing)} expected raw predictions are missing.\n"
            f"Examples: {missing[:5]}"
        )


# ============================================================
# 9. FINALIZE, RENAME TO SOURCE PATIENT ID, QC
# ============================================================

RESULT_FIELDS = [
    "anonymous_case_id",
    "source_case_id",
    "nnunet_case_id",
    "oof_fold",
    "checkpoint_name",
    "checkpoint_path",
    "source_image",
    "gt_mask",
    "raw_rcm_prediction",
    "oof_mask",
    "pred_geometry_matches_source_image",
    "pred_geometry_matches_gt",
    "prediction_unique_labels",
    "pred_voxels",
    "gt_voxels",
    "dice",
    "oof_mask_sha256",
]


def finalize_one_case(nn_id, fold, nn_to_row):
    source_id = str(nn_to_row[nn_id]["source_case_id"])

    source_img_path = locate_source_image(source_id)
    gt_path = locate_gt_mask(source_id) if CALCULATE_QC_DICE else None

    raw_pred = (
        OUTPUT_ROOT
        / "raw_predictions_by_fold"
        / f"fold_{fold}"
        / f"{nn_id}.nii.gz"
    )

    if not raw_pred.exists():
        raise FileNotFoundError(f"Raw OOF prediction not found:\n{raw_pred}")

    final_dir = OUTPUT_ROOT / "oof_masks_by_patient"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_pred = final_dir / f"{nn_id}_pred_oof.nii.gz"

    if final_pred.exists():
        if OVERWRITE_EXISTING_OOF:
            shutil.copy2(raw_pred, final_pred)
        # else: keep existing
    else:
        shutil.copy2(raw_pred, final_pred)

    source_img = sitk.ReadImage(str(source_img_path))
    pred_arr, pred_img = read_binary_mask(final_pred)

    pred_raw_arr = sitk.GetArrayFromImage(pred_img)
    unique_labels = sorted(int(x) for x in np.unique(pred_raw_arr))

    geometry_source_ok = geometry_equal(pred_img, source_img)

    gt_geometry_ok = ""
    gt_voxels = ""
    dsc = ""

    if CALCULATE_QC_DICE:
        gt_arr, gt_img = read_binary_mask(gt_path)

        if pred_arr.shape != gt_arr.shape:
            raise RuntimeError(
                f"Shape mismatch for anonymous case {nn_id}: "
                f"pred={pred_arr.shape}, gt={gt_arr.shape}"
            )

        gt_geometry_ok = geometry_equal(pred_img, gt_img)
        gt_voxels = int(gt_arr.sum())
        dsc = dice_score(gt_arr, pred_arr)

    if CALCULATE_QC_DICE and gt_geometry_ok is not True:
        raise RuntimeError(
            f"Geometry mismatch between OOF prediction and GT mask: {nn_id}"
        )
    if not geometry_source_ok:
        raise RuntimeError(
        f"Geometry mismatch between OOF prediction and source image: {nn_id}"
    )

    if not set(unique_labels).issubset({0, 1}):
        raise RuntimeError(
            f"Unexpected prediction labels for {nn_id}: {unique_labels}"
        )

    return {
        "anonymous_case_id": nn_id,
        "source_case_id": source_id,     
        "nnunet_case_id": nn_id,       
        "oof_fold": fold,
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_path": str(checkpoint_path(fold)),
        "source_image": str(source_img_path),
        "gt_mask": str(gt_path) if gt_path else "",
        "raw_rcm_prediction": str(raw_pred),
        "oof_mask": str(final_pred),
        "pred_geometry_matches_source_image": bool(geometry_source_ok),
        "pred_geometry_matches_gt": gt_geometry_ok,
        "prediction_unique_labels": json.dumps(unique_labels),
        "pred_voxels": int(pred_arr.sum()),
        "gt_voxels": gt_voxels,
        "dice": dsc,
        "oof_mask_sha256": sha256_file(final_pred),
    }

# ============================================================
# 10. FINAL INTEGRITY CHECK
# ============================================================

def validate_final_manifest(result_rows, nn_to_row, oof_fold_by_nn_id):
    if len(result_rows) != EXPECTED_NUM_CASES:
        raise RuntimeError(
            f"Final manifest contains {len(result_rows)} cases, "
            f"expected {EXPECTED_NUM_CASES}."
        )

    source_ids = [r["source_case_id"] for r in result_rows]
    nn_ids = [r["nnunet_case_id"] for r in result_rows]

    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate source_case_id detected in final manifest.")

    if len(nn_ids) != len(set(nn_ids)):
        raise RuntimeError("Duplicate nnunet_case_id detected in final manifest.")

    expected_nn_ids = set(nn_to_row.keys())
    result_nn_ids = set(nn_ids)

    missing = sorted(expected_nn_ids - result_nn_ids)
    extra = sorted(result_nn_ids - expected_nn_ids)

    if missing or extra:
        raise RuntimeError(
            f"Final case-set mismatch. missing={missing[:10]}, extra={extra[:10]}"
        )

    # Confirm every row used exactly its assigned OOF fold.
    wrong_fold = []
    for row in result_rows:
        nn_id = row["nnunet_case_id"]
        expected_fold = int(oof_fold_by_nn_id[nn_id])
        actual_fold = int(row["oof_fold"])
        if expected_fold != actual_fold:
            wrong_fold.append((nn_id, expected_fold, actual_fold))

    if wrong_fold:
        raise RuntimeError(
            f"OOF fold assignment mismatch detected: {wrong_fold[:10]}"
        )

    # Confirm all 617 final masks physically exist.
    missing_masks = [
        r["oof_mask"]
        for r in result_rows
        if not Path(r["oof_mask"]).exists()
    ]
    if missing_masks:
        raise RuntimeError(
            f"{len(missing_masks)} final OOF masks are missing. "
            f"Examples: {missing_masks[:5]}"
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print("\n" + "=" * 100)
    print("LEAKAGE-FREE OOF nnU-Net PREDICTION FOR 617 DEVELOPMENT PATIENTS")
    print("NOT A 5-FOLD ENSEMBLE")
    print("=" * 100)

    # Load exact training split and mapping
    case_mapping_rows = read_json(CASE_MAPPING_JSON)
    splits = read_json(SPLITS_JSON)

    nn_to_row, source_to_nn = build_case_mapping(case_mapping_rows)

    fold_val_ids, oof_fold_by_nn_id = validate_and_parse_splits(
        splits,
        all_nnunet_ids=nn_to_row.keys(),
    )

    fold_counts = strict_preflight(nn_to_row, fold_val_ids)

    # Print what will happen BEFORE creating predictions.
    print("\nValidated fold assignment:")
    for fold in FOLDS:
        print(
            f"  fold {fold}: "
            f"{len(splits[fold]['train'])} train + "
            f"{len(splits[fold]['val'])} val; "
            f"OOF prediction uses fold_{fold}/{CHECKPOINT_NAME}"
        )

    print(f"\nTotal unique OOF patients: {len(oof_fold_by_nn_id)}")
    print(f"Original development directory (READ ONLY): {CASES_DIR}")
    print(f"New OOF output directory: {OUTPUT_ROOT}")

    # Show first 5 source-patient examples for each fold
    print("\nFirst five OOF anonymous case IDs per fold:")
    for fold in FOLDS:
        examples = [str(nn_id) for nn_id in fold_val_ids[fold][:5]]
        print(f"  fold {fold}: " + "; ".join(examples))

    # Create output root only after all strict checks above pass.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    assignment_rows = build_assignment_table(
        nn_to_row,
        oof_fold_by_nn_id,
    )
    save_provenance_snapshot(assignment_rows, fold_counts)

    # --------------------------------------------------------
    # OOF inference:
    # fold k predicts ONLY split[k]["val"].
    # --------------------------------------------------------
    all_results = []

    for fold in FOLDS:
        val_nn_ids = fold_val_ids[fold]

        predict_one_fold(
            fold=fold,
            val_nn_ids=val_nn_ids,
            nn_to_row=nn_to_row,
        )

        # Convert RCM-named raw predictions to source-patient-named OOF masks.
        fold_results = []
        for nn_id in val_nn_ids:
            row = finalize_one_case(
                nn_id=nn_id,
                fold=fold,
                nn_to_row=nn_to_row,
            )
            fold_results.append(row)
            all_results.append(row)

        # Save partial manifest after each completed fold.
        partial_path = OUTPUT_ROOT / "manifests" / "oof_manifest_PARTIAL.csv"
        write_csv_atomic(
            sorted(all_results, key=lambda x: x["nnunet_case_id"]),
            partial_path,
            RESULT_FIELDS,
        )

        fold_manifest = (
            OUTPUT_ROOT
            / "manifests"
            / f"oof_manifest_fold_{fold}.csv"
        )
        write_csv_atomic(
            sorted(fold_results, key=lambda x: x["nnunet_case_id"]),
            fold_manifest,
            RESULT_FIELDS,
        )

    # --------------------------------------------------------
    # Final integrity audit
    # --------------------------------------------------------
    all_results = sorted(
        all_results,
        key=lambda x: x["nnunet_case_id"],
    )

    validate_final_manifest(
        all_results,
        nn_to_row,
        oof_fold_by_nn_id,
    )

    final_manifest_csv = (
        OUTPUT_ROOT
        / "manifests"
        / "oof_manifest_617_FINAL.csv"
    )
    write_csv_atomic(
        all_results,
        final_manifest_csv,
        RESULT_FIELDS,
    )

    # Separate classifier-friendly index:
    # downstream classification code only needs source_case_id + oof_mask,
    # but we retain the fold/RCM fields for traceability.
    classifier_index_fields = [
    "source_case_id",
    "nnunet_case_id",
    "oof_fold",
    "oof_mask",
]

    classifier_rows = [
        {k: row[k] for k in classifier_index_fields}
        for row in all_results
    ]

    classifier_index_csv = (
        OUTPUT_ROOT
        / "classifier_input"
        / "model2_oof_mask_index_617.csv"
    )
    write_csv_atomic(
        classifier_rows,
        classifier_index_csv,
        classifier_index_fields,
    )

    # Final summary
    dice_values = [
        float(r["dice"])
        for r in all_results
        if r["dice"] != ""
    ]

    summary = {
        "status": "SUCCESS",
        "not_an_ensemble": True,
        "num_cases": len(all_results),
        "fold_validation_counts": {
            str(fold): len(fold_val_ids[fold])
            for fold in FOLDS
        },
        "oof_rule": (
            "Each case was predicted by only the nnU-Net fold for which "
            "that case belonged to the validation subset."
        ),
        "original_results_modified": False,
        "original_cases_dir": str(CASES_DIR),
        "oof_output_root": str(OUTPUT_ROOT),
        "final_manifest": str(final_manifest_csv),
        "classifier_mask_index": str(classifier_index_csv),
        "checkpoint_name_within_each_fold": CHECKPOINT_NAME,
    }

    if dice_values:
        summary.update(
            {
                "mean_oof_dice": float(np.mean(dice_values)),
                "median_oof_dice": float(np.median(dice_values)),
                "min_oof_dice": float(np.min(dice_values)),
                "max_oof_dice": float(np.max(dice_values)),
            }
        )

    summary_path = OUTPUT_ROOT / "oof_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("OOF PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(f"617 OOF masks:")
    print(f"  {OUTPUT_ROOT / 'oof_masks_by_patient'}")
    print("\nFull traceability manifest:")
    print(f"  {final_manifest_csv}")
    print("\nClassifier-friendly OOF mask index:")
    print(f"  {classifier_index_csv}")

    if dice_values:
        print(f"\nMean OOF Dice: {np.mean(dice_values):.4f}")

    print("\nORIGINAL *_pred.nii.gz FILES WERE NOT MODIFIED.")
    print("NO OOF MASK WAS WRITTEN INTO THE ORIGINAL CASE DIRECTORIES.")
    print("THIS RUN DID NOT PERFORM A 5-FOLD ENSEMBLE.")
    print("=" * 100)


if __name__ == "__main__":
    # Required on Windows when nnU-Net uses multiprocessing workers.
    main()
