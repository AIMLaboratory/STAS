# -*- coding: utf-8 -*-
r"""
FULL PAPER-STYLE INDEPENDENT TEST: ResNet34 Models 4 / 5 / 6.

  Model 4: ResNet34, T2WI + manual tumor mask (*_mask.nii.gz)
  Model 5: ResNet34, T2WI + nnU-Net predicted mask (*_pred.nii.gz)
  Model 6: ResNet34, T2WI only

All 3 models were trained on exactly the same 617-patient 5-fold split.
Model 5 trained on 617 OOF masks; INDEPENDENT TEST cases use their original
nnU-Net *_pred.nii.gz masks, never development OOF masks.

Default: each model uses its highest-validation-AUC fold and its frozen
validation-derived threshold. Option --same_fold N uses fold N for ALL 3 models,
with each fold's own best epoch/threshold (a policy to predeclare before test).
Never choose a fold or decision threshold from test performance.

Each test cohort: IT1(47), IT2(31), ET(83). Per-case predictions, AUC and
bootstrap CIs, paired DeLong comparisons, radiologists, plots, Excel audits.
Historical DenseNet121 models and old predictions are NOT loaded or overwritten.
"""

import sys
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# 0. Local project imports
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import Config
from data_loader import (
    RectalMRIClassification3DDataset,
    load_samples_from_test_csv,
)
from models import create_model


# ============================================================
# 1. Global plotting setup
#    Keep the same visual convention as the original scripts.
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


# ============================================================
# 2. Shared ResNet34 Models 4,5,6 training root / independent test output
# ============================================================
ROOT = Path(r"RECTAL_STAS_DATA\ResNet34_Models4_5_6_Shared5Fold_OOF_v1")
TRAINING_PROVENANCE = ROOT / "provenance" / "TRAINING_DESIGN.json"
TRAINING_COMPLETION = ROOT / "TRAINING_COMPLETE_ALL_MODELS_4_5_6.json"
NEW_RESULT_ROOT = ROOT / "paper_style_test_results_IT1_IT2_ET_Models4_5_6"
ALLOW_OVERWRITE_NEW_RESULTS = False
FORCED_SHARED_FOLD = None  # CLI --same_fold overrides; do not choose by test results.
MODEL_EXPECTED_MODES = {
    "Model 4": "image_plus_gt_mask",
    "Model 5": "image_plus_pred_mask",
    "Model 6": "image_only",
}


# ============================================================
# 4. TRUE manuscript test cohorts
#    Recovered from the historical prediction/evaluation scripts.
# ============================================================

TEST_SPECS = {
        "IT1": {
        "title": "IT1",
        "description": "Internal test cohort A",
        "expected_n": 47,
        "cases_dir": Path(
            r"RECTAL_STAS_DATA\split_dataset_tstage_fixedtest10\test\cases"
        ),
        "label_file": Path(
            r"RECTAL_STAS_DATA\test_cohort_A\fixed_test_labels.csv"
        ),
        "label_format": "FIXED_CSV",
        "senior_file": Path(
            r"RECTAL_STAS_DATA\test_cohort_A\senior_radiologist.csv"
        ),
        "junior_file": Path(
            r"RECTAL_STAS_DATA\test_cohort_A\junior_radiologist.csv"
        ),
    },

    "IT2": {
        "title": "IT2",
        "description": "Internal test cohort B",
        "expected_n": 31,
        "cases_dir": Path(
            r"RECTAL_STAS_DATA\test_cohort_B\cases"
        ),
        "label_file": Path(
            r"RECTAL_STAS_DATA\test_cohort_B\labels.xlsx"
        ),
        "label_format": "FOLDERNAME_TSTAGE_XLSX",
    },

    "ET": {
        "title": "ET",
        "description": "External test cohort",
        "expected_n": 83,
        "cases_dir": Path(
            r"RECTAL_STAS_DATA\test_cohort_C\cases"
        ),
        "label_file": Path(
            r"RECTAL_STAS_DATA\test_cohort_C\labels.xlsx"
        ),
        "label_format": "FOLDERNAME_TSTAGE_XLSX",
    },
}


# ============================================================
# 5. Display labels
# ============================================================

MODEL_DISPLAY = {
    "Model 4": "Model 4 (ResNet34, T2WI + Manual masks)",
    "Model 5": "Model 5 (ResNet34, T2WI + Predicted masks)",
    "Model 6": "Model 6 (ResNet34, T2WI only)",
}
SUMMARY_METHOD = {
    "Model 4": "ResNet34 Model 4 (Manual Mask)",
    "Model 5": "ResNet34 Model 5 (Pred Mask)",
    "Model 6": "ResNet34 Model 6 (T2WI Only)",
}
INPUT_MODE_FOR_MODEL = dict(MODEL_EXPECTED_MODES)


# ============================================================
# 6. General utilities
# ============================================================

def print_header(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def require_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def require_dir(path: Path, description: str):
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 7. T-stage helpers
# ============================================================

STAGE_SCORE = {
    "Tis": 0,
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T3a": 3,
    "T3b": 3,
    "T3c": 3,
    "T3d": 3,
    "T4": 4,
    "T4a": 4,
    "T4b": 4,
}

EARLY = {
    "Tis", "T0", "T1", "T2"
}

ADVANCED = {
    "T3", "T3a", "T3b", "T3c", "T3d",
    "T4", "T4a", "T4b"
}


def extract_stage(text):
    if pd.isna(text):
        return None

    s = (
        str(text)
        .strip()
        .upper()
        .replace("YPT", "T")
        .replace("PT", "T")
        .replace("YT", "T")
    )

    if s and s[0].isdigit():
        s = "T" + s

    m = re.match(
        r"(TIS|T0|T1|T2|T3[ABCD]?|T4[AB]?)",
        s,
    )

    if not m:
        return None

    raw = m.group(1)

    if raw == "TIS":
        return "Tis"

    return raw[0] + raw[1:].lower()


def stage_to_binary(stage):
    if stage in EARLY:
        return 0
    if stage in ADVANCED:
        return 1
    return None


def stage_to_score(stage):
    return STAGE_SCORE.get(stage)


def map_t_stage_to_binary(raw_value):
    """
    Robust pathology mapping for IT2/ET XLSX:
      T1/T2 -> 0
      T3/T4 -> 1
    """
    if pd.isna(raw_value):
        return None

    if isinstance(
        raw_value,
        (int, float, np.integer, np.floating),
    ):
        v = int(raw_value)
        if v in (1, 2):
            return 0
        if v in (3, 4):
            return 1
        return None

    stage = extract_stage(raw_value)

    if stage is not None:
        return stage_to_binary(stage)

    m = re.search(
        r"([1-4])",
        str(raw_value),
    )

    if not m:
        return None

    v = int(m.group(1))

    if v in (1, 2):
        return 0
    if v in (3, 4):
        return 1

    return None


def map_t_stage_to_score(raw_value):
    if pd.isna(raw_value):
        return None

    if isinstance(
        raw_value,
        (int, float, np.integer, np.floating),
    ):
        v = int(raw_value)
        return v if v in (1, 2, 3, 4) else None

    stage = extract_stage(raw_value)

    if stage is not None:
        return stage_to_score(stage)

    m = re.search(
        r"([1-4])",
        str(raw_value),
    )

    if not m:
        return None

    v = int(m.group(1))

    return v if v in (1, 2, 3, 4) else None


def normalize_case_id_4digit(raw_id):
    """
    Normalize folder case IDs to 4-digit zero-padded form:
      11  -> 0011
      69  -> 0069
      543 -> 0543
    """
    if pd.isna(raw_id):
        raise ValueError(f"Missing case ID: {raw_id}")

    s = str(raw_id).strip()
    

    if re.fullmatch(r"\d+\.0+", s):
        s = str(int(float(s)))
    elif re.fullmatch(r"\d+", s):
        s = str(int(s))
    else:
        nums = re.findall(
            r"\d+",
            s,
        )
        if not nums:
            raise ValueError(
               f"Unexpected case ID: {raw_id}"
            )
        s = str(
            int(
                max(
                    nums,
                    key=len,
                )
            )
        )

    return s.zfill(4)


def radiologist_merge_id(raw_id):
    """
    Normalize folder IDs to an integer-like string for matching
    the two radiologist CSVs with model folder IDs.
    """
    folder_id = normalize_case_id_4digit(raw_id)
    return str(int(folder_id))


# ============================================================
# 8. Load three NEW ResNet34 selections and audit SAME patient-level split
# ============================================================
def check_training_provenance():
    require_file(TRAINING_PROVENANCE, "Shared-fold training provenance")
    require_file(TRAINING_COMPLETION, "All three ResNet34 models trained")
    p = json.loads(TRAINING_PROVENANCE.read_text(encoding="utf-8"))
    completed = json.loads(TRAINING_COMPLETION.read_text(encoding="utf-8"))
    if p.get("architecture") != "resnet34_3d":
        raise RuntimeError("Training provenance has the wrong architecture")
    if p.get("n_development") != 617 or len(p.get("folds", [])) != 5:
        raise RuntimeError("The ResNet34 run does not have 617 cases and 5 folds")
    if not p.get("same_patient_folds_across_models"):
        raise RuntimeError("Shared patient-fold guarantee missing")
    if set(completed.get("models", {})) != {"4", "5", "6"}:
        raise RuntimeError("Training completion lacks one of Models 4/5/6")
    shared = Path(p["shared_classifier_split"])
    index = Path(p["oof_index"])
    require_file(shared, "Original DenseNet121 Model 2 CV split")
    require_file(index, "nnU-Net original OOF index")
    if sha256_file(shared) != p["shared_classifier_split_sha256"]:
        raise RuntimeError("Shared classification fold JSON changed after training")
    if sha256_file(index) != p["oof_index_sha256"]:
        raise RuntimeError("OOF mask index CSV changed after training")
    for path_key, hash_key, meaning in (
        ("densenet_oof_index_used", "densenet_oof_index_used_sha256", "DenseNet121 OOF index snapshot"),
        ("densenet_split_used", "densenet_split_used_sha256", "DenseNet121 split snapshot"),
    ):
        baseline = Path(p[path_key])
        require_file(baseline, meaning)
        if sha256_file(baseline) != p[hash_key]:
            raise RuntimeError(meaning + " changed after training")
    if p["densenet_oof_index_used_sha256"] != p["oof_index_sha256"]:
        raise RuntimeError("DenseNet121/ResNet34 OOF-index SHA256 mismatch")
    if p["densenet_split_used_sha256"] != p["shared_classifier_split_sha256"]:
        raise RuntimeError("DenseNet121/ResNet34 shared-fold SHA256 mismatch")
    seen = []
    for expected_fold, fold in enumerate(p["folds"]):
        if int(fold["fold"]) != expected_fold:
            raise RuntimeError("Invalid fold order in training provenance")
        tr, va = fold["train_case_ids"], fold["val_case_ids"]
        if set(tr) & set(va):
            raise RuntimeError(f"Overlapping train/val patients in fold {expected_fold}")
        seen.extend(va)
    if len(seen) != 617 or len(set(seen)) != 617:
        raise RuntimeError("Five folds do not cover 617 unique validation patients")
    return p


def model_spec_from_training(model_id, provenance):
    label = f"Model {model_id}"
    mode = MODEL_EXPECTED_MODES[label]
    model_root = ROOT / f"Model{model_id}"
    selected_json = model_root / "selected_model" / "TRAINING_COMPLETE_selected_model.json"
    require_file(selected_json, label + " training completion")
    selected = json.loads(selected_json.read_text(encoding="utf-8"))
    if (selected.get("model_id") != model_id or
        selected.get("model_name") != "resnet34_3d" or
        selected.get("input_mode") != mode or
        selected.get("all_five_classifier_folds_trained") is not True):
        raise RuntimeError(f"{label}: training-completion metadata mismatch")
    if selected.get("shared_classifier_split_sha256") != provenance["shared_classifier_split_sha256"]:
        raise RuntimeError(f"{label}: shared fold SHA256 mismatch")
    if selected.get("oof_index_sha256") != provenance["oof_index_sha256"]:
        raise RuntimeError(f"{label}: OOF index SHA256 mismatch")

    if FORCED_SHARED_FOLD is None:
        fold = int(selected["selected_fold"])
        checkpoint = Path(selected["selected_best_model_path"])
        threshold = float(selected["selected_threshold"])
        validation_auc = float(selected["selected_validation_auc"])
        basis = "highest development validation AUC only"
        selection_source = selected_json
    else:
        fold = int(FORCED_SHARED_FOLD)
        rec = model_root / "fold_summaries" / f"fold_{fold}_training_result.json"
        require_file(rec, f"{label} required common fold {fold} result")
        r = json.loads(rec.read_text(encoding="utf-8"))
        if (int(r["model_id"]) != model_id or int(r["fold"]) != fold or
            r["input_mode"] != mode or
            r["shared_classifier_split_sha256"] != provenance["shared_classifier_split_sha256"]):
            raise RuntimeError(f"{label}: forced-fold record does not match training")
        checkpoint = Path(r["best_model_path"])
        threshold = float(r["best_threshold"])
        validation_auc = float(r["best_validation_auc"])
        basis = f"predeclared same classifier fold {fold}; best validation epoch"
        selection_source = rec
    if not (0 <= fold <= 4) or not (0 <= threshold <= 1):
        raise RuntimeError(f"{label}: invalid fold/threshold")
    require_file(checkpoint, f"{label} ResNet34 checkpoint")
    return {
        "model_label": label,
        "input_mode": mode,
        "fold": fold,
        "model_name": "resnet34_3d",
        "threshold": threshold,
        "checkpoint": checkpoint,
        "validation_auc": validation_auc,
        "selection_source": selection_source,
        "selection_basis": basis,
    }


def load_all_model_specs():
    provenance = check_training_provenance()
    return {f"Model {m}": model_spec_from_training(m, provenance) for m in (4, 5, 6)}


# ============================================================
# 9. Load fixed pathology/test cohort references
# ============================================================

def load_fixed_test_reference(spec):
    label_file = spec["label_file"]

    require_file(
        label_file,
        f"{spec['title']} reference label file",
    )

    if spec["label_format"] == "FIXED_CSV":
        df = pd.read_csv(
            label_file,
            encoding="utf-8-sig",
        )

        required = {
            "case_id_norm",
            "binary_label",
        }

        missing = required - set(
            df.columns
        )

        if missing:
            raise KeyError(
                f"{spec['title']}: missing columns "
                f"{sorted(missing)}.\n"
                f"Available: {list(df.columns)}"
            )

        if df[
            "case_id_norm"
        ].isna().any():
            raise ValueError(
                f"{spec['title']}: missing case_id_norm."
            )

        if df[
            "binary_label"
        ].isna().any():
            raise ValueError(
                f"{spec['title']}: missing binary_label."
            )

        case_ids = [
            normalize_case_id_4digit(x)
            for x in df[
                "case_id_norm"
            ].tolist()
        ]

        labels = (
            pd.to_numeric(
                df["binary_label"],
                errors="raise",
            )
            .astype(int)
            .tolist()
        )

    elif (
        spec["label_format"]
        == "FOLDERNAME_TSTAGE_XLSX"
    ):
        df = pd.read_excel(
            label_file
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        required = {
            "FolderName",
            "Histology results _Tstage",
        }

        missing = required - set(
            df.columns
        )

        if missing:
            raise KeyError(
                f"{spec['title']}: missing columns "
                f"{sorted(missing)}.\n"
                f"Available: {list(df.columns)}"
            )

        df = df.dropna(
            subset=[
                "Histology results _Tstage"
            ]
        ).copy()

        case_ids = (
            df[
                "FolderName"
            ]
            .astype(str)
            .str.strip()
            .tolist()
        )

        labels = [
            map_t_stage_to_binary(x)
            for x in df[
                "Histology results _Tstage"
            ].tolist()
        ]

        if any(
            x is None
            for x in labels
        ):
            bad = [
                raw
                for raw, mapped
                in zip(
                    df[
                        "Histology results _Tstage"
                    ].tolist(),
                    labels,
                )
                if mapped is None
            ]

            raise ValueError(
                f"{spec['title']}: cannot map "
                f"pathology T stages: {bad}"
            )

        labels = [
            int(x)
            for x in labels
        ]

    else:
        raise ValueError(
            f"Unknown label format: "
            f"{spec['label_format']}"
        )

    if len(
        case_ids
    ) != spec[
        "expected_n"
    ]:
        raise RuntimeError(
            f"{spec['title']}: expected "
            f"{spec['expected_n']} cases, "
            f"got {len(case_ids)}."
        )

    if len(
        case_ids
    ) != len(
        set(case_ids)
    ):
        dup = (
            pd.Series(
                case_ids
            )
            .value_counts()
        )

        dup = (
            dup[
                dup > 1
            ]
            .index
            .tolist()
        )

        raise RuntimeError(
            f"{spec['title']}: duplicate case IDs: "
            f"{dup}"
        )

    unique_labels = set(
        int(x)
        for x in labels
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise RuntimeError(
            f"{spec['title']}: invalid labels: "
            f"{sorted(unique_labels)}"
        )

    return case_ids, labels


# ============================================================
# 10. Strict test-input verification
# ============================================================

def expected_case_paths(
    cases_dir,
    case_id,
):
    case_dir = (
        cases_dir
        / case_id
    )

    image_path = (
        case_dir
        / f"{case_id}_image.nii.gz"
    )

    gt_mask_path = (
        case_dir
        / f"{case_id}_mask.nii.gz"
    )

    pred_mask_path = (
        case_dir
        / f"{case_id}_pred.nii.gz"
    )

    return (
        case_dir,
        image_path,
        gt_mask_path,
        pred_mask_path,
    )


def validate_nifti_pair(
    cohort,
    case_id,
    image_path,
    mask_path,
    mask_kind,
):
    require_file(
        image_path,
        f"{cohort}/{case_id} T2WI",
    )

    require_file(
        mask_path,
        f"{cohort}/{case_id} {mask_kind} mask",
    )

    image_nii = nib.load(
        str(image_path)
    )

    mask_nii = nib.load(
        str(mask_path)
    )

    image_shape = tuple(
        int(x)
        for x in image_nii.shape
    )

    mask_shape = tuple(
        int(x)
        for x in mask_nii.shape
    )

    if len(
        image_shape
    ) != 3 or len(
        mask_shape
    ) != 3:
        raise RuntimeError(
            f"{cohort}/{case_id}: expected 3D NIfTI. "
            f"image={image_shape}, mask={mask_shape}"
        )

    if (
        image_shape
        != mask_shape
    ):
        raise RuntimeError(
            f"{cohort}/{case_id}: image/mask shape mismatch. "
            f"image={image_shape}, mask={mask_shape}"
        )

    if not np.allclose(
        np.asarray(
            image_nii.affine,
            dtype=np.float64,
        ),
        np.asarray(
            mask_nii.affine,
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"{cohort}/{case_id}: "
            f"image/mask affine mismatch."
        )

    depth_axis = int(
        np.argmin(
            image_shape
        )
    )

    dhw = tuple(
        [
            image_shape[
                depth_axis
            ]
        ]
        + [
            image_shape[i]
            for i in range(3)
            if i != depth_axis
        ]
    )

    if dhw != tuple(
        Config.input_size_3d
    ):
        raise RuntimeError(
            f"{cohort}/{case_id}: "
            f"derived DHW={dhw}, expected="
            f"{tuple(Config.input_size_3d)}"
        )

    raw_unique = sorted(
        float(x)
        for x in np.unique(
            mask_nii.get_fdata()
        )
    )

    # Stronger binary/discrete check.
    arr = mask_nii.get_fdata()

    if not np.all(
        np.isclose(
            arr,
            0.0,
            atol=1e-6,
        )
        | np.isclose(
            arr,
            1.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError(
            f"{cohort}/{case_id}: "
            f"{mask_kind} mask contains non-binary values. "
            f"Unique values include: {raw_unique[:20]}"
        )

    return {
        "image_shape": list(
            image_shape
        ),
        "mask_shape": list(
            mask_shape
        ),
        "derived_dhw": list(
            dhw
        ),
        "affine_match": True,
        "mask_unique_values": raw_unique,
    }


def build_model_input_manifest(
    cohort_name,
    spec,
    case_ids,
    labels,
    model_label,
    input_mode,
):
    """
    Build an exact CSV compatible with load_samples_from_test_csv().
    """
    rows = []

    for case_id, label in zip(
        case_ids,
        labels,
    ):
        (
            case_dir,
            image_path,
            gt_mask_path,
            pred_mask_path,
        ) = expected_case_paths(
            spec["cases_dir"],
            case_id,
        )

        require_dir(
            case_dir,
            f"{cohort_name}/{case_id} case directory",
        )

        require_file(
            image_path,
            f"{cohort_name}/{case_id} T2WI",
        )

        row = {
            "case_id_folder": case_id,
            "label": int(
                label
            ),
            "image_path": str(
                image_path
            ),
        }

        if (
            input_mode
            == "image_plus_gt_mask"
        ):
            if not str(
                gt_mask_path
            ).endswith(
                "_mask.nii.gz"
            ):
                raise RuntimeError(
                    f"{cohort_name}/{case_id}: "
                    f"invalid GT mask path."
                )

            audit = validate_nifti_pair(
                cohort=cohort_name,
                case_id=case_id,
                image_path=image_path,
                mask_path=gt_mask_path,
                mask_kind="manual/GT",
            )

            row[
                "mask_path"
            ] = str(
                gt_mask_path
            )

            row[
                "_audit_mask_source"
            ] = "MANUAL_GT_MASK"

            row[
                "_audit_shape"
            ] = json.dumps(
                audit,
                ensure_ascii=False,
            )

        elif (
            input_mode
            == "image_plus_pred_mask"
        ):
            if not str(
                pred_mask_path
            ).endswith(
                "_pred.nii.gz"
            ):
                raise RuntimeError(
                    f"{cohort_name}/{case_id}: "
                    f"invalid predicted-mask path."
                )

            audit = validate_nifti_pair(
                cohort=cohort_name,
                case_id=case_id,
                image_path=image_path,
                mask_path=pred_mask_path,
                mask_kind="original nnU-Net predicted",
            )

            row[
                "mask_path"
            ] = str(
                pred_mask_path
            )

            row[
                "_audit_mask_source"
            ] = (
                "ORIGINAL_INDEPENDENT_TEST_"
                "NNUNET_PREDICTED_MASK"
            )

            row[
                "_audit_shape"
            ] = json.dumps(
                audit,
                ensure_ascii=False,
            )

        elif (
            input_mode
            == "image_only"
        ):
            row[
                "_audit_mask_source"
            ] = "NONE"

            row[
                "_audit_shape"
            ] = ""

        else:
            raise ValueError(
                f"Unknown input_mode: "
                f"{input_mode}"
            )

        row[
            "_audit_model"
        ] = model_label

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ) != spec[
        "expected_n"
    ]:
        raise RuntimeError(
            f"{cohort_name}/{model_label}: "
            f"manifest has {len(df)} cases; "
            f"expected {spec['expected_n']}."
        )

    return df


# ============================================================
# 11. Model loading and inference
# ============================================================

def load_checkpointed_model(
    model_spec,
    device,
):
    Config.input_mode = (
        model_spec[
            "input_mode"
        ]
    )

    Config.model_name = (
        model_spec[
            "model_name"
        ]
    )

    Config.label_mode = (
        "binary"
    )

    in_channels = (
        Config
        .get_num_input_channels()
    )

    expected_channels = (
        1
        if model_spec[
            "input_mode"
        ] == "image_only"
        else 2
    )

    if (
        in_channels
        != expected_channels
    ):
        raise RuntimeError(
            f"{model_spec['model_label']}: "
            f"input channel mismatch. "
            f"Expected={expected_channels}, "
            f"Config={in_channels}"
        )

    # ResNet34 OOF was trained with dropout=0.2 (matching DenseNet121),
    # overriding the original ResNet34 config default 0.5 at inference too.
    create_kwargs = {"dropout": 0.2} if model_spec["model_name"] == "resnet34_3d" else {}
    model = create_model(
        model_name=model_spec["model_name"],
        num_classes=2,
        in_channels=in_channels,
        **create_kwargs,
    )

    checkpoint_path = Path(
        model_spec[
            "checkpoint"
        ]
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            f"model_state_dict missing from "
            f"{checkpoint_path}"
        )

    # Check training checkpoint identity BEFORE inference (all 3 models).
    checkpoint_arch = checkpoint.get("model_name")
    if checkpoint_arch is not None and checkpoint_arch != model_spec["model_name"]:
        raise RuntimeError(
            f"{model_spec['model_label']}: checkpoint architecture mismatch: "
            f"checkpoint={checkpoint_arch}, expected={model_spec['model_name']}"
        )
    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict):
        checkpoint_mode = checkpoint_config.get("input_mode")
        if checkpoint_mode is not None and checkpoint_mode != model_spec["input_mode"]:
            raise RuntimeError(
                f"{model_spec['model_label']}: checkpoint input-mode mismatch: "
                f"checkpoint={checkpoint_mode}, expected={model_spec['input_mode']}"
            )
    ckpt_threshold = checkpoint.get("best_threshold")
    if ckpt_threshold is not None and not np.isclose(
        float(ckpt_threshold), float(model_spec["threshold"]), atol=1e-6, rtol=0.0
    ):
        raise RuntimeError(
            f"{model_spec['model_label']}: threshold mismatch between checkpoint "
            f"({ckpt_threshold}) and selection record ({model_spec['threshold']})."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    return model


def predict_from_manifest(
    model_spec,
    manifest_csv,
    expected_n,
    output_csv,
    device,
):
    Config.input_mode = (
        model_spec[
            "input_mode"
        ]
    )

    Config.model_name = model_spec["model_name"]

    Config.label_mode = (
        "binary"
    )

    # --------------------------------------------------------
    # CRITICAL ID-PRESERVATION FIX
    # --------------------------------------------------------
    
    # pandas inside the historical loader may parse those values as
    # integers and return 11, 41, 69 through the DataLoader.
    #
    # Therefore the authoritative case ID used for all saved predictions
    # is read DIRECTLY from the exact loader manifest with dtype=str.
   
    # --------------------------------------------------------
    manifest_df_exact = pd.read_csv(
        manifest_csv,
        dtype={
            "case_id_folder": str,
        },
        encoding="utf-8-sig",
    )

    if (
        "case_id_folder"
        not in manifest_df_exact.columns
    ):
        raise KeyError(
            f"case_id_folder missing from manifest: "
            f"{manifest_csv}"
        )

    exact_case_ids = (
        manifest_df_exact[
            "case_id_folder"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(
        exact_case_ids
    ) != expected_n:
        raise RuntimeError(
            f"{model_spec['model_label']}: "
            f"exact manifest contains "
            f"{len(exact_case_ids)} cases; "
            f"expected {expected_n}."
        )

    if len(
        exact_case_ids
    ) != len(
        set(
            exact_case_ids
        )
    ):
        raise RuntimeError(
            f"{model_spec['model_label']}: "
            f"duplicate case IDs in exact manifest."
        )

    samples = (
        load_samples_from_test_csv(
            str(
                manifest_csv
            ),
            input_mode=(
                model_spec[
                    "input_mode"
                ]
            ),
        )
    )

    if len(
        samples
    ) != expected_n:
        raise RuntimeError(
            f"{model_spec['model_label']}: "
            f"data loader returned "
            f"{len(samples)} cases; "
            f"expected {expected_n}."
        )

    dataset = (
        RectalMRIClassification3DDataset(
            samples=samples,
            is_train=False,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    model = load_checkpointed_model(
        model_spec=model_spec,
        device=device,
    )

    threshold = float(
        model_spec[
            "threshold"
        ]
    )

    rows = []

    print(
        f"\nPrediction: "
        f"{model_spec['model_label']} | "
        f"{model_spec['input_mode']} | "
        f"N={len(loader)}"
    )

    print(
        f"  fold      : "
        f"{model_spec['fold']}"
    )

    print(
        f"  threshold : "
        f"{threshold:.6f}"
    )

    print(
        f"  checkpoint: "
        f"{model_spec['checkpoint']}"
    )

    with torch.no_grad():

        for i, (
            images,
            labels,
            case_ids,
        ) in enumerate(
            loader
        ):

            # Use the exact manifest ID, not the potentially
            # numeric-converted ID returned by the historical DataLoader.
            case_id = exact_case_ids[
                i
            ]

            print(
                f"  {i+1}/{len(loader)}",
                end="\r",
            )

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                images
            )

            probs = F.softmax(
                outputs,
                dim=1,
            ).cpu().numpy()[0]

            prob1 = float(
                probs[1]
            )

            pred = int(
                prob1
                >= threshold
            )

            rows.append(
                {
                    "case_id": case_id,
                    "y_true": int(
                        labels
                        .cpu()
                        .numpy()[0]
                    ),
                    "y_pred": pred,
                    "prob_class0": float(
                        probs[0]
                    ),
                    "prob_class1": prob1,
                    "threshold_used": threshold,
                    "model": model_spec[
                        "model_label"
                    ],
                    "fold": int(
                        model_spec[
                            "fold"
                        ]
                    ),
                }
            )

    print()

    if len(
        rows
    ) != expected_n:
        raise RuntimeError(
            f"{model_spec['model_label']}: "
            f"inference incomplete "
            f"{len(rows)}/{expected_n}"
        )

    df = pd.DataFrame(
        rows
    )

    # Saved IDs must exactly match the physical manifest IDs,
    
    if (
        df[
            "case_id"
        ].astype(str).tolist()
        != exact_case_ids
    ):
        raise RuntimeError(
            f"{model_spec['model_label']}: "
            f"saved prediction case-ID order does not exactly match "
            f"the input manifest."
        )

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


# ============================================================
# 12. Radiologist loading
# ============================================================

def load_internal_test1_radiologists(
    spec,
    case_ids,
    labels,
):
    senior_path = spec[
        "senior_file"
    ]

    junior_path = spec[
        "junior_file"
    ]

    require_file(
        senior_path,
        "senior radiologist CSV",
    )

    require_file(
        junior_path,
        "junior radiologist CSV",
    )

    senior = pd.read_csv(
        senior_path,
        encoding="utf-8-sig",
    )

    junior = pd.read_csv(
        junior_path,
        encoding="utf-8-sig",
    )

    senior.columns = [
        str(c).strip()
        for c in senior.columns
    ]

    junior.columns = [
        str(c).strip()
        for c in junior.columns
    ]

    senior_required = {
        "case_id_folder",
        "Senior Radiologist_Tstage",
        "Pathology_t_stage_raw",
    }

    junior_required = {
        "case_id_folder",
        "junior radiologist_Tstage",
        "Pathology_binary_label",
    }

    miss = (
        senior_required
        - set(
            senior.columns
        )
    )

    if miss:
        raise KeyError(
            f"Senior radiologist CSV missing: "
            f"{sorted(miss)}"
        )

    miss = (
        junior_required
        - set(
            junior.columns
        )
    )

    if miss:
        raise KeyError(
            f"Junior radiologist CSV missing: "
            f"{sorted(miss)}"
        )

    senior[
        "merge_id"
    ] = senior[
        "case_id_folder"
    ].apply(
        radiologist_merge_id
    )

    junior[
        "merge_id"
    ] = junior[
        "case_id_folder"
    ].apply(
        radiologist_merge_id
    )

    senior[
        "senior_stage"
    ] = senior[
        "Senior Radiologist_Tstage"
    ].apply(
        extract_stage
    )

    senior[
        "senior_score"
    ] = senior[
        "senior_stage"
    ].apply(
        stage_to_score
    )

    senior[
        "senior_pred"
    ] = senior[
        "senior_stage"
    ].apply(
        stage_to_binary
    )

    senior[
        "path_stage"
    ] = senior[
        "Pathology_t_stage_raw"
    ].apply(
        extract_stage
    )

    senior[
        "path_bin"
    ] = senior[
        "path_stage"
    ].apply(
        stage_to_binary
    )

    junior[
        "junior_stage"
    ] = junior[
        "junior radiologist_Tstage"
    ].apply(
        extract_stage
    )

    junior[
        "junior_score"
    ] = junior[
        "junior_stage"
    ].apply(
        stage_to_score
    )

    junior[
        "junior_pred"
    ] = junior[
        "junior_stage"
    ].apply(
        stage_to_binary
    )

    junior[
        "path_bin_junior_file"
    ] = pd.to_numeric(
        junior[
            "Pathology_binary_label"
        ],
        errors="raise",
    ).astype(
        int
    )

    senior = senior.dropna(
        subset=[
            "merge_id",
            "senior_score",
            "senior_pred",
            "path_bin",
        ]
    ).copy()

    junior = junior.dropna(
        subset=[
            "merge_id",
            "junior_score",
            "junior_pred",
            "path_bin_junior_file",
        ]
    ).copy()

    if senior[
        "merge_id"
    ].duplicated().any():
        raise RuntimeError(
            "senior radiologist CSV "
            "contains duplicate patient IDs."
        )

    if junior[
        "merge_id"
    ].duplicated().any():
        raise RuntimeError(
            "junior radiologist CSV "
            "contains duplicate patient IDs."
        )

    senior = senior.set_index(
        "merge_id"
    )

    junior = junior.set_index(
        "merge_id"
    )

    requested_ids = [
        str(
            int(
                cid
            )
        )
        for cid in case_ids
    ]

    missing_s = (
        set(
            requested_ids
        )
        - set(
            senior.index
        )
    )

    missing_j = (
        set(
            requested_ids
        )
        - set(
            junior.index
        )
    )

    if missing_s:
        raise RuntimeError(
            f"Senior radiologist missing IDs: "
            f"{sorted(missing_s)}"
        )

    if missing_j:
        raise RuntimeError(
            f"Junior radiologist missing IDs: "
            f"{sorted(missing_j)}"
        )

    senior = senior.loc[
        requested_ids
    ]

    junior = junior.loc[
        requested_ids
    ]

    fixed_labels = np.asarray(
        labels,
        dtype=int,
    )

    senior_path_labels = senior[
        "path_bin"
    ].astype(
        int
    ).to_numpy()

    junior_path_labels = junior[
        "path_bin_junior_file"
    ].astype(
        int
    ).to_numpy()

    if not np.array_equal(
        fixed_labels,
        senior_path_labels,
    ):
        raise RuntimeError(
            "Pathology labels in the fixed-test CSV "
            "do not match the senior-radiologist CSV."
        )

    if not np.array_equal(
        fixed_labels,
        junior_path_labels,
    ):
        raise RuntimeError(
            "Pathology labels in the fixed-test CSV "
            "do not match the junior-radiologist CSV."
        )

    return pd.DataFrame(
        {
            "case_id": case_ids,
            "y_true": fixed_labels,
            "senior_radio_stage": senior[
                "senior_stage"
            ].tolist(),
            "senior_radio_score": senior[
                "senior_score"
            ].astype(
                float
            ).to_numpy(),
            "senior_radio_pred": senior[
                "senior_pred"
            ].astype(
                int
            ).to_numpy(),
            "junior_radio_stage": junior[
                "junior_stage"
            ].tolist(),
            "junior_radio_score": junior[
                "junior_score"
            ].astype(
                float
            ).to_numpy(),
            "junior_radio_pred": junior[
                "junior_pred"
            ].astype(
                int
            ).to_numpy(),
        }
    )


def load_xlsx_radiologists(
    spec,
    case_ids,
    labels,
):
    df = pd.read_excel(
        spec["label_file"]
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    required = {
        "FolderName",
        "Histology results _Tstage",
        "Senior Radiologist_Tstage",
        "Junior Radiologist_Tstage",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            f"{spec['title']}: radiologist XLSX "
            f"missing columns {sorted(missing)}"
        )

    df = df.dropna(
        subset=[
            "Histology results _Tstage"
        ]
    ).copy()

    df[
        "case_id"
    ] = (
        df[
            "FolderName"
        ]
        .astype(str)
        .str.strip()
    )

    if df[
        "case_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{spec['title']}: duplicate FolderName "
            f"in radiologist XLSX."
        )

    df = df.set_index(
        "case_id"
    )

    missing_cases = (
        set(
            case_ids
        )
        - set(
            df.index
        )
    )

    if missing_cases:
        raise RuntimeError(
            f"{spec['title']}: radiologist XLSX "
            f"missing cases: {sorted(missing_cases)}"
        )

    df = df.loc[
        case_ids
    ]

    path_bin = np.asarray(
        [
            map_t_stage_to_binary(x)
            for x in df[
                "Histology results _Tstage"
            ].tolist()
        ],
        dtype=int,
    )

    fixed_labels = np.asarray(
        labels,
        dtype=int,
    )

    if not np.array_equal(
        path_bin,
        fixed_labels,
    ):
        raise RuntimeError(
            f"{spec['title']}: pathology labels "
            f"changed during radiologist-table alignment."
        )

    senior_score = np.asarray(
        [
            map_t_stage_to_score(x)
            for x in df[
                "Senior Radiologist_Tstage"
            ].tolist()
        ],
        dtype=float,
    )

    senior_pred = np.asarray(
        [
            map_t_stage_to_binary(x)
            for x in df[
                "Senior Radiologist_Tstage"
            ].tolist()
        ],
        dtype=float,
    )

    junior_score = np.asarray(
        [
            map_t_stage_to_score(x)
            for x in df[
                "Junior Radiologist_Tstage"
            ].tolist()
        ],
        dtype=float,
    )

    junior_pred = np.asarray(
        [
            map_t_stage_to_binary(x)
            for x in df[
                "Junior Radiologist_Tstage"
            ].tolist()
        ],
        dtype=float,
    )

    if (
        np.isnan(
            senior_score
        ).any()
        or np.isnan(
            senior_pred
        ).any()
        or np.isnan(
            junior_score
        ).any()
        or np.isnan(
            junior_pred
        ).any()
    ):
        raise RuntimeError(
            f"{spec['title']}: one or more "
            f"radiologist T-stage values could not be mapped."
        )

    return pd.DataFrame(
        {
            "case_id": case_ids,
            "y_true": fixed_labels,
            "senior_radio_stage": [
                str(x)
                for x in df[
                    "Senior Radiologist_Tstage"
                ].tolist()
            ],
            "senior_radio_score": senior_score,
            "senior_radio_pred": senior_pred.astype(
                int
            ),
            "junior_radio_stage": [
                str(x)
                for x in df[
                    "Junior Radiologist_Tstage"
                ].tolist()
            ],
            "junior_radio_score": junior_score,
            "junior_radio_pred": junior_pred.astype(
                int
            ),
        }
    )


def load_radiologist_data(
    cohort_name,
    spec,
    case_ids,
    labels,
):
    if cohort_name == "IT1":
        return load_internal_test1_radiologists(
            spec=spec,
            case_ids=case_ids,
            labels=labels,
        )

    return load_xlsx_radiologists(
        spec=spec,
        case_ids=case_ids,
        labels=labels,
    )


# ============================================================
# 13. Metrics
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
    y_score,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    ).ravel()

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    ).ravel()

    y_score = np.asarray(
        y_score,
        dtype=float,
    ).ravel()

    acc = accuracy_score(
        y_true,
        y_pred,
    )

    prec = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    rec = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    spec = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    youden = (
        rec
        + spec
        - 1
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_score,
        )
    except ValueError:
        auc = np.nan

    return {
        "AUC": float(
            auc
        ),
        "Accuracy": float(
            acc
        ),
        "Sensitivity": float(
            rec
        ),
        "Specificity": float(
            spec
        ),
        "PPV": float(
            prec
        ),
        "NPV": float(
            npv
        ),
        "F1": float(
            f1
        ),
        "Youden": float(
            youden
        ),
        "TN": int(
            tn
        ),
        "FP": int(
            fp
        ),
        "FN": int(
            fn
        ),
        "TP": int(
            tp
        ),
    }


def compute_metrics_bootstrap_ci(
    y_true,
    y_pred,
    y_score,
    n_boot=1000,
    seed=42,
):
    point = compute_metrics(
        y_true,
        y_pred,
        y_score,
    )

    rng = np.random.default_rng(
        seed
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    n = len(
        y_true
    )

    metric_names = [
        "AUC",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1",
        "Youden",
    ]

    boot_metrics = {
        key: []
        for key in metric_names
    }

    for _ in range(
        n_boot
    ):
        idx = rng.choice(
            n,
            size=n,
            replace=True,
        )

        yt = y_true[
            idx
        ]

        yp = y_pred[
            idx
        ]

        ys = y_score[
            idx
        ]

        if len(
            np.unique(
                yt
            )
        ) < 2:
            continue

        m = compute_metrics(
            yt,
            yp,
            ys,
        )

        for key in metric_names:
            value = m[
                key
            ]

            if np.isfinite(
                value
            ):
                boot_metrics[
                    key
                ].append(
                    value
                )

    ci = {}

    for key in metric_names:
        arr = np.asarray(
            boot_metrics[
                key
            ],
            dtype=float,
        )

        if len(
            arr
        ) == 0:
            ci[
                key
            ] = (
                np.nan,
                np.nan,
            )
        else:
            low, high = np.percentile(
                arr,
                [2.5, 97.5],
            )

            ci[
                key
            ] = (
                float(low),
                float(high),
            )

    return point, ci


# ============================================================
# 14. DeLong / radiologist CIs
# ============================================================

def delong_roc_variance(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    n1 = np.sum(
        y_true == 1
    )

    n0 = np.sum(
        y_true == 0
    )

    if (
        n1 == 0
        or n0 == 0
    ):
        return (
            np.nan,
            np.nan,
        )

    sp = y_pred[
        y_true == 1
    ]

    sn = y_pred[
        y_true == 0
    ]

    V10 = np.zeros(
        n1
    )

    for i in range(
        n1
    ):
        V10[
            i
        ] = (
            (1 / n0)
            * np.sum(
                sp[i] > sn
            )
            + (
                1
                / (2 * n0)
            )
            * np.sum(
                sp[i] == sn
            )
        )

    V01 = np.zeros(
        n0
    )

    for j in range(
        n0
    ):
        V01[
            j
        ] = (
            (1 / n1)
            * np.sum(
                sn[j] < sp
            )
            + (
                1
                / (2 * n1)
            )
            * np.sum(
                sn[j] == sp
            )
        )

    auc = np.mean(
        V10
    )

    var = (
        np.var(
            V10,
            ddof=1,
        )
        / n1
        + np.var(
            V01,
            ddof=1,
        )
        / n0
    )

    return (
        float(
            auc
        ),
        float(
            var
        ),
    )


def radiologist_metrics_with_ci(
    y_true,
    y_pred,
    y_score,
):
    point = compute_metrics(
        y_true,
        y_pred,
        y_score,
    )

    ci = {}

    auc, var = delong_roc_variance(
        y_true,
        y_score,
    )

    if (
        np.isfinite(
            auc
        )
        and np.isfinite(
            var
        )
        and var > 0
    ):
        se = np.sqrt(
            var
        )

        low = max(
            0.0,
            auc
            - 1.96
            * se,
        )

        high = min(
            1.0,
            auc
            + 1.96
            * se,
        )

    else:
        low = np.nan
        high = np.nan

    ci[
        "AUC"
    ] = (
        low,
        high,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    N = len(
        y_true
    )

    ci[
        "Accuracy"
    ] = proportion_confint(
        count=(
            tp
            + tn
        ),
        nobs=N,
        method="wilson",
    )

    ci[
        "Sensitivity"
    ] = proportion_confint(
        count=tp,
        nobs=(
            tp
            + fn
        ),
        method="wilson",
    )

    ci[
        "Specificity"
    ] = proportion_confint(
        count=tn,
        nobs=(
            tn
            + fp
        ),
        method="wilson",
    )

    ci[
        "PPV"
    ] = proportion_confint(
        count=tp,
        nobs=(
            tp
            + fp
        ),
        method="wilson",
    )

    ci[
        "NPV"
    ] = proportion_confint(
        count=tn,
        nobs=(
            tn
            + fn
        ),
        method="wilson",
    )

    # F1: same bootstrap strategy as the original scripts.
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    rng = np.random.default_rng(
        42
    )

    n = len(
        y_true
    )

    f1_boot = []

    for _ in range(
        1000
    ):
        idx = rng.choice(
            n,
            size=n,
            replace=True,
        )

        yt = y_true[
            idx
        ]

        yp = y_pred[
            idx
        ]

        tp_b = np.sum(
            (yt == 1)
            & (yp == 1)
        )

        fp_b = np.sum(
            (yt == 0)
            & (yp == 1)
        )

        fn_b = np.sum(
            (yt == 1)
            & (yp == 0)
        )

        denom = (
            2 * tp_b
            + fp_b
            + fn_b
        )

        f1_boot.append(
            1.0
            if denom == 0
            else (
                2
                * tp_b
                / denom
            )
        )

    ci[
        "F1"
    ] = tuple(
        np.percentile(
            f1_boot,
            [2.5, 97.5],
        )
    )

    # Youden bootstrap for completeness.
    _, model_ci = compute_metrics_bootstrap_ci(
        y_true,
        y_pred,
        y_score,
        n_boot=1000,
        seed=42,
    )

    ci[
        "Youden"
    ] = model_ci[
        "Youden"
    ]

    return (
        point,
        ci,
    )


def delong_test(
    y_true,
    pred1,
    pred2,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    pred1 = np.asarray(
        pred1,
        dtype=float,
    )

    pred2 = np.asarray(
        pred2,
        dtype=float,
    )

    valid = ~(
        np.isnan(
            pred1
        )
        | np.isnan(
            pred2
        )
    )

    y_true = y_true[
        valid
    ]

    pred1 = pred1[
        valid
    ]

    pred2 = pred2[
        valid
    ]

    auc1, var1 = delong_roc_variance(
        y_true,
        pred1,
    )

    auc2, var2 = delong_roc_variance(
        y_true,
        pred2,
    )

    if (
        not np.isfinite(
            auc1
        )
        or not np.isfinite(
            auc2
        )
    ):
        return (
            auc1,
            auc2,
            np.nan,
            np.nan,
            np.nan,
        )

    n1 = np.sum(
        y_true == 1
    )

    n0 = np.sum(
        y_true == 0
    )

    sp1 = pred1[
        y_true == 1
    ]

    sn1 = pred1[
        y_true == 0
    ]

    sp2 = pred2[
        y_true == 1
    ]

    sn2 = pred2[
        y_true == 0
    ]

    V10_1 = np.zeros(
        n1
    )

    V10_2 = np.zeros(
        n1
    )

    for i in range(
        n1
    ):
        V10_1[
            i
        ] = (
            (1 / n0)
            * np.sum(
                sp1[i] > sn1
            )
            + (
                1
                / (2 * n0)
            )
            * np.sum(
                sp1[i] == sn1
            )
        )

        V10_2[
            i
        ] = (
            (1 / n0)
            * np.sum(
                sp2[i] > sn2
            )
            + (
                1
                / (2 * n0)
            )
            * np.sum(
                sp2[i] == sn2
            )
        )

    V01_1 = np.zeros(
        n0
    )

    V01_2 = np.zeros(
        n0
    )

    for j in range(
        n0
    ):
        V01_1[
            j
        ] = (
            (1 / n1)
            * np.sum(
                sn1[j] < sp1
            )
            + (
                1
                / (2 * n1)
            )
            * np.sum(
                sn1[j] == sp1
            )
        )

        V01_2[
            j
        ] = (
            (1 / n1)
            * np.sum(
                sn2[j] < sp2
            )
            + (
                1
                / (2 * n1)
            )
            * np.sum(
                sn2[j] == sp2
            )
        )

    cov = (
        np.cov(
            V10_1,
            V10_2,
            ddof=1,
        )[0, 1]
        / n1
        + np.cov(
            V01_1,
            V01_2,
            ddof=1,
        )[0, 1]
        / n0
    )

    delta = (
        auc1
        - auc2
    )

    var_diff = (
        var1
        + var2
        - 2
        * cov
    )

    if (
        var_diff
        <= 0
    ):
        z = 0.0
        p = 1.0
    else:
        z = (
            delta
            / np.sqrt(
                var_diff
            )
        )

        p = (
            2
            * (
                1
                - norm.cdf(
                    abs(
                        z
                    )
                )
            )
        )

    return (
        auc1,
        auc2,
        delta,
        z,
        p,
    )


def bootstrap_auc_diff(
    y_true,
    s1,
    s2,
    n_boot=1000,
    seed=42,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    s1 = np.asarray(
        s1,
        dtype=float,
    )

    s2 = np.asarray(
        s2,
        dtype=float,
    )

    valid = ~(
        np.isnan(
            s1
        )
        | np.isnan(
            s2
        )
    )

    y_true = y_true[
        valid
    ]

    s1 = s1[
        valid
    ]

    s2 = s2[
        valid
    ]

    rng = np.random.default_rng(
        seed
    )

    n = len(
        y_true
    )

    diffs = []

    for _ in range(
        n_boot
    ):
        idx = rng.choice(
            n,
            size=n,
            replace=True,
        )

        yb = y_true[
            idx
        ]

        if len(
            np.unique(
                yb
            )
        ) < 2:
            continue

        diffs.append(
            roc_auc_score(
                yb,
                s1[
                    idx
                ],
            )
            - roc_auc_score(
                yb,
                s2[
                    idx
                ],
            )
        )

    diffs = np.asarray(
        diffs,
        dtype=float,
    )

    if len(
        diffs
    ) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mean_diff = np.mean(
        diffs
    )

    low, high = np.percentile(
        diffs,
        [2.5, 97.5],
    )

    p = (
        2
        * min(
            np.mean(
                diffs <= 0
            ),
            np.mean(
                diffs >= 0
            ),
        )
    )

    return (
        float(
            mean_diff
        ),
        float(
            low
        ),
        float(
            high
        ),
        float(
            p
        ),
    )


# ============================================================
# 15. ROC confidence band
# ============================================================

def compute_roc_ci_band(
    y_true,
    y_score,
    n_boot=1000,
    seed=42,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    fpr_grid = np.linspace(
        0,
        1,
        100,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        y_true
    )

    tprs = []

    for _ in range(
        n_boot
    ):
        idx = rng.choice(
            n,
            size=n,
            replace=True,
        )

        yb = y_true[
            idx
        ]

        sb = y_score[
            idx
        ]

        if len(
            np.unique(
                yb
            )
        ) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            yb,
            sb,
        )

        tpr_interp = np.interp(
            fpr_grid,
            fpr,
            tpr,
        )

        tpr_interp[
            0
        ] = 0.0

        tprs.append(
            tpr_interp
        )

    if len(
        tprs
    ) == 0:
        return None

    tprs = np.asarray(
        tprs
    )

    tpr_mean = np.mean(
        tprs,
        axis=0,
    )

    tpr_low = np.percentile(
        tprs,
        2.5,
        axis=0,
    )

    tpr_high = np.percentile(
        tprs,
        97.5,
        axis=0,
    )

    return (
        fpr_grid,
        tpr_mean,
        tpr_low,
        tpr_high,
    )


# ============================================================
# 16. Plotting - same style as original scripts
# ============================================================

COLOR_MAP = {
    "Senior Radiologist": "#F5A623",
    "Junior Radiologist": "#4E91C4",
    MODEL_DISPLAY["Model 4"]: "#8E44AD",
    MODEL_DISPLAY["Model 5"]: "#009E73",
    MODEL_DISPLAY["Model 6"]: "#F4A3AA",
}


def plot_combined_roc(
    y_true,
    score_dict,
    save_path,
    title,
    with_ci=True,
):
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    order = [
        MODEL_DISPLAY[
            "Model 4"
        ],
        MODEL_DISPLAY[
            "Model 5"
        ],
        MODEL_DISPLAY[
            "Model 6"
        ],
        "Senior Radiologist",
        "Junior Radiologist",
    ]

    for label in order:
        score = np.asarray(
            score_dict[
                label
            ],
            dtype=float,
        )

        auc_value = roc_auc_score(
            y_true,
            score,
        )

        fpr, tpr, _ = roc_curve(
            y_true,
            score,
        )

        color = COLOR_MAP[
            label
        ]

        if with_ci:
            ci = compute_roc_ci_band(
                y_true,
                score,
            )

            if ci is not None:
                (
                    fpr_grid,
                    _,
                    tpr_low,
                    tpr_high,
                ) = ci

                ax.fill_between(
                    fpr_grid,
                    tpr_low,
                    tpr_high,
                    color=color,
                    alpha=0.15,
                )

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.0,
            label=(
                f"{label} "
                f"(AUC = {auc_value:.2f})"
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        lw=1.5,
    )

    ax.set_xlim(
        [0, 1]
    )

    ax.set_ylim(
        [0, 1.05]
    )

    ax.set_xlabel(
        "1 - Specificity",
        fontsize=20,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Sensitivity",
        fontsize=20,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=22,
        fontweight="bold",
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=13,
    )

    ax.tick_params(
        axis="both",
        labelsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(
            ".png"
        ),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_radar_chart(
    summary_df,
    save_path,
    title,
):
    metrics = [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1",
    ]

    desired_order = [
        "ResNet34 Model 4 (Manual Mask)",
        "ResNet34 Model 5 (Pred Mask)",
        "ResNet34 Model 6 (T2WI Only)",
        "Senior Radiologist",
        "Junior Radiologist",
    ]

    display_names = {
        "ResNet34 Model 4 (Manual Mask)":
            MODEL_DISPLAY[
                "Model 4"
            ],
        "ResNet34 Model 5 (Pred Mask)":
            MODEL_DISPLAY[
                "Model 5"
            ],
        "ResNet34 Model 6 (T2WI Only)":
            MODEL_DISPLAY[
                "Model 6"
            ],
        "Senior Radiologist":
            "Senior Radiologist",
        "Junior Radiologist":
            "Junior Radiologist",
    }

    linestyle_map = {
        "ResNet34 Model 4 (Manual Mask)": "-",
        "ResNet34 Model 5 (Pred Mask)": "--",
        "ResNet34 Model 6 (T2WI Only)": "-.",
        "Senior Radiologist": ":",
        "Junior Radiologist": (
            0,
            (
                3,
                1,
                1,
                1,
            ),
        ),
    }

    marker_map = {
        "ResNet34 Model 4 (Manual Mask)": "s",
        "ResNet34 Model 5 (Pred Mask)": "o",
        "ResNet34 Model 6 (T2WI Only)": "^",
        "Senior Radiologist": "D",
        "Junior Radiologist": "v",
    }

    summary = (
        summary_df
        .copy()
        .set_index(
            "Method"
        )
        .reindex(
            desired_order
        )
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        len(
            metrics
        ),
        endpoint=False,
    ).tolist()

    angles += angles[
        :1
    ]

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(
            polar=True
        ),
    )

    for method in desired_order:
        row = summary.loc[
            method
        ]

        values = (
            row[
                metrics
            ]
            .astype(
                float
            )
            .values
            .tolist()
        )

        values += values[
            :1
        ]

        display = display_names[
            method
        ]

        color = COLOR_MAP[
            display
        ]

        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2,
            linestyle=(
                linestyle_map[
                    method
                ]
            ),
            marker=(
                marker_map[
                    method
                ]
            ),
            markersize=6,
            label=display,
        )

        ax.fill(
            angles,
            values,
            color=color,
            alpha=0.1,
        )

    ax.set_yticks(
        np.arange(
            0,
            1.1,
            0.1,
        )
    )

    ax.set_xticks(
        angles[
            :-1
        ]
    )

    ax.set_xticklabels(
        metrics,
        fontsize=12,
        fontweight="bold",
    )

    ax.tick_params(
        axis="x",
        pad=20,
    )

    ax.set_rlabel_position(
        180
    )

    ax.tick_params(
        axis="y",
        labelsize=8,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.2,
        ),
        ncol=2,
        fontsize=12,
    )

    plt.tight_layout()

    fig.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        save_path.with_suffix(
            ".png"
        ),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_confusion_matrix_png(
    y_true,
    y_pred,
    save_path,
    title,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    labels = [
        "T1-2",
        "T3-4",
    ]

    deep_blue = "#1f77b4"
    mid_blue = "#5dade2"
    light_blue = "#d6eaf8"

    cmap = ListedColormap(
        [
            light_blue,
            mid_blue,
            deep_blue,
        ]
    )

    data_map = np.array(
        [
            [1, 0],
            [0, 2],
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(
        data_map,
        cmap=cmap,
        vmin=0,
        vmax=2,
        aspect="auto",
        origin="upper",
    )

    for i in range(
        2
    ):
        for j in range(
            2
        ):
            ax.text(
                j,
                i,
                str(
                    cm[
                        i,
                        j
                    ]
                ),
                ha="center",
                va="center",
                fontsize=24,
                fontweight="normal",
                color="black",
            )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        labels,
        fontsize=18,
        fontweight="bold",
    )

    ax.set_yticklabels(
        labels,
        fontsize=18,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=22,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Model Classification",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_ylabel(
        "Pathology Results",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )

    ax.tick_params(
        left=False,
        bottom=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )

    plt.tight_layout()

    fig.savefig(
        save_path,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 17. Cohort evaluation
# ============================================================

def align_predictions_to_case_order(
    prediction_df,
    case_ids,
    model_label,
):
    if prediction_df[
        "case_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{model_label}: duplicate case IDs "
            f"in prediction CSV."
        )

    p = (
        prediction_df
        .copy()
        .set_index(
            "case_id"
        )
    )

    missing = (
        set(
            case_ids
        )
        - set(
            p.index
        )
    )

    extra = (
        set(
            p.index
        )
        - set(
            case_ids
        )
    )

    if missing:
        raise RuntimeError(
            f"{model_label}: missing test predictions: "
            f"{sorted(missing)}"
        )

    if extra:
        raise RuntimeError(
            f"{model_label}: unexpected extra predictions: "
            f"{sorted(extra)}"
        )

    p = p.loc[
        case_ids
    ]

    return p


def make_pairwise_table(
    y_true,
    score_dict,
):
    rows = []

    def add_pair(
        method1,
        method2,
    ):
        d = delong_test(
            y_true,
            score_dict[
                method1
            ],
            score_dict[
                method2
            ],
        )

        b = bootstrap_auc_diff(
            y_true,
            score_dict[
                method1
            ],
            score_dict[
                method2
            ],
        )

        rows.append(
            {
                "Method 1": method1,
                "Method 2": method2,
                "AUC_1": d[
                    0
                ],
                "AUC_2": d[
                    1
                ],
                "Delta AUC": d[
                    0
                ]
                - d[
                    1
                ],
                "DeLong z": d[
                    3
                ],
                "DeLong p": d[
                    4
                ],
                "Bootstrap mean Delta AUC":
                    b[
                        0
                    ],
                "Bootstrap 95% CI low":
                    b[
                        1
                    ],
                "Bootstrap 95% CI high":
                    b[
                        2
                    ],
                "Bootstrap p":
                    b[
                        3
                    ],
            }
        )

    model_methods = [
        MODEL_DISPLAY[
            "Model 4"
        ],
        MODEL_DISPLAY[
            "Model 5"
        ],
        MODEL_DISPLAY[
            "Model 6"
        ],
    ]

    for model_method in model_methods:
        add_pair(
            "Senior Radiologist",
            model_method,
        )

        add_pair(
            "Junior Radiologist",
            model_method,
        )

    for i in range(
        len(
            model_methods
        )
    ):
        for j in range(
            i + 1,
            len(
                model_methods
            ),
        ):
            add_pair(
                model_methods[
                    i
                ],
                model_methods[
                    j
                ],
            )

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        n_comp = len(
            df
        )

        boot_p = df[
            "Bootstrap p"
        ].to_numpy(
            dtype=float
        )

        delong_p = df[
            "DeLong p"
        ].to_numpy(
            dtype=float
        )

        df[
            "Bootstrap p (Bonferroni)"
        ] = np.minimum(
            boot_p
            * n_comp,
            1.0,
        )

        _, boot_fdr, _, _ = multipletests(
            boot_p,
            method="fdr_bh",
        )

        df[
            "Bootstrap p (BH FDR)"
        ] = boot_fdr

        df[
            "DeLong p (Bonferroni)"
        ] = np.minimum(
            delong_p
            * n_comp,
            1.0,
        )

        _, delong_fdr, _, _ = multipletests(
            delong_p,
            method="fdr_bh",
        )

        df[
            "DeLong p (BH FDR)"
        ] = delong_fdr

    return df


def evaluate_one_cohort(
    cohort_name,
    spec,
    model_specs,
    prediction_dfs,
    cohort_dir,
):
    case_ids, labels = load_fixed_test_reference(
        spec
    )

    radio = load_radiologist_data(
        cohort_name=cohort_name,
        spec=spec,
        case_ids=case_ids,
        labels=labels,
    )

    y_true = np.asarray(
        labels,
        dtype=int,
    )

    aligned = {}

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        aligned[
            model_label
        ] = align_predictions_to_case_order(
            prediction_dfs[
                model_label
            ],
            case_ids,
            model_label,
        )

        if not np.array_equal(
            aligned[
                model_label
            ][
                "y_true"
            ].to_numpy(
                dtype=int
            ),
            y_true,
        ):
            raise RuntimeError(
                f"{cohort_name}/{model_label}: "
                f"prediction CSV pathology labels "
                f"do not match fixed reference labels."
            )

    detail = radio.copy()

    score_dict = {
        "Senior Radiologist":
            radio[
                "senior_radio_score"
            ].to_numpy(
                dtype=float
            ),
        "Junior Radiologist":
            radio[
                "junior_radio_score"
            ].to_numpy(
                dtype=float
            ),
    }

    pred_dict = {
        "Senior Radiologist":
            radio[
                "senior_radio_pred"
            ].to_numpy(
                dtype=int
            ),
        "Junior Radiologist":
            radio[
                "junior_radio_pred"
            ].to_numpy(
                dtype=int
            ),
    }

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        display = MODEL_DISPLAY[
            model_label
        ]

        scores = aligned[
            model_label
        ][
            "prob_class1"
        ].to_numpy(
            dtype=float
        )

        preds = aligned[
            model_label
        ][
            "y_pred"
        ].to_numpy(
            dtype=int
        )

        score_dict[
            display
        ] = scores

        pred_dict[
            display
        ] = preds

        detail[
            f"{model_label}_prob_class1"
        ] = scores

        detail[
            f"{model_label}_pred"
        ] = preds

        detail[
            f"{model_label}_threshold"
        ] = float(
            model_specs[
                model_label
            ][
                "threshold"
            ]
        )

        detail[
            f"{model_label}_fold"
        ] = int(
            model_specs[
                model_label
            ][
                "fold"
            ]
        )

    # --------------------
    # Point-estimate summary
    # --------------------
    summary_rows = []

    summary_rows.append(
        {
            "Method":
                "Senior Radiologist",
            **compute_metrics(
                y_true,
                pred_dict[
                    "Senior Radiologist"
                ],
                score_dict[
                    "Senior Radiologist"
                ],
            ),
        }
    )

    summary_rows.append(
        {
            "Method":
                "Junior Radiologist",
            **compute_metrics(
                y_true,
                pred_dict[
                    "Junior Radiologist"
                ],
                score_dict[
                    "Junior Radiologist"
                ],
            ),
        }
    )

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        display = MODEL_DISPLAY[
            model_label
        ]

        summary_rows.append(
            {
                "Method":
                    SUMMARY_METHOD[
                        model_label
                    ],
                **compute_metrics(
                    y_true,
                    pred_dict[
                        display
                    ],
                    score_dict[
                        display
                    ],
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------
    # CIs
    # --------------------
    ci_rows = []

    for method in [
        "Senior Radiologist",
        "Junior Radiologist",
    ]:
        point, ci = radiologist_metrics_with_ci(
            y_true,
            pred_dict[
                method
            ],
            score_dict[
                method
            ],
        )

        row = {
            "Method": method
        }

        for metric in [
            "AUC",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "F1",
            "Youden",
        ]:
            row[
                metric
            ] = point[
                metric
            ]

            row[
                f"{metric}_low"
            ] = ci[
                metric
            ][
                0
            ]

            row[
                f"{metric}_high"
            ] = ci[
                metric
            ][
                1
            ]

        ci_rows.append(
            row
        )

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        display = MODEL_DISPLAY[
            model_label
        ]

        point, ci = compute_metrics_bootstrap_ci(
            y_true,
            pred_dict[
                display
            ],
            score_dict[
                display
            ],
            n_boot=1000,
            seed=42,
        )

        row = {
            "Method":
                SUMMARY_METHOD[
                    model_label
                ]
        }

        for metric in [
            "AUC",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "F1",
            "Youden",
        ]:
            row[
                metric
            ] = point[
                metric
            ]

            row[
                f"{metric}_low"
            ] = ci[
                metric
            ][
                0
            ]

            row[
                f"{metric}_high"
            ] = ci[
                metric
            ][
                1
            ]

        ci_rows.append(
            row
        )

    ci_df = pd.DataFrame(
        ci_rows
    )

    # --------------------
    # Pairwise AUC comparisons
    # --------------------
    pairwise_df = make_pairwise_table(
        y_true=y_true,
        score_dict=score_dict,
    )

    # --------------------
    # Figures
    # --------------------
    figure_dir = (
        cohort_dir
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        display = MODEL_DISPLAY[
            model_label
        ]

        save_confusion_matrix_png(
            y_true=y_true,
            y_pred=pred_dict[
                display
            ],
            save_path=(
                figure_dir
                / (
                    f"{cohort_name}_"
                    f"{model_label.replace(' ', '_')}_"
                    f"confusion_matrix.png"
                )
            ),
            title=cohort_name,
        )

    plot_combined_roc(
        y_true=y_true,
        score_dict=score_dict,
        save_path=(
            figure_dir
            / f"{cohort_name}_ROC_with_95CI.pdf"
        ),
        title=cohort_name,
        with_ci=True,
    )

    plot_combined_roc(
        y_true=y_true,
        score_dict=score_dict,
        save_path=(
            figure_dir
            / f"{cohort_name}_ROC_no_CI.pdf"
        ),
        title=cohort_name,
        with_ci=False,
    )

    plot_radar_chart(
        summary_df=summary_df,
        save_path=(
            figure_dir
            / f"{cohort_name}_radar.pdf"
        ),
        title=cohort_name,
    )

    # --------------------
    # Workbook
    # --------------------
    workbook = (
        cohort_dir
        / f"{cohort_name}_ResNet34_Model5_full_evaluation.xlsx"
    )

    with pd.ExcelWriter(
        workbook,
        engine="openpyxl",
    ) as writer:
        detail.to_excel(
            writer,
            sheet_name="Detail",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        ci_df.to_excel(
            writer,
            sheet_name="Metrics_with_CI",
            index=False,
        )

        pairwise_df.to_excel(
            writer,
            sheet_name="Pairwise Comparisons",
            index=False,
        )

    # Save CSV versions too.
    summary_df.to_csv(
        cohort_dir
        / f"{cohort_name}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ci_df.to_csv(
        cohort_dir
        / f"{cohort_name}_metrics_with_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pairwise_df.to_csv(
        cohort_dir
        / f"{cohort_name}_pairwise_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model5_method = SUMMARY_METHOD[
        "Model 5"
    ]

    model5_summary = (
        summary_df[
            summary_df[
                "Method"
            ] == model5_method
        ]
        .iloc[0]
        .to_dict()
    )

    model5_ci = (
        ci_df[
            ci_df[
                "Method"
            ] == model5_method
        ]
        .iloc[0]
        .to_dict()
    )

    return {
        "detail_df": detail,
        "summary_df": summary_df,
        "ci_df": ci_df,
        "pairwise_df": pairwise_df,
        "model5_summary": model5_summary,
        "model5_ci": model5_ci,
        "workbook": workbook,
        "figure_dir": figure_dir,
    }



# ============================================================
# 17B. Human-readable Excel path audit
# ============================================================

def build_human_review_path_table(
    cohort_name,
    case_ids,
    labels,
    manifests,
    model_specs,
):
    """
    One row per patient, designed for manual review.

    Columns explicitly show:
      - fixed test cohort
      - case ID
      - pathology label
      - exact T2WI path
      - exact Model 4 manual-mask path
      - exact Model 5 ORIGINAL test predicted-mask path
      - Model 6 uses image only
      - exact checkpoint/fold/threshold for Models 4-6
    """

    m1 = manifests["Model 4"]["audit_df"].copy()
    m2 = manifests["Model 5"]["audit_df"].copy()
    m3 = manifests["Model 6"]["audit_df"].copy()

    m1 = m1.set_index("case_id_folder").loc[case_ids]
    m2 = m2.set_index("case_id_folder").loc[case_ids]
    m3 = m3.set_index("case_id_folder").loc[case_ids]

    if not (
        m1["image_path"].tolist()
        == m2["image_path"].tolist()
        == m3["image_path"].tolist()
    ):
        raise RuntimeError(
            f"{cohort_name}: T2WI paths differ across models while building Excel audit."
        )

    rows = []

    for case_id, label in zip(case_ids, labels):
        rows.append({
            "test_set": cohort_name,
            "case_id": case_id,
            "pathology_binary_label": int(label),

            "T2WI_path": str(
                m1.loc[case_id, "image_path"]
            ),

            "Model4_input_mode": model_specs["Model 4"]["input_mode"],
            "Model4_manual_mask_path": str(
                m1.loc[case_id, "mask_path"]
            ),
            "Model4_checkpoint": str(
                model_specs["Model 4"]["checkpoint"]
            ),
            "Model4_fold": int(
                model_specs["Model 4"]["fold"]
            ),
            "Model4_threshold": float(
                model_specs["Model 4"]["threshold"]
            ),

            "Model5_input_mode": model_specs["Model 5"]["input_mode"],
            "Model5_predicted_mask_path": str(
                m2.loc[case_id, "mask_path"]
            ),
            "Model5_mask_source": "same independent-test *_pred.nii.gz as DenseNet121 Model 2",
            "Model5_checkpoint": str(
                model_specs["Model 5"]["checkpoint"]
            ),
            "Model5_fold": int(
                model_specs["Model 5"]["fold"]
            ),
            "Model5_threshold": float(
                model_specs["Model 5"]["threshold"]
            ),
            "Model5_selected_validation_AUC": float(
                model_specs["Model 5"]["validation_auc"]
            ),

            "Model6_input_mode": model_specs["Model 6"]["input_mode"],
            "Model6_mask_path": "",
            "Model6_checkpoint": str(
                model_specs["Model 6"]["checkpoint"]
            ),
            "Model6_fold": int(
                model_specs["Model 6"]["fold"]
            ),
            "Model6_threshold": float(
                model_specs["Model 6"]["threshold"]
            ),

            "Model4_vs_Model5_mask_paths_different": (
                str(m1.loc[case_id, "mask_path"])
                != str(m2.loc[case_id, "mask_path"])
            ),
            "T2WI_same_for_all_3_models": True,
        })

    return pd.DataFrame(rows)


def write_cohort_path_audit_excel(
    cohort_name,
    cohort_dir,
    case_ids,
    labels,
    manifests,
    model_specs,
):
    """
    Create one audit workbook per cohort.
    """

    review_df = build_human_review_path_table(
        cohort_name=cohort_name,
        case_ids=case_ids,
        labels=labels,
        manifests=manifests,
        model_specs=model_specs,
    )

    excel_path = (
        cohort_dir
        / f"{cohort_name}_ALL_INPUT_PATHS_FOR_REVIEW.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        # Most useful sheet: one patient per row.
        review_df.to_excel(
            writer,
            sheet_name="Patient_Path_Audit",
            index=False,
        )

        # Exact per-model manifests used by the inference pipeline.
        manifests["Model 4"]["audit_df"].to_excel(
            writer,
            sheet_name="Model4_ManualMask",
            index=False,
        )

        manifests["Model 5"]["audit_df"].to_excel(
            writer,
            sheet_name="Model5_PredMask",
            index=False,
        )

        manifests["Model 6"]["audit_df"].to_excel(
            writer,
            sheet_name="Model6_ImageOnly",
            index=False,
        )

        # Model-level provenance in a compact table.
        model_rows = []

        for model_label in [
            "Model 4",
            "Model 5",
            "Model 6",
        ]:
            spec = model_specs[model_label]

            model_rows.append({
                "model": model_label,
                "input_mode": spec["input_mode"],
                "fold": int(spec["fold"]),
                "threshold": float(spec["threshold"]),
                "checkpoint": str(spec["checkpoint"]),
                "selection_source": str(spec["selection_source"]),
                "selection_basis": spec["selection_basis"],
                "selected_validation_AUC": (
                    float(spec["validation_auc"])
                    if model_label == "Model 5"
                    else np.nan
                ),
            })

        pd.DataFrame(
            model_rows
        ).to_excel(
            writer,
            sheet_name="Model_Provenance",
            index=False,
        )

    return excel_path, review_df


# ============================================================
# 18. Run one full test cohort
# ============================================================

def run_one_cohort(
    cohort_name,
    spec,
    model_specs,
    device,
):
    print_header(
        f"{cohort_name} - {spec['description']}"
    )

    cohort_dir = (
        NEW_RESULT_ROOT
        / cohort_name
    )

    prediction_dir = (
        cohort_dir
        / "predictions"
    )

    manifest_dir = (
        cohort_dir
        / "input_manifests"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_ids, labels = load_fixed_test_reference(
        spec
    )

    print(
        f"Cases directory : "
        f"{spec['cases_dir']}"
    )

    print(
        f"Reference labels: "
        f"{spec['label_file']}"
    )

    # Build all 3 exact input manifests first.
    manifests = {}

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        input_mode = INPUT_MODE_FOR_MODEL[
            model_label
        ]

        full_manifest = build_model_input_manifest(
            cohort_name=cohort_name,
            spec=spec,
            case_ids=case_ids,
            labels=labels,
            model_label=model_label,
            input_mode=input_mode,
        )

        audit_csv = (
            manifest_dir
            / (
                f"{cohort_name}_"
                f"{model_label.replace(' ', '_')}_"
                f"input_audit.csv"
            )
        )

        full_manifest.to_csv(
            audit_csv,
            index=False,
            encoding="utf-8-sig",
        )

        # Loader-compatible CSV: no audit helper columns.
        loader_cols = [
            "case_id_folder",
            "label",
            "image_path",
        ]

        if (
            "mask_path"
            in full_manifest.columns
        ):
            loader_cols.append(
                "mask_path"
            )

        loader_manifest = (
            full_manifest[
                loader_cols
            ].copy()
        )

        loader_csv = (
            manifest_dir
            / (
                f"{cohort_name}_"
                f"{model_label.replace(' ', '_')}_"
                f"loader_input.csv"
            )
        )

        loader_manifest.to_csv(
            loader_csv,
            index=False,
            encoding="utf-8-sig",
        )

        manifests[
            model_label
        ] = {
            "audit_df":
                full_manifest,
            "audit_csv":
                audit_csv,
            "loader_csv":
                loader_csv,
        }

    # Cross-model input identity checks.
    m1 = manifests[
        "Model 4"
    ][
        "audit_df"
    ]

    m2 = manifests[
        "Model 5"
    ][
        "audit_df"
    ]

    m3 = manifests[
        "Model 6"
    ][
        "audit_df"
    ]

    if not (
        m1[
            "case_id_folder"
        ].tolist()
        == m2[
            "case_id_folder"
        ].tolist()
        == m3[
            "case_id_folder"
        ].tolist()
    ):
        raise RuntimeError(
            f"{cohort_name}: case IDs differ "
            f"across Model 4/5/6."
        )

    if not (
        m1[
            "label"
        ].tolist()
        == m2[
            "label"
        ].tolist()
        == m3[
            "label"
        ].tolist()
    ):
        raise RuntimeError(
            f"{cohort_name}: pathology labels differ "
            f"across Model 4/5/6."
        )

    if not (
        m1[
            "image_path"
        ].tolist()
        == m2[
            "image_path"
        ].tolist()
        == m3[
            "image_path"
        ].tolist()
    ):
        raise RuntimeError(
            f"{cohort_name}: T2WI paths differ "
            f"across Model 4/5/6."
        )

    if not all(
        str(
            p
        ).endswith(
            "_mask.nii.gz"
        )
        and not str(
            p
        ).endswith(
            "_pred.nii.gz"
        )
        for p in m1[
            "mask_path"
        ].tolist()
    ):
        raise RuntimeError(
            f"{cohort_name}: invalid Model 4 mask paths."
        )

    if not all(
        str(
            p
        ).endswith(
            "_pred.nii.gz"
        )
        for p in m2[
            "mask_path"
        ].tolist()
    ):
        raise RuntimeError(
            f"{cohort_name}: invalid Model 5 predicted-mask paths."
        )

    if (
        "mask_path"
        in m3.columns
    ):
        raise RuntimeError(
            f"{cohort_name}: Model 6 unexpectedly "
            f"contains mask input."
        )

    same_mask_path = (
        np.asarray(
            m1[
                "mask_path"
            ].astype(
                str
            )
        )
        == np.asarray(
            m2[
                "mask_path"
            ].astype(
                str
            )
        )
    )

    if np.any(
        same_mask_path
    ):
        raise RuntimeError(
            f"{cohort_name}: Model 4 manual masks and "
            f"Model 5 predicted masks are not distinct."
        )

    print(
        "Input audit passed:"
    )

    print(
        f"  cases identical           : "
        f"{len(case_ids)}/{len(case_ids)}"
    )

    print(
        f"  pathology labels identical: "
        f"{len(case_ids)}/{len(case_ids)}"
    )

    print(
        f"  T2WI identical            : "
        f"{len(case_ids)}/{len(case_ids)}"
    )

    print(
        f"  Model 4 = manual masks    : "
        f"{len(case_ids)}/{len(case_ids)}"
    )

    print(
        f"  Model 5 = ORIGINAL "
        f"test *_pred.nii.gz: "
        f"{len(case_ids)}/{len(case_ids)}"
    )

    print(
        "  Model 6 = image only      : yes"
    )

    # --------------------------------------------------------
    # Human-readable Excel audit BEFORE inference.
    # This lets the user inspect every patient/file path used.
    # --------------------------------------------------------
    path_audit_excel, path_review_df = write_cohort_path_audit_excel(
        cohort_name=cohort_name,
        cohort_dir=cohort_dir,
        case_ids=case_ids,
        labels=labels,
        manifests=manifests,
        model_specs=model_specs,
    )

    print(
        f"  Path-audit Excel          : {path_audit_excel}"
    )

    # Run all 3 models.
    prediction_dfs = {}

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        spec_model = model_specs[
            model_label
        ]

        output_csv = (
            prediction_dir
            / (
                f"{cohort_name}_"
                f"{model_label.replace(' ', '_')}_"
                f"predictions.csv"
            )
        )

        prediction_dfs[
            model_label
        ] = predict_from_manifest(
            model_spec=spec_model,
            manifest_csv=(
                manifests[
                    model_label
                ][
                    "loader_csv"
                ]
            ),
            expected_n=spec[
                "expected_n"
            ],
            output_csv=output_csv,
            device=device,
        )

    evaluation = evaluate_one_cohort(
        cohort_name=cohort_name,
        spec=spec,
        model_specs=model_specs,
        prediction_dfs=prediction_dfs,
        cohort_dir=cohort_dir,
    )

    provenance = {
        "cohort": cohort_name,
        "description": spec[
            "description"
        ],
        "expected_n": spec[
            "expected_n"
        ],
        "evaluated_n": len(
            case_ids
        ),
        "cases_dir": str(
            spec[
                "cases_dir"
            ]
        ),
        "label_file": str(
            spec[
                "label_file"
            ]
        ),
        "label_file_sha256":
            sha256_file(
                spec[
                    "label_file"
                ]
            ),
        "model4": {
            **{
                k: str(v)
                if isinstance(
                    v,
                    Path
                )
                else v
                for k, v
                in model_specs[
                    "Model 4"
                ].items()
            }
        },
        "model5_NEW_OOF": {
            **{
                k: str(v)
                if isinstance(
                    v,
                    Path
                )
                else v
                for k, v
                in model_specs[
                    "Model 5"
                ].items()
            },
            "test_mask_source":
                "same independent-test *_pred.nii.gz as DenseNet121 Model 2",
            "development_oof_masks_used_for_test":
                False,
        },
        "model6": {
            **{
                k: str(v)
                if isinstance(
                    v,
                    Path
                )
                else v
                for k, v
                in model_specs[
                    "Model 6"
                ].items()
            }
        },
        "test_threshold_optimization":
            False,
        "test_model_selection":
            False,
        "workbook": str(
            evaluation[
                "workbook"
            ]
        ),
        "figure_dir": str(
            evaluation[
                "figure_dir"
            ]
        ),
        "path_audit_excel": str(
            path_audit_excel
        ),
    }

    save_json(
        provenance,
        cohort_dir
        / f"{cohort_name}_provenance.json",
    )

    print(
        f"\n{cohort_name} complete."
    )

    print(
        f"Results:\n  {cohort_dir}"
    )

    return {
        "cohort": cohort_name,
        "model5_summary":
            evaluation[
                "model5_summary"
            ],
        "model5_ci":
            evaluation[
                "model5_ci"
            ],
        "summary_df":
            evaluation[
                "summary_df"
            ],
        "path_review_df":
            path_review_df,
        "path_audit_excel":
            path_audit_excel,
    }


# ============================================================
# 19. Main
# ============================================================

def main():
    import argparse
    global FORCED_SHARED_FOLD
    argp = argparse.ArgumentParser(description="Paper-style IT1/IT2/ET test: ResNet34 Models 4,5,6")
    argp.add_argument("--same_fold", type=int, choices=range(5), default=None,
                      help="Preselect the same classifier fold (0-4) for Models 4/5/6 before testing")
    argp.add_argument("--dry_run", action="store_true",
                      help="Verify selected checkpoints and all test files; do not infer/write")
    args = argp.parse_args()
    FORCED_SHARED_FOLD = args.same_fold
    global NEW_RESULT_ROOT
    if FORCED_SHARED_FOLD is not None:
        NEW_RESULT_ROOT = ROOT / f"paper_style_test_results_samefold{FORCED_SHARED_FOLD}"
    print_header("FULL TEST: ResNet34 MODEL 4 (manual), MODEL 5 (predicted), MODEL 6 (T2WI)")
    print("Testing only, no training, no selection by test performance.")
    print("Shared 617-case five-fold assignment is verified using training provenance.")
    print("Model 4: ResNet34 T2WI + manual tumor masks")
    print("Model 5: ResNet34 T2WI + original INDEPENDENT-TEST nnU-Net *_pred.nii.gz")
    print("Model 6: ResNet34 T2WI only")
    print("Fold rule:", f"predeclared common fold {FORCED_SHARED_FOLD}" if FORCED_SHARED_FOLD is not None
          else "each model selected by highest DEVELOPMENT validation AUC")
    print("Patient IDs preserved exactly as in the input manifests (zero-padded where applicable).")

    # Fail on missing/invalid training provenance before creating result directories.
    model_specs = load_all_model_specs()
    if FORCED_SHARED_FOLD is not None and len({m["fold"] for m in model_specs.values()}) != 1:
        raise RuntimeError("--same_fold requested but selected fold numbers differ")
    if args.dry_run:
        print("\nTEST INPUT PREFLIGHT: no results written and no inference")
        for cohort_name, spec in TEST_SPECS.items():
            case_ids, labels = load_fixed_test_reference(spec)
            for model_label in ("Model 4", "Model 5", "Model 6"):
                build_model_input_manifest(cohort_name, spec, case_ids, labels,
                                           model_label, INPUT_MODE_FOR_MODEL[model_label])
            print(f"  {cohort_name}: {len(case_ids)} patients / 3 modes checked")
        print("TEST PREFLIGHT PASSED")
        return

    # Safety.
    if NEW_RESULT_ROOT.exists():
        contents = list(
            NEW_RESULT_ROOT.iterdir()
        )

        if (
            contents
            and not ALLOW_OVERWRITE_NEW_RESULTS
        ):
            raise RuntimeError(
                "SAFETY STOP: ResNet34 Model 4/5/6 "
                "test-result folder already exists and is non-empty.\n"
                f"{NEW_RESULT_ROOT}\n\n"
                "This script will not overwrite a previous test run. "
                "Inspect the folder first, or change NEW_RESULT_ROOT "
                "to a new versioned path."
            )

    NEW_RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_header(
        "FIXED MODEL SPECIFICATIONS"
    )

    for model_label in [
        "Model 4",
        "Model 5",
        "Model 6",
    ]:
        s = model_specs[
            model_label
        ]

        print(
            f"\n{model_label}"
        )

        print(
            f"  input mode : "
            f"{s['input_mode']}"
        )

        print(
            f"  fold       : "
            f"{s['fold']}"
        )

        print(
            f"  threshold  : "
            f"{s['threshold']:.6f}"
        )

        print(
            f"  checkpoint : "
            f"{s['checkpoint']}"
        )

        print(
            f"  selection  : "
            f"{s['selection_basis']}"
        )

        if (
            model_label
            == "Model 5"
        ):
            print(
                f"  dev val AUC: "
                f"{s['validation_auc']:.6f}"
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    results = []

    for cohort_name in [
        "IT1",
        "IT2",
        "ET",
    ]:
        results.append(
            run_one_cohort(
                cohort_name=cohort_name,
                spec=TEST_SPECS[
                    cohort_name
                ],
                model_specs=model_specs,
                device=device,
            )
        )

    # --------------------------------------------
    # Cross-cohort Model 5 summary
    # --------------------------------------------
    model5_rows = []

    all_models_rows = []

    for result in results:
        cohort = result[
            "cohort"
        ]

        model5_summary = result[
            "model5_summary"
        ]

        model5_ci = result[
            "model5_ci"
        ]

        row = {
            "Test set": cohort,
            "N": TEST_SPECS[
                cohort
            ][
                "expected_n"
            ],
            "Selected fold":
                model_specs[
                    "Model 5"
                ][
                    "fold"
                ],
            "Validation AUC used for model selection":
                model_specs[
                    "Model 5"
                ][
                    "validation_auc"
                ],
            "Fixed threshold":
                model_specs[
                    "Model 5"
                ][
                    "threshold"
                ],
        }

        for metric in [
            "AUC",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "F1",
            "Youden",
        ]:
            row[
                metric
            ] = model5_summary[
                metric
            ]

            row[
                f"{metric}_95CI_low"
            ] = model5_ci[
                f"{metric}_low"
            ]

            row[
                f"{metric}_95CI_high"
            ] = model5_ci[
                f"{metric}_high"
            ]

        row[
            "TN"
        ] = model5_summary[
            "TN"
        ]

        row[
            "FP"
        ] = model5_summary[
            "FP"
        ]

        row[
            "FN"
        ] = model5_summary[
            "FN"
        ]

        row[
            "TP"
        ] = model5_summary[
            "TP"
        ]

        model5_rows.append(
            row
        )

        s_df = result[
            "summary_df"
        ].copy()

        s_df.insert(
            0,
            "Test set",
            cohort,
        )

        all_models_rows.append(
            s_df
        )

    model5_summary_df = pd.DataFrame(
        model5_rows
    )

    model5_summary_df.to_csv(
        NEW_RESULT_ROOT
        / "ResNet34_Model5_3cohort_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_models_df = pd.concat(
        all_models_rows,
        ignore_index=True,
    )

    all_models_df.to_csv(
        NEW_RESULT_ROOT
        / "all_models_3cohort_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Cohort-by-model full table with confidence intervals for ALL 3 ResNet34 models.
    all_ci_rows = []
    for result in results:
        cohort = result["cohort"]
        cohort_ci_file = NEW_RESULT_ROOT / cohort / f"{cohort}_metrics_with_CI.csv"
        cdf = pd.read_csv(cohort_ci_file, encoding="utf-8-sig")
        for m in ("Model 4", "Model 5", "Model 6"):
            method = SUMMARY_METHOD[m]
            row = cdf.loc[cdf["Method"] == method]
            if len(row) != 1:
                raise RuntimeError(f"{cohort}: missing {m} metrics CI row")
            item = row.iloc[0].to_dict()
            item.update({"Test set": cohort, "N": TEST_SPECS[cohort]["expected_n"],
                         "Model ID": m, "Architecture": "resnet34_3d",
                         "Input mode": model_specs[m]["input_mode"],
                         "Fold": model_specs[m]["fold"],
                         "Fixed threshold": model_specs[m]["threshold"]})
            all_ci_rows.append(item)
    pd.DataFrame(all_ci_rows).to_csv(
        NEW_RESULT_ROOT / "RESNET34_MODELS_4_5_6_ALL_3_COHORTS_METRICS_WITH_CI.csv",
        index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # MASTER Excel audit across IT1 / IT2 / ET.
    # One workbook for manual review of every physical input path.
    # --------------------------------------------------------
    master_path_audit_excel = (
        NEW_RESULT_ROOT
        / "ALL_3_TEST_SETS_INPUT_PATHS_FOR_REVIEW.xlsx"
    )

    with pd.ExcelWriter(
        master_path_audit_excel,
        engine="openpyxl",
    ) as writer:

        combined_path_rows = []

        for result in results:
            cohort = result["cohort"]
            review_df = result["path_review_df"].copy()

            review_df.to_excel(
                writer,
                sheet_name=f"{cohort}_Patient_Paths",
                index=False,
            )

            combined_path_rows.append(
                review_df
            )

        pd.concat(
            combined_path_rows,
            ignore_index=True,
        ).to_excel(
            writer,
            sheet_name="ALL_PATIENTS",
            index=False,
        )

        model_rows = []

        for model_label in [
            "Model 4",
            "Model 5",
            "Model 6",
        ]:
            spec = model_specs[model_label]

            model_rows.append({
                "model": model_label,
                "input_mode": spec["input_mode"],
                "fold": int(spec["fold"]),
                "threshold": float(spec["threshold"]),
                "checkpoint": str(spec["checkpoint"]),
                "selection_source": str(spec["selection_source"]),
                "selection_basis": spec["selection_basis"],
                "selected_validation_AUC": (
                    float(spec["validation_auc"])
                    if model_label == "Model 5"
                    else np.nan
                ),
            })

        pd.DataFrame(
            model_rows
        ).to_excel(
            writer,
            sheet_name="Model_Provenance",
            index=False,
        )

    # Master provenance
    master = {
        "run_at":
            datetime.now().isoformat(),
        "purpose":
            (
                "Three NEW OOF-pipeline ResNet34 classification models 4/5/6, "
                "with original fixed independent-test inputs "
                "and original radiologist data."
            ),
        "training_performed":
            False,
        "classification_folds_fixed": True,
        "classifier_fold_policy": "predeclared common fold" if FORCED_SHARED_FOLD is not None else "each highest validation AUC",
        "forced_common_fold": FORCED_SHARED_FOLD,
        "training_design_sha256": sha256_file(TRAINING_PROVENANCE),
        "test_model_selection":
            False,
        "test_threshold_optimization":
            False,
        "model5_test_mask_source":
            "same independent-test *_pred.nii.gz as DenseNet121 Model 2",
        "development_oof_masks_used_on_test":
            False,
        "new_result_root":
            str(
                NEW_RESULT_ROOT
            ),
        "master_input_path_audit_excel":
            str(
                master_path_audit_excel
            ),
        "model_specs": {
            label: {
                k: str(v)
                if isinstance(
                    v,
                    Path
                )
                else v
                for k, v
                in spec.items()
            }
            for label, spec
            in model_specs.items()
        },
        "test_sets": {
            name: {
                "description":
                    spec[
                        "description"
                    ],
                "expected_n":
                    spec[
                        "expected_n"
                    ],
                "cases_dir":
                    str(
                        spec[
                            "cases_dir"
                        ]
                    ),
                "label_file":
                    str(
                        spec[
                            "label_file"
                        ]
                    ),
            }
            for name, spec
            in TEST_SPECS.items()
        },
    }

    save_json(
        master,
        NEW_RESULT_ROOT
        / "MASTER_PROVENANCE.json",
    )

    print_header(
        "ALL THREE TEST SETS COMPLETED SUCCESSFULLY"
    )

    print(
        "Expected / evaluated:"
    )

    print(
        "  IT1: 47 / 47"
    )

    print(
        "  IT2: 31 / 31"
    )

    print(
        "  ET : 83 / 83"
    )

    print(
        "\nNEW results root:"
    )

    print(
        f"  {NEW_RESULT_ROOT}"
    )

    print(
        "\nKey cross-cohort files:"
    )

    print("  RESNET34_MODELS_4_5_6_ALL_3_COHORTS_METRICS_WITH_CI.csv")
    print("  ResNet34_Model5_3cohort_summary.csv")

    print(
        "  all_models_3cohort_summary.csv"
    )

    print(
        "  MASTER_PROVENANCE.json"
    )

    print(
        "  ALL_3_TEST_SETS_INPUT_PATHS_FOR_REVIEW.xlsx"
    )

    print(
        "\nEach cohort folder also contains:"
    )

    print(
        "  <IT1/IT2/ET>_ALL_INPUT_PATHS_FOR_REVIEW.xlsx"
    )

    print(
        "\nEach cohort folder contains:"
    )

    print(
        "  predictions/"
    )

    print(
        "  input_manifests/"
    )

    print(
        "  figures/"
    )

    print(
        "  full evaluation Excel workbook"
    )

    print(
        "\nORIGINAL RESULTS WERE NOT MODIFIED."
    )


if __name__ == "__main__":
    main()
