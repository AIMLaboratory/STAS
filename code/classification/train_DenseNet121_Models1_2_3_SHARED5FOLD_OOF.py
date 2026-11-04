# -*- coding: utf-8 -*-
r"""
DenseNet121 Models 1, 2, 3; 617 development patients; ONE fixed five-fold split.

Model 1: axial T2WI + manually delineated tumor mask (*_mask.nii.gz)
Model 2: axial T2WI + nnU-Net OOF predicted mask (*_pred_oof.nii.gz)
Model 3: axial T2WI only

Every model uses exactly the same train/validation patients for folds 0..4.
The ORIGINAL DenseNet121 Model 2 split is the authoritative fixed split.
No independent test cases are read. No historical files are written to.

Run:
  python train_DenseNet121_Models1_2_3_SHARED5FOLD_OOF.py --dry_run
  python train_DenseNet121_Models1_2_3_SHARED5FOLD_OOF.py
  python train_DenseNet121_Models1_2_3_SHARED5FOLD_OOF.py --resume
  python train_DenseNet121_Models1_2_3_SHARED5FOLD_OOF.py --hash_masks --dry_run
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from config import Config
from data_loader import get_data_loader_from_samples
from models import create_model
from trainer import ClassificationTrainer

# ----- Edit only these paths if your project directory differs -----
ROOT = Path(r"RECTAL_STAS_DATA\DenseNet121_Models1_2_3_Shared5Fold_OOF_v1")
OOF_INDEX = Path(
    r"RECTAL_STAS_DATA\nnUNet_OOF_617_exploratory_v1"
    r"\classifier_input\model2_oof_mask_index_617.csv"
)
OOF_PROVENANCE = Path(
    r"RECTAL_STAS_DATA\nnUNet_OOF_617_exploratory_v1"
    r"\provenance\run_provenance.json"
)
# This is intentionally the exact ORIGINAL DenseNet121 Model 2 split file.
SHARED_SPLIT = (HERE / "splits" /
    "T_stage_classification_3D_image_plus_pred_mask_binary_densenet121_3d_5fold_cv_split.json")

ARCHITECTURE = "densenet121_3d"
EXPECTED_N = 617
EXPECTED_FOLDS = 5
LR = 5e-5
DROPOUT = 0.2
MODELS = {
    1: {"mode": "image_plus_gt_mask", "mask_source": "manual GT"},
    2: {"mode": "image_plus_pred_mask", "mask_source": "nnU-Net OOF"},
    3: {"mode": "image_only", "mask_source": "none"},
}
RUN_TAG = "DenseNet121_Models1_2_3_Shared5Fold_OOF_v1"


def log(msg):
    print(msg, flush=True)


def require_file(p, meaning):
    if not Path(p).is_file():
        raise FileNotFoundError(f"{meaning}: {p}")


def digest(p, chunk=1024 * 1024):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_json(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(p):
    with Path(p).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure():
    Config.model_name = ARCHITECTURE
    Config.input_mode = "image_only"
    Config.label_mode = "binary"
    Config.use_cross_validation = True
    Config.n_folds = EXPECTED_FOLDS
    Config.learning_rate = LR
    Config.task_name = RUN_TAG
    Config.experiments_dir = ROOT / "fold_runs"
    Config.splits_dir = ROOT / "split_records"
    if Config.monitor_metric != "auc":
        raise RuntimeError("Expected monitor_metric='auc'.")
    if Config.loss_type != "focal":
        raise RuntimeError("Expected the original focal loss.")
    old_get_config = Config.get_model_config

    def fixed_get_model_config(model_name):
        cfg = dict(old_get_config(model_name))
        if model_name == ARCHITECTURE:
            cfg.update(learning_rate=LR, dropout=DROPOUT)
        return cfg

    Config.get_model_config = staticmethod(fixed_get_model_config)
    if Config.get_model_config(ARCHITECTURE).get("dropout") != DROPOUT:
        raise RuntimeError("Dropout override failed")
    if not np.isclose(Config.get_model_config(ARCHITECTURE).get("learning_rate"), LR):
        raise RuntimeError("Learning rate override failed")


def validate_pair(case_id, image_path, mask_path, suffix, kind):
    """Check actual file pairing in physical coordinates, not only case-ID strings."""
    if image_path.name != f"{case_id}_image.nii.gz":
        raise ValueError(f"Wrong T2WI filename for {case_id}: {image_path}")
    if mask_path.name != f"{case_id}{suffix}":
        raise ValueError(f"Wrong {kind} filename for {case_id}: {mask_path}")
    require_file(image_path, f"{case_id} image")
    require_file(mask_path, f"{case_id} {kind}")
    img = nib.load(str(image_path))
    msk = nib.load(str(mask_path))
    a, b = tuple(img.shape), tuple(msk.shape)
    if len(a) != 3 or a != b:
        raise ValueError(f"{case_id}: image/mask shape mismatch: {a} vs {b}")
    d = int(np.argmin(a))
    dhw = (a[d],) + tuple(a[i] for i in range(3) if i != d)
    if dhw != tuple(Config.input_size_3d):
        raise ValueError(f"{case_id}: unexpected image shape DHW={dhw}")
    if not np.allclose(img.affine, msk.affine, rtol=0, atol=1e-5):
        raise ValueError(f"{case_id}: image/{kind} affine differs")
    mask = np.asarray(msk.dataobj)
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"{case_id}: {kind} contains NaN or Inf")
    if not np.all(np.isclose(mask, 0, atol=1e-6) | np.isclose(mask, 1, atol=1e-6)):
        raise ValueError(f"{case_id}: {kind} is not binary (0/1)")
    return {"nifti_shape": list(a), "dhw_shape": list(dhw), "affine_equal": True}


def build_617_samples(hash_masks=False):
    require_file(OOF_INDEX, "OOF-mask patient index")
    df = pd.read_csv(OOF_INDEX, dtype={"source_case_id": str, "nnunet_case_id": str},
                     encoding="utf-8-sig")
    required = {"source_case_id", "nnunet_case_id", "oof_fold", "oof_mask"}
    if not required.issubset(df.columns):
        raise KeyError(f"OOF index missing columns: {sorted(required-set(df.columns))}")
    if len(df) != EXPECTED_N:
        raise ValueError(f"OOF index has {len(df)} cases, expected {EXPECTED_N}")
    for col in ("source_case_id", "nnunet_case_id"):
        if df[col].isna().any() or df[col].astype(str).str.strip().duplicated().any():
            raise ValueError(f"OOF index has invalid or duplicate {col}")
    if not set(pd.to_numeric(df["oof_fold"], errors="raise").astype(int)).issubset(set(range(5))):
        raise ValueError("Unexpected segmentation OOF-fold index")

    labels = Config.load_train_labels()
    cases = []
    for row in df.to_dict("records"):
        cid = str(row["source_case_id"]).strip()
        if cid not in labels:
            raise KeyError(f"Missing label for development case {cid}")
        image = Config.cases_root / cid / f"{cid}_image.nii.gz"
        gt = Config.cases_root / cid / f"{cid}_mask.nii.gz"
        oof = Path(str(row["oof_mask"]).strip())
        validate_pair(cid, image, gt, "_mask.nii.gz", "manual GT")
        validate_pair(cid, image, oof, "_pred_oof.nii.gz", "nnU-Net OOF")
        item = {
            "case_id": cid, "image_path": str(image), "gt_mask_path": str(gt),
            "pred_mask_path": str(oof), "label": int(labels[cid]["target_label"]),
            "raw_label": labels[cid]["raw_label"],
            "normalized_label": labels[cid]["normalized_label"],
            "nnunet_case_id": str(row["nnunet_case_id"]).strip(),
            "segmentation_oof_fold": int(row["oof_fold"]),
        }
        if item["label"] not in (0, 1):
            raise ValueError(f"{cid}: unexpected class label {item['label']}")
        if hash_masks:
            item["gt_mask_sha256"] = digest(gt)
            item["oof_mask_sha256"] = digest(oof)
        cases.append(item)

    if len(set(s["case_id"] for s in cases)) != EXPECTED_N:
        raise RuntimeError("617 unique case IDs required")
    return cases


def read_fixed_split(samples):
    require_file(SHARED_SPLIT, "original DenseNet121 Model 2 CV split")
    info = load_json(SHARED_SPLIT)
    if "folds" not in info and "split" in info:
        info = info["split"]
    raw_folds = info.get("folds", [])
    if len(raw_folds) != EXPECTED_FOLDS:
        raise ValueError("Original CV split is not five-fold")
    known = {s["case_id"] for s in samples}
    val_appearances = []
    folds = []
    for fid, row in enumerate(raw_folds):
        if int(row["fold"]) != fid:
            raise ValueError(f"Split fold IDs must follow 0,1,2,3,4; got {row['fold']}")
        train = [str(v).strip() for v in row["train_case_ids"]]
        val = [str(v).strip() for v in row["val_case_ids"]]
        if (len(train) != len(set(train)) or len(val) != len(set(val)) or
            (set(train) & set(val)) or set(train).union(val) != known):
            raise ValueError(f"Fold {fid}: duplicate, overlapping, or missing patients")
        val_appearances.extend(val)
        folds.append({"fold": fid, "train_ids": train, "val_ids": val})
    if len(val_appearances) != EXPECTED_N or set(val_appearances) != known:
        raise ValueError("Each of 617 development patients must occur once in validation")
    return folds


def root_metadata(cases, folds):
    return {
        "run_tag": RUN_TAG,
        "architecture": ARCHITECTURE,
        "created_at": datetime.now().isoformat(),
        "n_development": len(cases),
        "models": {str(k): v for k, v in MODELS.items()},
        "shared_classifier_split": str(SHARED_SPLIT),
        "shared_classifier_split_sha256": digest(SHARED_SPLIT),
        "oof_index": str(OOF_INDEX),
        "oof_index_sha256": digest(OOF_INDEX),
        "same_patient_folds_across_models": True,
        "folds": [{"fold": f["fold"], "train_case_ids": f["train_ids"],
                    "val_case_ids": f["val_ids"]} for f in folds],
        "model_config": Config.get_model_config(ARCHITECTURE),
        "normalization": Config.intensity_normalization,
        "volume_shape_dhw": list(Config.input_size_3d),
        "num_epochs": Config.num_epochs,
        "batch_size": Config.get_model_config(ARCHITECTURE).get("batch_size", Config.batch_size),
        "loss_type": Config.loss_type,
        "focal_alpha": Config.focal_alpha,
        "focal_gamma": Config.focal_gamma,
        "weight_decay": Config.weight_decay,
        "augmentation": Config.use_augmentation,
        "monitor_metric": Config.monitor_metric,
        "early_stopping_patience": Config.early_stopping_patience,
        "test_data_read_for_training": False,
        "selection_uses_test_performance": False,
    }


def create_or_resume_root(cases, folds, resume=False):
    metadata = root_metadata(cases, folds)
    manifest = pd.DataFrame(cases).sort_values("case_id")
    if ROOT.exists() and any(ROOT.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory nonempty; use --resume after checking it: {ROOT}")
        old_file = ROOT / "provenance" / "TRAINING_DESIGN.json"
        require_file(old_file, "previous training provenance for --resume")
        previous = load_json(old_file)
        fields = ["architecture", "shared_classifier_split_sha256", "oof_index_sha256",
                  "models", "model_config", "normalization", "volume_shape_dhw",
                  "num_epochs", "batch_size", "loss_type", "focal_alpha", "focal_gamma",
                  "weight_decay", "augmentation", "monitor_metric", "early_stopping_patience"]
        for field in fields:
            if previous.get(field) != metadata.get(field):
                raise RuntimeError(f"--resume refused: changed training parameter {field}")
        older_manifest = ROOT / "provenance" / "development_617_SHARED_inputs.csv"
        require_file(older_manifest, "previous patient manifest")
        old = pd.read_csv(older_manifest, dtype={"case_id": str}, encoding="utf-8-sig")
        new = manifest.copy()
        if not old.fillna("").astype(str).equals(new.fillna("").astype(str)):
            raise RuntimeError("--resume refused: development input manifest changed")
        log("Previous run validated; completed folds will be skipped.")
        return

    if resume:
        raise RuntimeError("--resume requires an existing experiment root")
    (ROOT / "provenance").mkdir(parents=True)
    save_json(metadata, ROOT / "provenance" / "TRAINING_DESIGN.json")
    manifest.to_csv(ROOT / "provenance" / "development_617_SHARED_inputs.csv",
                    index=False, encoding="utf-8-sig")
    shutil.copy2(SHARED_SPLIT, ROOT / "provenance" / "ORIGINAL_Model2_5fold_SPLIT_REUSED.json")
    shutil.copy2(OOF_INDEX, ROOT / "provenance" / "OOF_INDEX_617_REUSED.csv")
    if OOF_PROVENANCE.exists():
        shutil.copy2(OOF_PROVENANCE, ROOT / "provenance" / "OOF_SEGMENTATION_PROVENANCE.json")
    membership = []
    for f in folds:
        for role, ids in (("train", f["train_ids"]), ("val", f["val_ids"])):
            membership += [{"model": f"Model {m}", "fold": f["fold"],
                            "role": role, "case_id": cid} for m in MODELS for cid in ids]
    pd.DataFrame(membership).to_csv(ROOT / "provenance" / "ALL_3_MODELS_5FOLD_MEMBERSHIP.csv",
                                     index=False, encoding="utf-8-sig")


def training_record_path(model_id, fold_id):
    return ROOT / f"Model{model_id}" / "fold_summaries" / f"fold_{fold_id}_training_result.json"


def train_fold(model_id, fold, by_id):
    mode = MODELS[model_id]["mode"]
    fid = int(fold["fold"])
    # Same seed for equivalent fold numbers across all 3 models.
    set_seed(42 + fid)
    Config.input_mode = mode
    Config.model_name = ARCHITECTURE
    Config.task_name = f"{RUN_TAG}_Model{model_id}_fold{fid}"
    Config.experiments_dir = ROOT / f"Model{model_id}" / "fold_runs"

    train_samples = [by_id[cid] for cid in fold["train_ids"]]
    val_samples = [by_id[cid] for cid in fold["val_ids"]]
    log(f"\n{'='*90}\nMODEL {model_id} ({mode}) / FOLD {fid}: "
        f"train={len(train_samples)} val={len(val_samples)}\n{'='*90}")
    labels_for_weights = {s["case_id"]: {"target_label": s["label"]}
                          for s in train_samples}
    Config.class_weights = Config.calculate_class_weights(labels_for_weights)
    bs = int(Config.get_model_config(ARCHITECTURE).get("batch_size", Config.batch_size))
    tr = get_data_loader_from_samples(train_samples, bs, is_train=True)
    va = get_data_loader_from_samples(val_samples, bs, is_train=False)
    model = create_model(model_name=ARCHITECTURE)
    trainer = ClassificationTrainer(model, model_name=ARCHITECTURE)
    result = trainer.fit(tr, va, start_epoch=1)
    record = {
        "model": f"Model {model_id}", "model_id": model_id,
        "model_name": ARCHITECTURE, "input_mode": mode,
        "fold": fid, "train_cases": len(train_samples), "val_cases": len(val_samples),
        "train_ids_sha256": hashlib.sha256("\n".join(fold["train_ids"]).encode()).hexdigest(),
        "val_ids_sha256": hashlib.sha256("\n".join(fold["val_ids"]).encode()).hexdigest(),
        "best_validation_auc": float(result["best_score"]),
        "best_epoch": int(result["best_epoch"]),
        "best_threshold": float(result["best_threshold"]),
        "best_model_path": str(result["best_model_path"]),
        "final_model_path": str(result["final_model_path"]),
        "experiment_dir": str(result["experiment_dir"]),
        "shared_classifier_split_sha256": digest(SHARED_SPLIT),
        "oof_index_sha256": digest(OOF_INDEX),
    }
    if not np.isfinite(record["best_validation_auc"]):
        raise RuntimeError(f"Model {model_id} fold {fid}: invalid validation AUC")
    save_json(record, training_record_path(model_id, fid))
    del trainer, model, tr, va
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def gather_and_select(model_id):
    records = []
    for fid in range(EXPECTED_FOLDS):
        path = training_record_path(model_id, fid)
        require_file(path, f"Model {model_id} fold {fid} summary")
        r = load_json(path)
        if r["model_id"] != model_id or r["fold"] != fid:
            raise RuntimeError(f"Model {model_id} fold {fid} record ID mismatch")
        if r["input_mode"] != MODELS[model_id]["mode"]:
            raise RuntimeError(f"Model {model_id} fold {fid} input mode mismatch")
        if r["shared_classifier_split_sha256"] != digest(SHARED_SPLIT):
            raise RuntimeError(f"Model {model_id} fold {fid} split hash mismatch")
        require_file(Path(r["best_model_path"]), "fold best checkpoint")
        records.append(r)
    records.sort(key=lambda r: (-r["best_validation_auc"], r["fold"]))
    selected = records[0]
    model_root = ROOT / f"Model{model_id}"
    (model_root / "selected_model").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(model_root / "selected_model" / "all_folds_validation_summary.csv",
                                 index=False, encoding="utf-8-sig")
    save_json({"model_id": model_id, "selection_rule": "highest validation AUC; tie -> lowest fold",
               "selected_fold": selected["fold"], "all_fold_records": records},
              model_root / "selected_model" / "selected_fold.json")
    completion = {
        "run_tag": RUN_TAG, "model_label": f"Model {model_id}",
        "model_id": model_id, "model_name": ARCHITECTURE,
        "input_mode": MODELS[model_id]["mode"],
        "mask_source": MODELS[model_id]["mask_source"],
        "selected_fold": selected["fold"],
        "selected_validation_auc": selected["best_validation_auc"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_threshold": selected["best_threshold"],
        "selected_best_model_path": selected["best_model_path"],
        "all_five_classifier_folds_trained": True,
        "shared_classifier_split_sha256": digest(SHARED_SPLIT),
        "oof_index_sha256": digest(OOF_INDEX),
        "selected_by": "development validation AUC only",
        "test_performance_used_for_selection": False,
        "created_at": datetime.now().isoformat(),
    }
    save_json(completion, model_root / "selected_model" / "TRAINING_COMPLETE_selected_model.json")
    log(f"Model {model_id} complete: selected fold={selected['fold']}, "
        f"validation AUC={selected['best_validation_auc']:.5f}, "
        f"threshold={selected['best_threshold']:.6f}")
    return completion


def main():
    p = argparse.ArgumentParser(description="Train DenseNet121 Models 1/2/3: one shared 5-fold split")
    p.add_argument("--dry_run", action="store_true", help="Check 617 cases and all 5 folds; do not train")
    p.add_argument("--resume", action="store_true", help="Skip finished folds in a verified existing run")
    p.add_argument("--hash_masks", action="store_true", help="Also hash all manual and OOF masks")
    args = p.parse_args()
    configure()
    log("DenseNet121 Models 1/2/3; 617 development cases; one fixed five-fold split")
    log(f"Shared original DenseNet121 split: {SHARED_SPLIT}")
    log(f"Model 2 ORIGINAL OOF index: {OOF_INDEX}")
    cases = build_617_samples(hash_masks=args.hash_masks)
    folds = read_fixed_split(cases)
    for f in folds:
        log(f"  fold {f['fold']}: train={len(f['train_ids'])}, val={len(f['val_ids'])}")
    if args.dry_run:
        log("DRY RUN PASSED: all 617 images + manual GT + OOF masks, identical five folds; no test read")
        return
    create_or_resume_root(cases, folds, resume=args.resume)
    by_id = {r["case_id"]: r for r in cases}
    selected = {}
    for model_id in (1, 2, 3):
        Config.input_mode = MODELS[model_id]["mode"]
        log(f"\n========== TRAINING MODEL {model_id}: {Config.input_mode} ==========")
        for fold in folds:
            dest = training_record_path(model_id, fold["fold"])
            if args.resume and dest.exists():
                r = load_json(dest)
                if r.get("model_id") != model_id or r.get("fold") != fold["fold"]:
                    raise RuntimeError(f"--resume: invalid record {dest}")
                require_file(Path(r["best_model_path"]), "previous best checkpoint")
                log(f"Skipping existing Model {model_id} fold {fold['fold']}: {dest}")
            else:
                train_fold(model_id, fold, by_id)
        selected[str(model_id)] = gather_and_select(model_id)
    save_json({"run_tag": RUN_TAG, "finished_at": datetime.now().isoformat(),
               "models": selected, "train_test_isolation": True},
              ROOT / "TRAINING_COMPLETE_ALL_MODELS_1_2_3.json")
    log("ALL 15 CLASSIFICATION RUNS FINISHED; independent test sets were not accessed.")


if __name__ == "__main__":
    main()