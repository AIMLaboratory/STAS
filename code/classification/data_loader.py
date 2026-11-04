import os
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

from config import Config


# =========================================================

# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_dhw(volume: np.ndarray):
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume.shape}")
    d_axis = int(np.argmin(volume.shape))
    volume = np.moveaxis(volume, d_axis, 0)
    return volume

def zscore_nonzero(volume: np.ndarray):
    volume = volume.astype(np.float32)
    mask = volume != 0
    if np.sum(mask) == 0:
        return volume
    vals = volume[mask]
    mean = vals.mean()
    std = vals.std()
    if std < 1e-8:
        out = volume.copy()
        out[mask] = out[mask] - mean
        return out
    out = volume.copy()
    out[mask] = (out[mask] - mean) / std
    return out

def percentile_normalize(volume: np.ndarray, lower=1, upper=99):
    volume = volume.astype(np.float32)
    p_low = np.percentile(volume, lower)
    p_high = np.percentile(volume, upper)
    if p_high <= p_low + 1e-8:
        vmin, vmax = volume.min(), volume.max()
        if vmax <= vmin + 1e-8:
            return np.zeros_like(volume, dtype=np.float32)
        return (volume - vmin) / (vmax - vmin + 1e-8)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume

def simple_3d_augmentation(volume: np.ndarray):
    if random.random() < 0.5:
        volume = volume[:, :, :, ::-1].copy()
    if random.random() < 0.5:
        volume = volume[:, :, ::-1, :].copy()
    if random.random() < 0.3:
        volume = volume[:, ::-1, :, :].copy()
    return volume


# =========================================================
# 2. 3D Dataset
# =========================================================
class RectalMRIClassification3DDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train
        self.input_mode = Config.input_mode
        print(f"Created {'training' if is_train else 'validation/test'} dataset with {len(samples)} cases")

    def __len__(self):
        return len(self.samples)

    def _load_nii_volume(self, path):
        nii = nib.load(str(path))
        data = nii.get_fdata().astype(np.float32)
        data = to_dhw(data)
        return data

    def _normalize_image(self, image):
        if Config.intensity_normalization == "zscore_nonzero":
            return zscore_nonzero(image)
        elif Config.intensity_normalization == "percentile":
            return percentile_normalize(image)
        else:
            return image.astype(np.float32)

    def _load_mask_binary(self, path):
        mask = self._load_nii_volume(path)
        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_id = sample["case_id"]
        image_path = sample["image_path"]
        pred_mask_path = sample.get("pred_mask_path", "")
        gt_mask_path = sample.get("gt_mask_path", "")
        label = sample["label"]

        image = self._load_nii_volume(image_path)
        image = self._normalize_image(image)

        expected_shape = Config.input_size_3d
        if tuple(image.shape) != tuple(expected_shape):
            raise ValueError(f"[{case_id}] image shape mismatch: got {image.shape}, expected {expected_shape}")

        if self.input_mode == "image_only":
            volume = np.expand_dims(image, axis=0)
        elif self.input_mode == "image_plus_pred_mask":
            if not pred_mask_path or not os.path.exists(pred_mask_path):
                raise FileNotFoundError(f"[{case_id}] pred mask not found: {pred_mask_path}")
            pred_mask = self._load_mask_binary(pred_mask_path)
            if pred_mask.shape != image.shape:
                raise ValueError(f"[{case_id}] pred mask shape mismatch")
            volume = np.stack([image, pred_mask], axis=0)
        elif self.input_mode == "image_plus_gt_mask":
            if not gt_mask_path or not os.path.exists(gt_mask_path):
                raise FileNotFoundError(f"[{case_id}] gt mask not found: {gt_mask_path}")
            gt_mask = self._load_mask_binary(gt_mask_path)
            if gt_mask.shape != image.shape:
                raise ValueError(f"[{case_id}] gt mask shape mismatch")
            volume = np.stack([image, gt_mask], axis=0)
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        if self.is_train and Config.use_augmentation:
            volume = simple_3d_augmentation(volume)

        image_tensor = torch.tensor(volume, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor, case_id


# =========================================================

def load_samples_from_dir(data_dir, label_dict, input_mode=None):
    
    if input_mode is None:
        input_mode = Config.input_mode
    if not data_dir.exists():
        print(f"Warning: directory not found: {data_dir}")
        return []

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    samples = []
    dropped = []

    for case_dir in case_dirs:
        case_id = case_dir.name   
        if case_id not in label_dict:
            dropped.append((case_id, "no_label"))
            continue

        image_path = case_dir / Config.get_image_filename(case_id)
        pred_mask_path = case_dir / Config.get_pred_mask_filename(case_id)
        gt_mask_path = case_dir / Config.get_gt_mask_filename(case_id)

        if not image_path.exists():
            dropped.append((case_id, "missing_image"))
            continue
        if input_mode == "image_plus_pred_mask" and not pred_mask_path.exists():
            dropped.append((case_id, "missing_pred_mask"))
            continue
        if input_mode == "image_plus_gt_mask" and not gt_mask_path.exists():
            dropped.append((case_id, "missing_gt_mask"))
            continue

        samples.append({
            "case_id": case_id,
            "image_path": str(image_path),
            "pred_mask_path": str(pred_mask_path) if pred_mask_path.exists() else "",
            "gt_mask_path": str(gt_mask_path) if gt_mask_path.exists() else "",
            "label": label_dict[case_id]["target_label"],
            "raw_label": label_dict[case_id]["raw_label"],
            "normalized_label": label_dict[case_id]["normalized_label"],
        })

    print(f"Loaded {len(samples)} valid samples from {data_dir}, dropped {len(dropped)}")
    return samples


def load_samples_from_test_csv(test_label_file, input_mode=None):
   
    if input_mode is None:
        input_mode = Config.input_mode
    test_label_file = Path(test_label_file)
    if not test_label_file.exists():
        raise FileNotFoundError(f"Test label file not found: {test_label_file}")

    df = pd.read_csv(test_label_file, encoding='utf-8-sig')
    label_dict = Config._parse_label_df(df, case_id_col="case_id_folder")
    samples, dropped = [], []

    if 'image_path' not in df.columns:
        raise KeyError(f"CSV missing 'image_path': {test_label_file}")

    for _, row in df.iterrows():
        case_id = str(row['case_id_folder']).strip()
        if case_id not in label_dict:
            dropped.append((case_id, "no_label")); continue
        img_path = row['image_path']
        if pd.isna(img_path) or not Path(img_path).exists():
            dropped.append((case_id, "missing_image")); continue

        pred_mask = ""; gt_mask = ""
        if input_mode in ("image_plus_pred_mask", "image_plus_gt_mask"):
            if 'mask_path' not in df.columns or pd.isna(row['mask_path']):
                dropped.append((case_id, "missing_mask")); continue
            mpath = row['mask_path']
            if not Path(mpath).exists():
                dropped.append((case_id, "mask_not_found")); continue
            if input_mode == "image_plus_pred_mask":
                pred_mask = mpath
            else:
                gt_mask = mpath

        samples.append({
            "case_id": case_id,
            "image_path": str(img_path),
            "pred_mask_path": str(pred_mask) if pred_mask else "",
            "gt_mask_path": str(gt_mask) if gt_mask else "",
            "label": label_dict[case_id]["target_label"],
            "raw_label": label_dict[case_id]["raw_label"],
            "normalized_label": label_dict[case_id]["normalized_label"],
        })

    print(f"Loaded {len(samples)} test samples from {test_label_file.name}, dropped {len(dropped)}")
    return samples



def load_all_trainval_samples(input_mode=None):
    label_dict = Config.load_train_labels()
    return load_samples_from_dir(Config.cases_root, label_dict, input_mode)




def prepare_standard_split(
    train_ratio=None,
    random_state=None,
    input_mode=None
):
    """
    Prepare a development-only train/validation split.

    This utility is not used for the reported five-fold
    cross-validation experiments. Independent test sets
    are not accessed here.
    """
    if train_ratio is None:
        train_ratio = Config.train_val_split_ratio

    if random_state is None:
        random_state = Config.split_random_seed

    if input_mode is None:
        input_mode = Config.input_mode

    trainval = load_all_trainval_samples(input_mode)

    if not trainval:
        raise ValueError(
            "No training/validation samples found."
        )

    existing = Config.load_split_info("standard")

    if existing is not None:
        if "split" in existing and "train_case_ids" not in existing:
            existing = existing["split"]

        if (
            "train_case_ids" not in existing
            or "val_case_ids" not in existing
        ):
            raise KeyError(
                "Loaded standard split does not contain "
                "train_case_ids and val_case_ids."
            )

        sample_map = {
            s["case_id"]: s
            for s in trainval
        }

        missing_train = [
            cid
            for cid in existing["train_case_ids"]
            if cid not in sample_map
        ]

        missing_val = [
            cid
            for cid in existing["val_case_ids"]
            if cid not in sample_map
        ]

        if missing_train or missing_val:
            raise RuntimeError(
                "Saved standard split is not compatible "
                "with the current development dataset."
            )

        train_split = [
            sample_map[cid]
            for cid in existing["train_case_ids"]
        ]

        val_split = [
            sample_map[cid]
            for cid in existing["val_case_ids"]
        ]

        print(
            "Using existing development-only "
            "standard split."
        )

    else:
        labels = [
            s["label"]
            for s in trainval
        ]

        train_split, val_split = train_test_split(
            trainval,
            test_size=1 - train_ratio,
            random_state=random_state,
            stratify=labels,
        )

        Config.save_split_info(
            {
                "train_case_ids": [
                    s["case_id"]
                    for s in train_split
                ],
                "val_case_ids": [
                    s["case_id"]
                    for s in val_split
                ],
            },
            "standard",
        )

    split_dict = {
        "train_cases": train_split,
        "val_cases": val_split,
        "input_mode": input_mode,
    }

    label_dict_for_weight = {
        s["case_id"]: {
            "target_label": s["label"]
        }
        for s in train_split
    }

    Config.class_weights = (
        Config.calculate_class_weights(
            label_dict_for_weight
        )
    )

    print(
        f"\nStandard development split: "
        f"train={len(train_split)}, "
        f"val={len(val_split)}"
    )

    return split_dict



def prepare_kfold_split(
    n_folds=None,
    random_state=None,
    input_mode=None
):
    """
    Prepare stratified K-fold splits using the development
    cohort only.

    Independent test sets are not accessed during
    cross-validation, model development, or fold selection.
    """
    if n_folds is None:
        n_folds = Config.n_folds

    if random_state is None:
        random_state = Config.cv_random_seed

    if input_mode is None:
        input_mode = Config.input_mode

    existing = Config.load_split_info(
        f"{n_folds}fold_cv"
    )

    if existing is not None:
        if "split" in existing and "folds" not in existing:
            existing = existing["split"]

        if (
            "folds" not in existing
            or len(existing["folds"]) != n_folds
        ):
            print(
                "Warning: saved CV split is incomplete. "
                "Regenerating."
            )
            existing = None

    trainval = load_all_trainval_samples(
        input_mode
    )

    if not trainval:
        raise ValueError(
            "No training/validation samples found."
        )

    sample_map = {
        s["case_id"]: s
        for s in trainval
    }

    # -----------------------------------------------------
    # Use an existing development-only CV split
    # -----------------------------------------------------
    if existing is not None:
        print(
            "Using existing development-only "
            "k-fold split."
        )

        folds = []

        for fold_info in existing["folds"]:

            train_ids = fold_info[
                "train_case_ids"
            ]

            val_ids = fold_info[
                "val_case_ids"
            ]

            missing_train = [
                cid
                for cid in train_ids
                if cid not in sample_map
            ]

            missing_val = [
                cid
                for cid in val_ids
                if cid not in sample_map
            ]

            if missing_train or missing_val:
                raise RuntimeError(
                    f"Saved split for fold "
                    f"{fold_info['fold']} is not "
                    f"compatible with the current "
                    f"development dataset."
                )

            fold_train = [
                sample_map[cid]
                for cid in train_ids
            ]

            fold_val = [
                sample_map[cid]
                for cid in val_ids
            ]

            folds.append(
                {
                    "fold": fold_info["fold"],
                    "train_cases": fold_train,
                    "val_cases": fold_val,
                    "input_mode": input_mode,
                }
            )

            print(
                f"Fold {fold_info['fold']}: "
                f"train={len(fold_train)}, "
                f"val={len(fold_val)}"
            )

        return folds

    # -----------------------------------------------------
    # Generate a new development-only CV split
    # -----------------------------------------------------
    labels = [
        s["label"]
        for s in trainval
    ]

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    fold_records = []

    for fold_idx, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(
            np.arange(len(trainval)),
            labels
        )
    ):

        fold_train = [
            trainval[i]
            for i in train_idx
        ]

        fold_val = [
            trainval[i]
            for i in val_idx
        ]

        folds.append(
            {
                "fold": fold_idx,
                "train_cases": fold_train,
                "val_cases": fold_val,
                "input_mode": input_mode,
            }
        )

        train_dist = (
            pd.Series(
                [x["label"] for x in fold_train]
            )
            .value_counts()
            .sort_index()
            .to_dict()
        )

        val_dist = (
            pd.Series(
                [x["label"] for x in fold_val]
            )
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"Fold {fold_idx}: "
            f"train={len(fold_train)} "
            f"({train_dist}), "
            f"val={len(fold_val)} "
            f"({val_dist})"
        )

        fold_records.append(
            {
                "fold": fold_idx,
                "train_case_ids": [
                    s["case_id"]
                    for s in fold_train
                ],
                "val_case_ids": [
                    s["case_id"]
                    for s in fold_val
                ],
            }
        )

    label_dict_for_weight = {
        s["case_id"]: {
            "target_label": s["label"]
        }
        for s in trainval
    }

    Config.class_weights = (
        Config.calculate_class_weights(
            label_dict_for_weight
        )
    )

    cv_split_info = {
        "n_folds": n_folds,
        "random_state": random_state,
        "input_mode": input_mode,
        "total_trainval": len(trainval),
        "class_distribution": {
            "class0": int(
                np.sum(
                    np.array(labels) == 0
                )
            ),
            "class1": int(
                np.sum(
                    np.array(labels) == 1
                )
            ),
        },
        "folds": fold_records,
        "timestamp": datetime.now().isoformat(),
    }

    split_file = (
        Config.splits_dir
        / (
            f"{Config.get_experiment_name()}_"
            f"{n_folds}fold_cv_split.json"
        )
    )

    with open(
        split_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            cv_split_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Development CV split saved to: "
        f"{split_file}"
    )

    return folds

# =========================================================

# =========================================================
def get_data_loader_from_samples(samples, batch_size, is_train=True):
    dataset = RectalMRIClassification3DDataset(samples=samples, is_train=is_train)
    if is_train:
        labels = [s['label'] for s in samples]
        class_counts = np.bincount(labels)          
        weights = 1.0 / np.sqrt(class_counts)[labels]   
        sampler = WeightedRandomSampler(weights, len(weights), replacement=False)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=Config.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    return loader


def get_standard_data_loaders(
    split_dict=None
):
    if split_dict is None:
        split_dict = prepare_standard_split(
            input_mode=Config.input_mode
    )

    batch_size = Config.batch_size

    model_cfg = Config.get_model_config(
        Config.model_name
    )

    if "batch_size" in model_cfg:
        batch_size = model_cfg["batch_size"]

    train_loader = (
        get_data_loader_from_samples(
            split_dict["train_cases"],
            batch_size,
            is_train=True,
        )
    )

    val_loader = (
        get_data_loader_from_samples(
            split_dict["val_cases"],
            batch_size,
            is_train=False,
        )
    )

    return train_loader, val_loader

def get_kfold_data_loaders(folds=None, fold_index=0):
    if folds is None:
        folds = prepare_kfold_split(input_mode=Config.input_mode)
    fold_data = folds[fold_index]
    batch_size = Config.batch_size
    model_cfg = Config.get_model_config(Config.model_name)
    if "batch_size" in model_cfg:
        batch_size = model_cfg["batch_size"]
    train_loader = get_data_loader_from_samples(
    fold_data["train_cases"],
    batch_size,
    is_train=True
    )

    val_loader = get_data_loader_from_samples(
    fold_data["val_cases"],
    batch_size,
    is_train=False
    )

    return train_loader, val_loader, fold_data

# =========================================================

# =========================================================
def validate_data_paths(
    input_mode=None,
    include_test=False
):
    if input_mode is None:
        input_mode = Config.input_mode

    print(
        f"\nValidating data paths "
        f"(input_mode={input_mode}) ..."
    )

    train_samples = (
        load_all_trainval_samples(
            input_mode
        )
    )

    print(
        f"Training/validation cases: "
        f"{len(train_samples)}"
    )

    train_dist = (
        pd.Series(
            [s["label"] for s in train_samples]
        )
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "Train+Val class distribution: "
        f"{train_dist}"
    )



def debug_data_statistics(input_mode=None, max_cases=20):
    if input_mode is not None:
        Config.input_mode = input_mode
    print("\n3D Data Statistics (on train/val subset)")
    samples = load_all_trainval_samples(Config.input_mode)[:max_cases]
    dataset = RectalMRIClassification3DDataset(samples, is_train=False)
    all_voxels = []
    for i in range(min(10, len(dataset))):
        img, label, _ = dataset[i]
        img_np = img.cpu().numpy()
        all_voxels.extend(img_np.flatten())
        print(f"Sample {i}: shape={img.shape}, range=[{img_np.min():.4f}, {img_np.max():.4f}]")
    all_voxels = np.array(all_voxels)
    print(f"\nOverall: mean={all_voxels.mean():.4f}, std={all_voxels.std():.4f}")
    plt.figure(); plt.hist(all_voxels, bins=100); plt.savefig("voxel_dist.png"); plt.close()
    print("Histogram saved to voxel_dist.png")


def inspect_one_batch(input_mode=None):
    if input_mode is not None:
        Config.input_mode = input_mode
    split_dict = prepare_standard_split()
    train_loader, _ = get_standard_data_loaders(
    split_dict
)
    images, labels, case_ids = next(iter(train_loader))
    print(f"\nBatch: images {images.shape}, labels {labels[:5]}, cases {list(case_ids[:5])}")
    return images, labels, case_ids