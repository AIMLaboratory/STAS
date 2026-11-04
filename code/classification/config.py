import os
import json
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd


class Config:
    # =========================================================
    # 1. Task
    # =========================================================
    task_name = "T_stage_classification_3D"
    input_mode = "image_only"
    label_mode = "binary"

    # =========================================================
    # 2. Data paths
    # =========================================================
    cases_root = Path(
        r"RECTAL_STAS_DATA\RectSeg_two_centers_trainval_merged\cases"
    )
    label_file = Path(
        r"RECTAL_STAS_DATA\RectSeg_two_centers_trainval_merged\merged_tstage_labels.csv"
    )
  


    radiologist_file = Path(r"RECTAL_STAS_DATA\radiologist_t_stage.xlsx")

    # =========================================================
    # 3. Split configuration
    # =========================================================
    use_cross_validation = True
    n_folds = 5
    cv_random_seed = 42
    train_val_split_ratio = 0.8
    split_random_seed = 42

    # =========================================================
    # 4. Input settings
    # =========================================================
    model_dimension = "3d"
    input_size_3d = (24, 400, 400)
    intensity_normalization = "zscore_nonzero"

    # =========================================================
    # 5. Training hyperparameters
    # =========================================================
    batch_size = 2
    num_epochs = 200
    learning_rate = 5e-5 
    weight_decay = 1e-3
    num_workers = 2
    use_amp = False
    use_augmentation = True
    early_stopping_patience = 50
    early_stopping_min_delta = 1e-4
    
    monitor_metric = "auc"

    # =========================================================
    # 6. Device
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # 7. Model
    # =========================================================
    model_name = "densenet121_3d"

    @staticmethod
    def get_model_config(model_name):
        model_configs = {
            "resnet18_3d": {"dimension": "3d", "batch_size": 1, "learning_rate": 1e-4,
                            "dropout": 0.2, "pretrained": False},
            "resnet34_3d": {"dimension": "3d", "batch_size": 2, "learning_rate": 5e-5,
                            "dropout": 0.2,"pretrained": False},
            "densenet121_3d": {"dimension": "3d", "batch_size": 2, "learning_rate": 5e-5,
                               "dropout": 0.2, "pretrained": False},
            "med3d": {"dimension": "3d", "batch_size": 2, "learning_rate": 1e-4,
                      "dropout": 0.2, "pretrained": False},
        }
        return model_configs.get(model_name, {})

    # =========================================================
    # 8. Loss
    # =========================================================
    loss_type = "focal"
    focal_gamma = 2.0
    focal_alpha = [0.6, 0.4]
    class_weights = None

    # =========================================================
    # 9. File naming
    # =========================================================
    image_suffix = "_image.nii.gz"
    pred_mask_suffix = "_pred.nii.gz"
    gt_mask_suffix = "_mask.nii.gz"

    @staticmethod
    def get_image_filename(case_id: str):
        return f"{case_id}{Config.image_suffix}"

    @staticmethod
    def get_pred_mask_filename(case_id: str):
        return f"{case_id}{Config.pred_mask_suffix}"

    @staticmethod
    def get_gt_mask_filename(case_id: str):
        return f"{case_id}{Config.gt_mask_suffix}"

    # =========================================================
    # 10. Input channels  (critical fix)
    # =========================================================
    @staticmethod
    def get_num_input_channels():
        """
        Channel count per input configuration:

        - "image_only":            T2WI only                -> 1 channel
        - "image_plus_pred_mask":  T2WI + predicted mask    -> 2 channels
        - "image_plus_gt_mask":    T2WI + ground-truth mask -> 2 channels
        """
        if Config.input_mode == "image_only":
            return 1
        elif Config.input_mode in ["image_plus_pred_mask", "image_plus_gt_mask"]:
            return 2
        else:
            raise ValueError(f"Unknown input_mode: {Config.input_mode}")

    # =========================================================
    # 11. T-stage cleaning & label mapping
    # =========================================================
    invalid_labels = {"", "none", "nan", "na", "unknown"}

    @staticmethod
    def normalize_t_stage(raw_label):
        if pd.isna(raw_label):
            return None
        s = str(raw_label).strip()
        s = s.replace(" ", "")
        # Normalize full-width CJK punctuation to ASCII equivalents,
        # because pathology reports may contain mixed-width characters.
        s = s.replace("\uff08", "(").replace("\uff09", ")")  # full-width parentheses
        s = s.replace("\uff0c", ",")                         # full-width comma
        s = s.replace("\uff1b", ";")                         # full-width semicolon
        if s.lower() in Config.invalid_labels:
            return None
        # Long-form labels that mention "rectal" or "sigmoid" together with
        # T4 are all mapped to T4.
        if "T4" in s and ("\u76f4\u80a0" in s or "\u4e59\u72b6\u7ed3\u80a0" in s):
            return "T4"
        patterns = ["T4a", "T4b", "T1", "T2", "T3", "T4"]
        for p in patterns:
            if p.lower() in s.lower():
                return p
        return None

    @staticmethod
    def map_t_stage_to_target(normalized_label):
        if normalized_label is None:
            return None
        if Config.label_mode == "binary":
            early = {"T1", "T2"}
            advanced = {"T3", "T4", "T4a", "T4b"}
            if normalized_label in early:
                return 0
            elif normalized_label in advanced:
                return 1
            else:
                return None
        elif Config.label_mode == "multiclass4":
            mapping = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T4a": 3, "T4b": 3}
            return mapping.get(normalized_label, None)
        else:
            raise ValueError(f"Unknown label_mode: {Config.label_mode}")

    @staticmethod
    def get_class_names():
        if Config.label_mode == "binary":
            return ["T1/T2", "T3/T4"]
        elif Config.label_mode == "multiclass4":
            return ["T1", "T2", "T3", "T4"]
        else:
            raise ValueError(f"Unknown label_mode: {Config.label_mode}")

    @staticmethod
    def get_num_classes():
        return len(Config.get_class_names())

    # =========================================================

    # =========================================================
    @staticmethod
    def _parse_label_df(df, case_id_col="case_id"):
        """
        """
      
        cleaned_map = {col.strip().lower(): col for col in df.columns}

      
        if case_id_col not in df.columns:
            candidate_keys = [
                "case_id", "caseid", "folder_name", "foldername",
                "case_id_folder", "case_folder", "patient_id",
                "id",
            ]
            found = False
            for key in candidate_keys:
                if key in cleaned_map:
                    case_id_col = cleaned_map[key]
                    found = True
                    print(f"  (Auto-detected case-ID column: {case_id_col})")
                    break
            if not found:
               
                for clean_name, orig_name in cleaned_map.items():
                    if "case" in clean_name or "id" in clean_name:
                        case_id_col = orig_name
                        found = True
                        print(f"  (Fuzzy-matched case-ID column: {case_id_col})")
                        break
            if not found:
                raise KeyError(
                    f"Unable to find a case-ID column in the label table.\n"
                    f"Available columns: {list(df.columns)}"
                )

     
        if "binary_label" in df.columns:
            label_col = "binary_label"
            direct_numeric = True
        elif "label" in df.columns:
            label_col = "label"
            direct_numeric = True
        elif "tstage_label" in df.columns:
            label_col = "tstage_label"
            direct_numeric = False
        elif "t_stage_raw" in df.columns:
            label_col = "t_stage_raw"
            direct_numeric = False
        else:
            raise KeyError(
                f"Unable to find a label column. "
                f"Available columns: {list(df.columns)}"
            )

        
        label_dict = {}
        for _, row in df.iterrows():
            case_id = row[case_id_col]
            if pd.isna(case_id):
                continue
            case_id = str(case_id).strip()
            if case_id == "":
                continue

            raw_value = row[label_col]
            if direct_numeric:
                try:
                    target = int(float(raw_value))
                except (ValueError, TypeError):
                    continue
                normalized = str(target)
            else:
                normalized = Config.normalize_t_stage(raw_value)
                target = Config.map_t_stage_to_target(normalized)
                if target is None:
                    continue

            label_dict[case_id] = {
                "raw_label": str(raw_value),
                "normalized_label": normalized,
                "target_label": int(target),
            }
        return label_dict

    @staticmethod
    def load_train_labels():
        label_file = Config.label_file
        if not label_file.exists():
            raise FileNotFoundError(f"Train label file not found: {label_file}")
        if label_file.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(label_file)
        else:
            df = pd.read_csv(label_file, encoding="utf-8-sig")
        label_dict = Config._parse_label_df(df, case_id_col="case_id")
        print(f"Loaded {len(label_dict)} training labels from {label_file}")
        return label_dict

    @staticmethod
    def load_test_labels():
        all_labels = {}
        for test_label_file in Config.test_label_files:
            if not test_label_file.exists():
                raise FileNotFoundError(f"Test label file not found: {test_label_file}")
            try:
                if test_label_file.suffix.lower() in [".xlsx", ".xls"]:
                    df = pd.read_excel(test_label_file)
                else:
                    df = pd.read_csv(test_label_file, encoding="utf-8-sig")
                print(f"  Reading {test_label_file.name}, columns: {list(df.columns)}")
                labels = Config._parse_label_df(df, case_id_col="case_id")
                overlap = set(labels.keys()) & set(all_labels.keys())
                if overlap:
                    print(f"Warning: duplicate case IDs: {overlap}")
                all_labels.update(labels)
                print(f"Loaded {len(labels)} test labels from {test_label_file.name}")
            except Exception as e:
                print(f"Error loading {test_label_file}: {e}")
                raise
        return all_labels

    # =========================================================
    # 13. Class weights
    # =========================================================
    @staticmethod
    def calculate_class_weights(label_dict):
        if not label_dict:
            return None
        num_classes = Config.get_num_classes()
        class_counts = {i: 0 for i in range(num_classes)}
        for item in label_dict.values():
            y = item["target_label"]
            if y in class_counts:
                class_counts[y] += 1
        total_samples = sum(class_counts.values())
        print(f"Class distribution: {class_counts}")
        if any(v == 0 for v in class_counts.values()):
            print("Warning: At least one class has zero samples")
            weight_tensor = torch.ones(num_classes, dtype=torch.float32)
        else:
            weights = {}
            for cls, count in class_counts.items():
                weights[cls] = total_samples / (num_classes * count)
            print(f"Class weights: {weights}")
            weight_tensor = torch.tensor(
                [weights[i] for i in range(num_classes)],
                dtype=torch.float32
            )
            weight_tensor = weight_tensor / torch.sum(weight_tensor)
        return weight_tensor

    # =========================================================
    
    # =========================================================
    experiments_dir = Path("./experiments")
    splits_dir = Path("./splits")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_experiment_name():
        return f"{Config.task_name}_{Config.input_mode}_{Config.label_mode}_{Config.model_name}"

    @staticmethod
    def save_split_info(split_dict, split_type="cv"):
        split_info = {
            "task_name": Config.task_name,
            "input_mode": Config.input_mode,
            "label_mode": Config.label_mode,
            "split_type": split_type,
            "split": split_dict,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_folds": Config.n_folds if split_type == "cv" else None,
                "random_state": Config.cv_random_seed,
                "train_samples": len(split_dict.get("train_case_ids", [])),
                "val_samples": len(split_dict.get("val_case_ids", [])),
                "test_samples": len(split_dict.get("test_case_ids", [])),
            }
        }
        filename = f"{Config.get_experiment_name()}_{split_type}_split.json"
        out_path = Config.splits_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"Split information saved to: {out_path}")

    @staticmethod
    def load_split_info(split_type="cv"):
        filename = f"{Config.get_experiment_name()}_{split_type}_split.json"
        split_file = Config.splits_dir / filename
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            print(f"Loaded split information (created: {split_info['timestamp']})")
            return split_info.get("split", split_info)
        return None