from pathlib import Path


class Config:
    # =========================================================
    # 1. Source data
    # =========================================================
  
    source_cases_dir = Path(
        r"RECTAL_STAS_DATA\RectSeg_two_centers_trainval_merged\cases"
    )

    # =========================================================
    # 2. Official nnU-Net dataset info
    # =========================================================
    dataset_id = 304
    dataset_name = "RectalCancerMRI_TwoCenters_TrainvalMerged_XY400Z24"
    modality_name = "T2WI"
    file_ending = ".nii.gz"

    image_suffix = "_image.nii.gz"
    label_suffix = "_mask.nii.gz"

    labels = {
        "background": 0,
        "tumor": 1,
    }

    force_binary_labels = True
    overwrite = True

    # =========================================================
    # 3. nnU-Net working directories
    # =========================================================
    nnunet_raw = Path(r"RECTAL_STAS_DATA\nnUNet_raw")
    nnunet_preprocessed = Path(r"RECTAL_STAS_DATA\nnUNet_preprocessed")
    nnunet_results = Path(r"RECTAL_STAS_DATA\nnUNet_results")


    # =========================================================
    # 4. Official nnU-Net training config
    # =========================================================
    configuration = "3d_fullres"
    trainer = "nnUNetTrainer"
    plans = "nnUNetPlans"
    gpu_id = 0
    folds = (0, 1, 2, 3, 4)

    
    max_epochs = 1000
    initial_lr = 1e-2

    use_early_stopping = False
    early_stopping_patience = None
    early_stopping_min_delta = None

    # =========================================================
    # 6. Helpers
    # =========================================================
    @classmethod
    def dataset_folder_name(cls):
        return f"Dataset{cls.dataset_id:03d}_{cls.dataset_name}"

    @classmethod
    def dataset_raw_dir(cls):
        return cls.nnunet_raw / cls.dataset_folder_name()

    @classmethod
    def imagesTr_dir(cls):
        return cls.dataset_raw_dir() / "imagesTr"

    @classmethod
    def labelsTr_dir(cls):
        return cls.dataset_raw_dir() / "labelsTr"

    @classmethod
    def imagesTs_dir(cls):
        return cls.dataset_raw_dir() / "imagesTs"

    @classmethod
    def dataset_json_path(cls):
        return cls.dataset_raw_dir() / "dataset.json"

    @classmethod
    def case_mapping_path(cls):
        return cls.dataset_raw_dir() / "case_mapping.json"

    @classmethod
    def split_summary_path(cls):
        return cls.dataset_raw_dir() / "split_summary.json"

    @classmethod
    def ensure_base_dirs(cls):
        cls.nnunet_raw.mkdir(parents=True, exist_ok=True)
        cls.nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
        cls.nnunet_results.mkdir(parents=True, exist_ok=True)

        cls.dataset_raw_dir().mkdir(parents=True, exist_ok=True)
        cls.imagesTr_dir().mkdir(parents=True, exist_ok=True)
        cls.labelsTr_dir().mkdir(parents=True, exist_ok=True)
        cls.imagesTs_dir().mkdir(parents=True, exist_ok=True)
