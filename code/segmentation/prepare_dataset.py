import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from config import Config


def list_cases(source_cases_dir: Path):
    cases = []
    if not source_cases_dir.exists():
        raise RuntimeError(f"source_cases_dir does not exist: {source_cases_dir}")

    for case_dir in sorted(source_cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name
        image_path = case_dir / f"{case_id}_image.nii.gz"
        mask_path = case_dir / f"{case_id}_mask.nii.gz"

        if image_path.exists() and mask_path.exists():
            cases.append({
                "case_id": case_id,
                "image_path": image_path,
                "mask_path": mask_path,
            })

    return cases


def copy_image(src: Path, dst: Path):
    shutil.copy2(src, dst)


def copy_label(src: Path, dst: Path):
    if not Config.force_binary_labels:
        shutil.copy2(src, dst)
        return

    img = sitk.ReadImage(str(src))
    arr = sitk.GetArrayFromImage(img)
    arr = (arr > 0).astype(np.uint8)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst))


def copy_case_to_train(case, train_case_name: str):
    dst_img = Config.imagesTr_dir() / f"{train_case_name}_0000.nii.gz"
    dst_lbl = Config.labelsTr_dir() / f"{train_case_name}.nii.gz"

    copy_image(case["image_path"], dst_img)
    copy_label(case["mask_path"], dst_lbl)

    return {
        "source_case_id": case["case_id"],
        "nnunet_case_id": train_case_name,
        "imageTr": str(dst_img),
        "labelTr": str(dst_lbl),
        "split": "train_all",
    }


def write_dataset_json(num_training_cases: int):
    dataset_json = {
        "channel_names": {
            "0": Config.modality_name
        },
        "labels": Config.labels,
        "numTraining": num_training_cases,
        "file_ending": Config.file_ending,
        "name": Config.dataset_name,
        "description": "Rectal Cancer MRI tumor segmentation, merged two-center trainval cases",
        "reference": "",
        "licence": "",
        "release": "1.0"
    }

    with open(Config.dataset_json_path(), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)


def prepare_dataset():
    if Config.overwrite:
        if Config.dataset_raw_dir().exists():
            shutil.rmtree(Config.dataset_raw_dir())

    Config.ensure_base_dirs()

    raw_dir = Config.dataset_raw_dir()
    imagesTr = Config.imagesTr_dir()
    labelsTr = Config.labelsTr_dir()
    imagesTs = Config.imagesTs_dir()

    raw_dir.mkdir(parents=True, exist_ok=True)
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs.mkdir(parents=True, exist_ok=True)

    all_cases = list_cases(Config.source_cases_dir)
    if len(all_cases) == 0:
        raise RuntimeError(
            f"No valid cases found under: {Config.source_cases_dir}"
        )
    EXPECTED_DEV_CASES = 617
    if len(all_cases) != EXPECTED_DEV_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_DEV_CASES} development cases, "
            f"found {len(all_cases)} in {Config.source_cases_dir}"
        )

    mapping_rows = []

    for idx, case in enumerate(all_cases):
        nnunet_case_id = f"RCM_{idx:04d}"
        mapping_rows.append(copy_case_to_train(case, nnunet_case_id))

    write_dataset_json(num_training_cases=len(all_cases))

    with open(Config.case_mapping_path(), "w", encoding="utf-8") as f:
        json.dump(mapping_rows, f, indent=2, ensure_ascii=False)

    split_summary = {
        "source_cases_dir": str(Config.source_cases_dir),
        "dataset_raw_dir": str(Config.dataset_raw_dir()),
        "num_total_cases": len(all_cases),
        "num_train_cases_for_nnunet": len(all_cases),
        "num_test_cases": 0,
        "split_strategy": "all_cases_used_as_imagesTr_labelsTr__no_explicit_heldout_test",
        "dataset_id": Config.dataset_id,
        "dataset_name": Config.dataset_name,
    }

    with open(Config.split_summary_path(), "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ nnU-Net dataset prepared (ALL TRAINING CASES)")
    print(f"Source cases       : {Config.source_cases_dir}")
    print(f"Dataset raw dir    : {Config.dataset_raw_dir()}")
    print(f"Training cases     : {len(all_cases)}")
    print(f"dataset.json       : {Config.dataset_json_path()}")
    print(f"case_mapping       : {Config.case_mapping_path()}")
    print(f"split_summary      : {Config.split_summary_path()}")
    print("=" * 70)


if __name__ == "__main__":
    prepare_dataset()