import argparse
import os
import shutil
import subprocess
import sys
import time

from config import Config
from prepare_dataset import prepare_dataset


def set_nnunet_env():
    Config.ensure_base_dirs()

    os.environ["nnUNet_raw"] = str(Config.nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(Config.nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(Config.nnunet_results)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.gpu_id)

    os.environ["nnUNet_n_proc_DA"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def print_env():
    print("\n" + "=" * 70)
    print("nnU-Net Environment")
    print("=" * 70)
    print(f"nnUNet_raw           : {os.environ.get('nnUNet_raw')}")
    print(f"nnUNet_preprocessed  : {os.environ.get('nnUNet_preprocessed')}")
    print(f"nnUNet_results       : {os.environ.get('nnUNet_results')}")
    print(f"CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"nnUNet_n_proc_DA     : {os.environ.get('nnUNet_n_proc_DA')}")
    print(f"OMP_NUM_THREADS      : {os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS      : {os.environ.get('MKL_NUM_THREADS')}")
    print("=" * 70)


def print_config():
    print("\n" + "=" * 70)
    print("Official nnU-Net v2 Rectal Cancer Segmentation Training")
    print("=" * 70)
    print(f"Source cases dir     : {Config.source_cases_dir}")
    print(f"Dataset ID           : {Config.dataset_id}")
    print(f"Dataset Name         : {Config.dataset_name}")
    print(f"Modality             : {Config.modality_name}")
    print(f"Configuration        : {Config.configuration}")
    print(f"Trainer              : {Config.trainer}")
    print(f"Plans                : {Config.plans}")
    print(f"GPU ID               : {Config.gpu_id}")
    print(f"Folds                : {Config.folds}")
    print(f"Max epochs(record)   : {Config.max_epochs}")
    print(f"Initial LR(record)   : {Config.initial_lr}")
    print(f"nnUNet_raw           : {Config.nnunet_raw}")
    print(f"nnUNet_preprocessed  : {Config.nnunet_preprocessed}")
    print(f"nnUNet_results       : {Config.nnunet_results}")
    print("=" * 70)


def check_command_exists(command_name: str):
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"Command not found: {command_name}\n"
            f"Please confirm nnunetv2 is installed in the current environment."
        )


def run_cmd(cmd):
    print("\nRunning command:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


def prepare():
    print("\nStep 1/3: Preparing nnU-Net dataset...")
    prepare_dataset()


def preprocess():
    print("\nStep 2/3: Planning and preprocessing...")
    check_command_exists("nnUNetv2_plan_and_preprocess")

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(Config.dataset_id),
        "--verify_dataset_integrity",
    ]
    run_cmd(cmd)


def train_fold(fold: int, continue_training: bool = False):
    print(f"\nStep 3/3: Training fold {fold} ...")

    if fold not in Config.folds:
        raise ValueError(f"fold {fold} not in Config.folds={Config.folds}")

    check_command_exists("nnUNetv2_train")

    cmd = [
        "nnUNetv2_train",
        str(Config.dataset_id),
        Config.configuration,
        str(fold),
        "-tr", Config.trainer,
        "-p", Config.plans,
    ]

    if continue_training:
        cmd.append("--c")

    run_cmd(cmd)


def train_all():
    for fold in Config.folds:
        train_fold(fold, continue_training=False)


def show_paths():
    print("\n" + "=" * 70)
    print("Important Paths")
    print("=" * 70)
    print(f"Source cases dir      : {Config.source_cases_dir}")
    print(f"Dataset raw dir       : {Config.dataset_raw_dir()}")
    print(f"imagesTr              : {Config.imagesTr_dir()}")
    print(f"labelsTr              : {Config.labelsTr_dir()}")
    print(f"imagesTs              : {Config.imagesTs_dir()}")
    print(f"dataset.json          : {Config.dataset_json_path()}")
    print(f"case_mapping          : {Config.case_mapping_path()}")
    print(f"split_summary         : {Config.split_summary_path()}")
    print("=" * 70)


def full_fold0():
    prepare()
    preprocess()
    train_fold(0, continue_training=False)


def full_all():
    prepare()
    preprocess()
    train_all()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Official nnU-Net v2 for Rectal Cancer Segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show_config", help="Show current config")
    subparsers.add_parser("show_env", help="Show environment variables")
    subparsers.add_parser("show_paths", help="Show important paths")

    subparsers.add_parser("prepare", help="Convert merged cases into official nnU-Net dataset")
    subparsers.add_parser("preprocess", help="Run official planning + preprocessing")
    subparsers.add_parser("train_all", help="Train all folds")
    subparsers.add_parser("full_fold0", help="prepare + preprocess + train fold 0")
    subparsers.add_parser("full_all", help="prepare + preprocess + train all folds")

    p_train_fold = subparsers.add_parser("train_fold", help="Train one fold")
    p_train_fold.add_argument("--fold", type=int, required=True, help="For example 0")
    p_train_fold.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from latest checkpoint"
    )

    return parser


def main():
    set_nnunet_env()
    parser = build_parser()
    args = parser.parse_args()

    start = time.time()

    try:
        if args.command == "show_config":
            print_config()

        elif args.command == "show_env":
            print_env()

        elif args.command == "show_paths":
            show_paths()

        elif args.command == "prepare":
            print_config()
            prepare()

        elif args.command == "preprocess":
            print_config()
            preprocess()

        elif args.command == "train_fold":
            print_config()
            train_fold(args.fold, continue_training=args.continue_training)

        elif args.command == "train_all":
            print_config()
            train_all()

        elif args.command == "full_fold0":
            print_config()
            full_fold0()

        elif args.command == "full_all":
            print_config()
            full_all()

        else:
            raise ValueError(f"Unknown command: {args.command}")

    except KeyboardInterrupt:
        print("\nUser interrupted")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nExternal command failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nRun failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()