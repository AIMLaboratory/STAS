from pathlib import Path
import json
import math
import traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk


# =========================================================
# 1. Paths and parameters
# =========================================================
# Anonymous input and output roots. Set these to your local
# raw-case folder and target standardized-case folder.
INPUT_ROOT = Path(r"RECTAL_STAS_DATA\raw_cases")
OUTPUT_ROOT = Path(r"RECTAL_STAS_DATA\RectSeg_two_centers_trainval_merged")
CASES_DIR = OUTPUT_ROOT / "cases"

# Crop or pad XY to 400 x 400
TARGET_X = 400
TARGET_Y = 400

# Keep the central 24 slices along Z
TARGET_Z = 24

SKIP_EXISTING = True
START_CASE_ID = None


# =========================================================
# 2. Main class
# =========================================================
class PreprocessXYCrop400MiddleZ24Center:
    def __init__(self):
        self.input_root = INPUT_ROOT
        self.output_root = OUTPUT_ROOT
        self.cases_dir = CASES_DIR

        self.target_x = int(TARGET_X)
        self.target_y = int(TARGET_Y)
        self.target_z = int(TARGET_Z)

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cases_dir.mkdir(parents=True, exist_ok=True)

        self.success = []
        self.failed = []

        self.success_csv = self.output_root / "preprocess_success.csv"
        self.failed_csv = self.output_root / "preprocess_failed.csv"
        self.summary_json = self.output_root / "preprocess_summary.json"

    # =========================================================
    # 3. Locate image and mask
    # =========================================================
    def find_image_mask(self, case_dir: Path):
        image_files = sorted(case_dir.glob("*_image.nii.gz"))
        mask_files = sorted(case_dir.glob("*_mask.nii.gz"))

        if len(image_files) != 1 or len(mask_files) != 1:
            raise FileNotFoundError(
                "Expected exactly one image and one mask per case folder; "
                f"found image={len(image_files)}, mask={len(mask_files)}"
            )
        return image_files[0], mask_files[0]

    # =========================================================
    # 4. Geometry consistency check
    # =========================================================
    def same_geometry(self, img1, img2):
        return (
            img1.GetSize() == img2.GetSize()
            and np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=1e-6)
            and np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=1e-6)
            and np.allclose(img1.GetDirection(), img2.GetDirection(), atol=1e-6)
        )

    def print_geometry(self, name, img):
        print(f"    {name}")
        print(f"      Size      : {img.GetSize()}")
        print(f"      Spacing   : {img.GetSpacing()}")
        print(f"      Origin    : {img.GetOrigin()}")
        print(f"      Direction : {img.GetDirection()}")

    # =========================================================
    # 5. Binarize mask
    # =========================================================
    def ensure_binary_mask(self, image):
        arr = sitk.GetArrayFromImage(image)
        arr = (arr > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(image)
        return out

    # =========================================================
    # 6. Resample mask to image grid
    #    (called only when there is a true geometric mismatch)
    # =========================================================
    def resample_mask_to_image(self, mask, image):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt8)

        out = resampler.Execute(mask)
        return self.ensure_binary_mask(out)

    # =========================================================
    # 7. Centered start index for a target size
    # =========================================================
    def compute_centered_start(self, size_dim, target_dim):
        geometric_center = (size_dim - 1) / 2.0
        target_center = (target_dim - 1) / 2.0
        start = geometric_center - target_center
        return int(math.floor(start + 1e-6))

    # =========================================================
    # 8. Shared XY crop / pad parameters
    # =========================================================
    def get_xy_crop_pad_params(self, image, target_x, target_y):
        size = list(image.GetSize())  # (x, y, z)

        start_x = self.compute_centered_start(size[0], target_x)
        start_y = self.compute_centered_start(size[1], target_y)
        end_x = start_x + target_x
        end_y = start_y + target_y

        pad_lower_x = max(0, -start_x)
        pad_lower_y = max(0, -start_y)
        pad_upper_x = max(0, end_x - size[0])
        pad_upper_y = max(0, end_y - size[1])

        return {
            "start_x": int(start_x),
            "start_y": int(start_y),
            "end_x": int(end_x),
            "end_y": int(end_y),
            "pad_lower_x": int(pad_lower_x),
            "pad_lower_y": int(pad_lower_y),
            "pad_upper_x": int(pad_upper_x),
            "pad_upper_y": int(pad_upper_y),
        }

    # =========================================================
    # 9. Apply shared XY crop / pad parameters
    # =========================================================
    def crop_or_pad_xy_with_params(self, image, params, target_x, target_y, is_mask=False):
        if (
            params["pad_lower_x"] > 0 or params["pad_lower_y"] > 0
            or params["pad_upper_x"] > 0 or params["pad_upper_y"] > 0
        ):
            image = sitk.ConstantPad(
                image,
                padLowerBound=[params["pad_lower_x"], params["pad_lower_y"], 0],
                padUpperBound=[params["pad_upper_x"], params["pad_upper_y"], 0],
                constant=0
            )

        roi_index = [
            params["start_x"] + params["pad_lower_x"],
            params["start_y"] + params["pad_lower_y"],
            0
        ]
        roi_size = [
            int(target_x),
            int(target_y),
            int(image.GetSize()[2])
        ]

        out = sitk.RegionOfInterest(image, size=roi_size, index=roi_index)

        if is_mask:
            return self.ensure_binary_mask(out)
        return out

    # =========================================================
    # 10. Fix Z to 24 slices: keep middle; center pad zeros if fewer
    # =========================================================
    def fix_z_to_24_keep_middle(self, image, is_mask=False):
        original_size = list(image.GetSize())  # (x, y, z)
        current_z = int(original_size[2])

        start_z = self.compute_centered_start(current_z, self.target_z)
        end_z = start_z + self.target_z

        pad_lower_z = max(0, -start_z)
        pad_upper_z = max(0, end_z - current_z)

        if pad_lower_z > 0 or pad_upper_z > 0:
            image = sitk.ConstantPad(
                image,
                padLowerBound=[0, 0, pad_lower_z],
                padUpperBound=[0, 0, pad_upper_z],
                constant=0
            )

        roi_index = [0, 0, start_z + pad_lower_z]
        roi_size = [image.GetSize()[0], image.GetSize()[1], self.target_z]

        out = sitk.RegionOfInterest(image, size=roi_size, index=roi_index)

        if is_mask:
            return self.ensure_binary_mask(out)
        return out

    # =========================================================
    # 11. Describe the Z action
    # =========================================================
    def get_z_action(self, original_z):
        if original_z < self.target_z:
            return f"pad_zero_by_center_to_{self.target_z}"
        elif original_z > self.target_z:
            return f"keep_middle_{self.target_z}"
        else:
            return "keep_unchanged"

    # =========================================================
    # 12. Check whether an existing output pair is valid
    # =========================================================
    def existing_output_is_valid(self, out_image, out_mask):
        try:
            if (not out_image.exists()) or (not out_mask.exists()):
                return False

            image = sitk.ReadImage(str(out_image))
            mask = sitk.ReadImage(str(out_mask))

            if image.GetSize() != (self.target_x, self.target_y, self.target_z):
                return False
            if mask.GetSize() != (self.target_x, self.target_y, self.target_z):
                return False
            if not self.same_geometry(image, mask):
                return False

            return True
        except Exception:
            return False

    # =========================================================
    # 13. Process one case
    # =========================================================
    def process_one_case(self, case_dir: Path, case_index: int = None):
        case_id = case_dir.name
        label = f"case #{case_index}" if case_index is not None else "case"
        print(f"\n[preprocess] {label}")

        out_case_dir = self.cases_dir / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        out_image = out_case_dir / f"{case_id}_image.nii.gz"
        out_mask = out_case_dir / f"{case_id}_mask.nii.gz"

        try:
            image_path, mask_path = self.find_image_mask(case_dir)

            if SKIP_EXISTING and self.existing_output_is_valid(out_image, out_mask):
                print("  already exists and valid, skip")
                return

            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))

            # -------------------------------------------------
            # Avoid unintended resampling caused by header mismatch.
            # -------------------------------------------------
            size_img = image.GetSize()
            size_mask = mask.GetSize()
            spacing_img = image.GetSpacing()
            spacing_mask = mask.GetSpacing()

            geometry_needs_resampling = False
            if size_img != size_mask or not np.allclose(spacing_img, spacing_mask, atol=1e-6):
                geometry_needs_resampling = True
                print("  WARNING: image and mask differ in size or spacing; "
                      "a physical resampling alignment will be performed.")
            else:
                # Same size and spacing: force mask to inherit image metadata
                # so the original alignment is preserved.
                mask.CopyInformation(image)
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                print("  info: mask inherited geometry metadata from image "
                      "(original alignment preserved).")

            geometry_same_before = self.same_geometry(image, mask)
            if not geometry_same_before:
                if geometry_needs_resampling:
                    print("  WARNING: mask and image geometry mismatch; "
                          "resampling mask to image grid first.")
                    self.print_geometry("image", image)
                    self.print_geometry("mask ", mask)
                    mask = self.resample_mask_to_image(mask, image)
                else:
                    raise RuntimeError(
                        "Same size and spacing but geometry still differs "
                        "after CopyInformation; please inspect the input data."
                    )

            geometry_same_after_align = self.same_geometry(image, mask)
            if not geometry_same_after_align:
                raise RuntimeError(
                    "Geometry still differs after aligning mask to image."
                )
            # -------------------------------------------------

            original_image_size = tuple(int(v) for v in image.GetSize())
            original_image_spacing = tuple(float(v) for v in image.GetSpacing())
            original_mask_size = tuple(int(v) for v in mask.GetSize())
            original_mask_spacing = tuple(float(v) for v in mask.GetSpacing())

            print(f"  original image size xyz    : {original_image_size}")
            print(f"  original image spacing     : {original_image_spacing}")
            print(f"  original mask  size xyz    : {original_mask_size}")
            print(f"  original mask  spacing     : {original_mask_spacing}")

            mask_voxels_before = int(sitk.GetArrayFromImage(mask).sum())

            # -------------------------------------------------
            # Crop or pad XY to 400 x 400
            # -------------------------------------------------
            crop400_params = self.get_xy_crop_pad_params(
                image,
                target_x=self.target_x,
                target_y=self.target_y
            )

            print(
                f"  info: crop400 params: "
                f"start=({crop400_params['start_x']}, {crop400_params['start_y']}), "
                f"pad_lower=({crop400_params['pad_lower_x']}, {crop400_params['pad_lower_y']}), "
                f"pad_upper=({crop400_params['pad_upper_x']}, {crop400_params['pad_upper_y']})"
            )

            image_crop = self.crop_or_pad_xy_with_params(
                image, crop400_params,
                target_x=self.target_x, target_y=self.target_y,
                is_mask=False
            )
            mask_crop = self.crop_or_pad_xy_with_params(
                mask, crop400_params,
                target_x=self.target_x, target_y=self.target_y,
                is_mask=True
            )

            crop_size = tuple(int(v) for v in image_crop.GetSize())
            crop_spacing = tuple(float(v) for v in image_crop.GetSpacing())

            print(f"  info: after crop400 size xyz    = {crop_size}")
            print(f"  info: after crop400 spacing xyz = {crop_spacing}")

            if image_crop.GetSize()[:2] != (self.target_x, self.target_y):
                raise RuntimeError(
                    f"Unexpected image XY size after crop400: {image_crop.GetSize()}"
                )
            if mask_crop.GetSize()[:2] != (self.target_x, self.target_y):
                raise RuntimeError(
                    f"Unexpected mask XY size after crop400: {mask_crop.GetSize()}"
                )
            if not self.same_geometry(image_crop, mask_crop):
                raise RuntimeError(
                    "Image and mask geometry mismatch after crop400."
                )

            mask_voxels_after_crop = int(sitk.GetArrayFromImage(mask_crop).sum())

            # -------------------------------------------------
            # Keep the central 24 slices along Z
            # -------------------------------------------------
            original_z_after_xy = int(image_crop.GetSize()[2])
            z_action = self.get_z_action(original_z_after_xy)
            print(f"  info: Z action = {z_action}")

            image_out = self.fix_z_to_24_keep_middle(image_crop, is_mask=False)
            mask_out = self.fix_z_to_24_keep_middle(mask_crop, is_mask=True)

            final_size = tuple(int(v) for v in image_out.GetSize())
            final_spacing = tuple(float(v) for v in image_out.GetSpacing())
            final_mask_spacing = tuple(float(v) for v in mask_out.GetSpacing())
            mask_voxels_after_final = int(sitk.GetArrayFromImage(mask_out).sum())

            if image_out.GetSize() != (self.target_x, self.target_y, self.target_z):
                raise RuntimeError(
                    f"Unexpected final image size: {image_out.GetSize()}"
                )
            if mask_out.GetSize() != (self.target_x, self.target_y, self.target_z):
                raise RuntimeError(
                    f"Unexpected final mask size: {mask_out.GetSize()}"
                )
            if not self.same_geometry(image_out, mask_out):
                raise RuntimeError(
                    "Image and mask geometry mismatch before writing outputs."
                )

            sitk.WriteImage(sitk.Cast(image_out, sitk.sitkFloat32), str(out_image))
            sitk.WriteImage(sitk.Cast(mask_out, sitk.sitkUInt8), str(out_mask))

            self.success.append({
                "case_id": case_id,
                "input_image_path": str(image_path),
                "input_mask_path": str(mask_path),
                "output_image_path": str(out_image),
                "output_mask_path": str(out_mask),

                "original_image_size_xyz": str(original_image_size),
                "original_image_spacing_xyz": str(original_image_spacing),
                "original_mask_size_xyz": str(original_mask_size),
                "original_mask_spacing_xyz": str(original_mask_spacing),

                "geometry_same_before": bool(geometry_same_before),
                "geometry_same_after_align": bool(geometry_same_after_align),

                "crop400_start_x": int(crop400_params["start_x"]),
                "crop400_start_y": int(crop400_params["start_y"]),
                "crop400_end_x": int(crop400_params["end_x"]),
                "crop400_end_y": int(crop400_params["end_y"]),
                "crop400_pad_lower_x": int(crop400_params["pad_lower_x"]),
                "crop400_pad_lower_y": int(crop400_params["pad_lower_y"]),
                "crop400_pad_upper_x": int(crop400_params["pad_upper_x"]),
                "crop400_pad_upper_y": int(crop400_params["pad_upper_y"]),
                "crop400_size_xyz": str(crop_size),
                "crop400_spacing_xyz": str(crop_spacing),

                "original_z_after_xy": int(original_z_after_xy),
                "z_action": z_action,

                "final_size_xyz": str(final_size),
                "final_spacing_xyz": str(final_spacing),
                "final_mask_spacing_xyz": str(final_mask_spacing),

                "mask_voxels_before": int(mask_voxels_before),
                "mask_voxels_after_crop": int(mask_voxels_after_crop),
                "mask_voxels_after_final": int(mask_voxels_after_final),

                "status": "success"
            })

            if mask_voxels_before > 0 and mask_voxels_after_final == 0:
                print("  WARNING: final mask is empty; the ROI may have been "
                      "completely removed by cropping.")

            print("  saved")
            print(f"  final xyz size          : {final_size}")
            print(f"  final image spacing xyz : {final_spacing}")
            print(f"  final mask  spacing xyz : {final_mask_spacing}")

        except Exception as e:
            self.failed.append({
                "case_id": case_id,
                "case_dir": str(case_dir),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "failed"
            })

            print(f"  FAILED: {e}")
            traceback.print_exc()

    # =========================================================
    # 14. Save reports
    # =========================================================
    def save_reports(self):
        if len(self.success) > 0:
            pd.DataFrame(self.success).to_csv(
                self.success_csv,
                index=False,
                encoding="utf-8-sig"
            )
        else:
            pd.DataFrame(columns=["case_id"]).to_csv(
                self.success_csv,
                index=False,
                encoding="utf-8-sig"
            )

        if len(self.failed) > 0:
            pd.DataFrame(self.failed).to_csv(
                self.failed_csv,
                index=False,
                encoding="utf-8-sig"
            )
        else:
            pd.DataFrame(columns=["case_id", "error"]).to_csv(
                self.failed_csv,
                index=False,
                encoding="utf-8-sig"
            )

        summary = {
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "cases_dir": str(self.cases_dir),
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_z": self.target_z,
            "process_order": (
                "1) crop_or_pad_xy_to_400_by_image_geometric_center; "
                "2) keep_middle_24_in_z"
            ),
            "center_mode": "image_geometric_center",
            "z_mode": "keep_middle_24__pad_zero_by_center_if_less_than_24",
            "skip_existing": SKIP_EXISTING,
            "success_cases": len(self.success),
            "failed_cases": len(self.failed),
        }

        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # =========================================================
    # 15. Main run
    # =========================================================
    def run(self):
        case_dirs = sorted([p for p in self.input_root.iterdir() if p.is_dir()])

        if START_CASE_ID is not None:
            case_dirs = [p for p in case_dirs if p.name >= START_CASE_ID]

        print(f"Found {len(case_dirs)} cases")
        print(f"Input  root: {self.input_root}")
        print(f"Output root: {self.output_root}")
        print(f"Target XY  : ({self.target_x}, {self.target_y})")
        print(f"Target Z   : {self.target_z}")

        for idx, case_dir in enumerate(case_dirs, start=1):
            self.process_one_case(case_dir, case_index=idx)
            self.save_reports()

        self.save_reports()

        print("\n" + "=" * 70)
        print("Preprocess done")
        print("Process order:")
        print("  1) crop/pad XY to 400x400 by image geometric center")
        print("  2) keep middle 24 slices in Z")
        print(f"success: {len(self.success)}")
        print(f"failed : {len(self.failed)}")
        print(f"output : {self.output_root}")
        print("=" * 70)


# =========================================================
# 16. Entry point
# =========================================================
if __name__ == "__main__":
    runner = PreprocessXYCrop400MiddleZ24Center()
    runner.run()