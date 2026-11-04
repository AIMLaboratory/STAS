# STAS
# STAS — Segmentation-Guided Deep Learning Pipeline for Pathologic T-Stage Classification in Rectal Cancer

Reference implementation of the STAS pipeline used in the manuscript
*Deep Learning-Based Segmentation-Guided T-Stage Assessment System (STAS) of Rectal Cancer on MRI: A Multicenter Study*.

The pipeline has two stages:

1. **Tumor segmentation** — nnU-Net v2 `3d_fullres` on 617 development patients
   with fixed five-fold cross-validation. The official nnU-Net network
   architecture, trainer, and plans are used **without modification**; planning
   (target spacing, patch size, batch size) and intensity normalization are
   determined automatically by nnU-Net. Prior to nnU-Net preprocessing, a
   study-specific spatial standardization step (described below) resamples each
   image–mask pair to 24 × 400 × 400 voxels.
2. **pT-stage classification** (pT1–2 vs pT3–4) — 3D DenseNet121 (Models 1–3) and
   ResNet34 (Models 4–6) over three input configurations, sharing the same
   617-patient five-fold split as segmentation.

All test-set predictions use frozen models selected on development validation
data only. No test-set data were used for model selection or threshold tuning.

---

## Repository layout

```
code/
  segmentation/      # nnU-Net data prep, five-fold training, OOF, IT1/IT2/ET inference
  classification/    # DenseNet121 / ResNet34 training, independent test, DeLong, jackknife, Grad-CAM
  requirements.txt
README.md

```

---

## Environment

* Python 3.10.19, PyTorch 2.5.1 (CUDA 12.x)
* nnU-Net v2 (version pinned in `requirements.txt`)
* Single NVIDIA GPU with ≥ 24 GB memory

```bash
pip install -r requirements.txt
```

`nnunetv2` CLI tools (`nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`) must be on `PATH`.

---

## Data layout

Each patient folder:

```
<cases_root>/<case_id>/<case_id>_image.nii.gz
<cases_root>/<case_id>/<case_id>_mask.nii.gz     # manual mask, binary 0/1
<cases_root>/<case_id>/<case_id>_pred.nii.gz     # nnU-Net predicted mask, binary 0/1
```

**Study-specific spatial standardization (before nnU-Net).**
`preprocess_standardization_400x400x24.py` centre-crops or zero-pads each
image–mask pair (preserving image–mask spatial correspondence) to
24 × 400 × 400 voxels. This is the only step performed outside the official
nnU-Net pipeline. All subsequent nnU-Net planning, preprocessing, training, and
inference use the unmodified official configuration
(`configuration = "3d_fullres"`, `trainer = "nnUNetTrainer"`, `plans = "nnUNetPlans"`).

Development cohort (n = 617). Required auxiliary files:

| File | Purpose |
|---|---|
| `merged_tstage_labels.csv` | pT-stage labels |
| `case_mapping.json` | `source_case_id` ↔ `RCM_XXXX` (from `prepare_dataset.py`) |
| `splits_final.json` | nnU-Net five-fold split (frozen) |
| `model2_oof_mask_index_617.csv` | per-patient OOF mask index |

Independent test cohorts: IT1 (n = 47), IT2 (n = 31), ET (n = 83).
Each cohort supplies its own `cases_dir` and either a fixed CSV
(`case_id_norm`, `binary_label`) or an XLSX with `FolderName`,
`Histology results _Tstage`, and — for internal cohorts — radiologist T-stage columns.

---

## Running the pipeline

```bash
# ---- Segmentation ----
python code/segmentation/preprocess_standardization_400x400x24.py
python code/segmentation/run_nnunet_pipeline.py prepare
python code/segmentation/run_nnunet_pipeline.py preprocess
python code/segmentation/run_nnunet_pipeline.py train_all

python code/segmentation/predict_best_fold_official_nnunet.py           # IT1, IT2
python code/segmentation/predict_best_fold_official_nnunet_external.py  # ET
python code/segmentation/predict_oof_nnunet_617_SAFE.py                 # 617 OOF masks
python code/segmentation/boxplot_dice0_ttest_0914.py                    # Figure 3 + Welch t-test

# ---- Classification training ----
python code/classification/train_DenseNet121_Models1_2_3_SHARED5FOLD_OOF.py
python code/classification/train_ResNet34_Models4_5_6_SHARED5FOLD_OOF.py

# ---- Independent testing ----
python code/classification/test_DenseNet121_Models1_2_3_IT1_IT2_ET_FULL_PAPER_STYLE.py
python code/classification/test_ResNet34_Models4_5_6_IT1_IT2_ET_FULL_PAPER_STYLE.py

# ---- Exploratory IT1-vs-IT2 comparison ----
python code/classification/compare_internal_test_sets_within_method0914.py

# ---- Model 2 Grad-CAM ----
python code/classification/generate_Model2_OOF_GradCAM_6PNG_STRICT_ALIGNED_MIXED_LAYOUT.py
```

**Segmentation fold selection.** `run_nnunet_pipeline.py train_all` trains folds 0–4
sequentially. The fixed segmentation model is chosen by reading each fold's
`validation/summary.json` and averaging all reported Dice values
(`FOLD_SELECTION_MODE = "validation_dice_mean"`). Test-set metrics are never read.

**Classification fold selection.** Each training script freezes the same
5-fold patient membership across all three input configurations of a backbone,
trains each fold with focal loss + AdamW + cosine warm restart + early stopping
on validation AUC, then selects — across the five folds of a model — the fold with
the highest validation AUC. The validation-derived threshold (Youden index on
that fold's validation subset) is frozen.

Both test scripts support `--same_fold N`, a **pre-registered** fallback policy
that forces the same classifier fold for all three input modes. Do not choose
`--same_fold` or the threshold from test performance.

---

## Data availability

Patient images and pathology labels cannot be shared publicly due to
institutional IRB restrictions. 

---

## Citation

If you use this code, please cite:

> *Deep Learning-Based Segmentation-Guided T-Stage Assessment System (STAS) of Rectal Cancer on MRI: A Multicenter Study*

BibTeX will be added after acceptance.

---

## License

Source code is released under the MIT license. See `LICENSE`. No patient data
are included under this license.
