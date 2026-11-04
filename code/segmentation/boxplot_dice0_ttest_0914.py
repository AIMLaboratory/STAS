import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


# ============================================================
# FIGURE 3
# nnU-Net segmentation performance across independent test sets
#
# a: Dice coefficient
# b: HD95
#
# Features:
# - 3 independent test sets only
# - transparent boxplots
# - patient-level points visible above boxes
# - mean = yellow triangle
# - median = center line
# - whiskers = 2 × IQR
# - PNG/TIFF: 600 dpi
# - PDF vector output
# ============================================================


# ============================================================
# 1. INPUT FILES
# ============================================================

# Must match the per-dataset output paths produced by
#   predict_best_fold_official_nnunet.py            (IT1, IT2)
#   predict_best_fold_official_nnunet_external.py   (ET)
_METRICS_ROOT = Path(
    r"RECTAL_STAS_DATA\nnUNet_bestfold_prediction_and_metrics"
)

FILES = {
    "Internal test set 1": (
        _METRICS_ROOT / "Internal test set 1" / "Internal test set 1_metrics.xlsx"
    ),
    "Internal test set 2": (
        _METRICS_ROOT / "Internal test set 2" / "Internal test set 2_metrics.xlsx"
    ),
    "external_test_set": (
        _METRICS_ROOT / "external_test_set" / "external_test_set_metrics.xlsx"
    ),
}


# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path(
    r"RECTAL_STAS_DATA\nnUNet_bestfold_prediction_and_metrics"
    r"\metrics_only\Figure3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. DATASET ORDER
# ============================================================

dataset_order = [
    "Internal test set 1",
    "Internal test set 2",
    "external_test_set",
]

dataset_labels = {
    "Internal test set 1": "Internal test 1",
    "Internal test set 2": "Internal test 2",
    "external_test_set":   "external test set",
}




# ============================================================
# 4. COLORS
# matched to Figure 1
# ============================================================

PALETTE = {
    "Internal test set 1": "#83CBEB",
    "Internal test set 2": "#F6C6AD",
    "external_test_set":   "#F2CFEE",
}


# ============================================================
# 5. GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,

    "axes.labelsize": 12,
    "axes.labelweight": "bold",

    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    "axes.linewidth": 1.1,

    # Keep fonts editable in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

sns.set_style("white")


# ============================================================
# 6. READ DATA
# ============================================================

all_dfs = []

for dataset_name, file_path in FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    df = pd.read_excel(file_path)

    # force consistent dataset identifier
    df["dataset"] = dataset_name

    if "modality" not in df.columns:
        df["modality"] = "T2WI"

    all_dfs.append(df)


merged_df = pd.concat(
    all_dfs,
    ignore_index=True
)


# ============================================================
# 7. SAVE MERGED DATA
# ============================================================

merged_path = (
    OUTPUT_DIR /
    "merged_3_independent_test_sets.xlsx"
)

merged_df.to_excel(
    merged_path,
    index=False
)

print(
    f"\nMerged patient-level data saved to:\n"
    f"{merged_path}"
)


# ============================================================
# 8. DATA CLEANING
# ============================================================

merged_df = merged_df[
    merged_df["dataset"].isin(dataset_order)
].copy()


# Convert metrics to numeric
merged_df["dice"] = pd.to_numeric(
    merged_df["dice"],
    errors="coerce"
)

merged_df["hd95"] = pd.to_numeric(
    merged_df["hd95"],
    errors="coerce"
)


# ------------------------------------------------------------
# Dice
# ------------------------------------------------------------

dice_df = (
    merged_df
    .dropna(subset=["dice"])
    .copy()
)

dice_df = dice_df[
    (dice_df["dice"] >= 0) &
    (dice_df["dice"] <= 1)
]


# ------------------------------------------------------------
# HD95
# ------------------------------------------------------------

hd95_df = (
    merged_df
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(subset=["hd95"])
    .copy()
)

hd95_df = hd95_df[
    hd95_df["hd95"] >= 0
]


# ============================================================
# 9. SAMPLE SIZE CHECK
# ============================================================

print("\n====================================")
print("Sample sizes")
print("====================================")

for ds in dataset_order:

    n = (
        merged_df["dataset"] == ds
    ).sum()

    print(
        f"{dataset_labels[ds]}: n = {n}"
    )


# ============================================================
# 10. DESCRIPTIVE STATISTICS
# ============================================================

print("\n====================================")
print("Dice descriptive statistics")
print("====================================")

print(
    dice_df
    .groupby("dataset")["dice"]
    .describe()
    .reindex(dataset_order)
)


print("\n====================================")
print("HD95 descriptive statistics")
print("====================================")

print(
    hd95_df
    .groupby("dataset")["hd95"]
    .describe()
    .reindex(dataset_order)
)


# ============================================================
# 11. SUMMARY STATISTICS
# ============================================================

def summary_stats(series):

    x = (
        series
        .dropna()
        .astype(float)
    )

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    return pd.Series({
        "N": len(x),
        "Mean": x.mean(),
        "SD": x.std(ddof=1),
        "Median": x.median(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "Min": x.min(),
        "Max": x.max()
    })


dice_summary = (
    dice_df
    .groupby("dataset")["dice"]
    .apply(summary_stats)
    .unstack()
    .reindex(dataset_order)
)


hd95_summary = (
    hd95_df
    .groupby("dataset")["hd95"]
    .apply(summary_stats)
    .unstack()
    .reindex(dataset_order)
)


summary_path = (
    OUTPUT_DIR /
    "segmentation_descriptive_statistics.xlsx"
)


with pd.ExcelWriter(summary_path) as writer:

    dice_summary.to_excel(
        writer,
        sheet_name="Dice"
    )

    hd95_summary.to_excel(
        writer,
        sheet_name="HD95"
    )


print(
    f"\nDescriptive statistics saved to:\n"
    f"{summary_path}"
)


# ============================================================
# 11b. STATISTICAL TESTS: Internal test 1 vs Internal test 2
# ============================================================

def statistical_tests(df, metric, group1, group2, label1, label2):
    """
    Two-sided Welch's two-sample t-test, as specified in the manuscript.
    """
    g1 = df[df["dataset"] == group1][metric].dropna().astype(float)
    g2 = df[df["dataset"] == group2][metric].dropna().astype(float)

    n1, n2 = len(g1), len(g2)
    mean1, mean2 = g1.mean(), g2.mean()
    sd1, sd2 = g1.std(ddof=1), g2.std(ddof=1)
    median1, median2 = g1.median(), g2.median()
    q1_1, q3_1 = g1.quantile(0.25), g1.quantile(0.75)
    q1_2, q3_2 = g2.quantile(0.25), g2.quantile(0.75)

    # Two-sided Welch's t-test (equal_var=False)
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

    # Welch–Satterthwaite degrees of freedom
    mean_diff = mean1 - mean2
    se_welch = np.sqrt(sd1**2 / n1 + sd2**2 / n2) if n1 > 0 and n2 > 0 else np.nan
    df_welch = (
        (sd1**2 / n1 + sd2**2 / n2) ** 2
        / ((sd1**2 / n1) ** 2 / (n1 - 1) + (sd2**2 / n2) ** 2 / (n2 - 1))
        if n1 > 1 and n2 > 1 else np.nan
    )
    t_crit = stats.t.ppf(0.975, df_welch) if not np.isnan(df_welch) else np.nan
    ci_low = mean_diff - t_crit * se_welch if not np.isnan(se_welch) else np.nan
    ci_high = mean_diff + t_crit * se_welch if not np.isnan(se_welch) else np.nan

    # Cohen's d (based on pooled SD)
    pooled_sd = (
        np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        if (n1 + n2 - 2) > 0 else np.nan
    )
    cohens_d = mean_diff / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

    print(f"\n{'='*70}")
    print(f"{metric.upper()} — {label1} vs {label2}")
    print(f"{'='*70}")
    print(f"{label1}: n={n1}, mean±SD = {mean1:.4f} ± {sd1:.4f}, "
          f"median (IQR) = {median1:.4f} ({q1_1:.4f}–{q3_1:.4f})")
    print(f"{label2}: n={n2}, mean±SD = {mean2:.4f} ± {sd2:.4f}, "
          f"median (IQR) = {median2:.4f} ({q1_2:.4f}–{q3_2:.4f})")
    print(f"\n--- Two-sided Welch's t-test ---")
    print(f"Welch's t-test: t = {t_stat:.4f}, df = {df_welch:.4f}, p = {p_val:.4e}, "
          f"95% CI ({ci_low:.4f}, {ci_high:.4f}), Cohen's d = {cohens_d:.4f}")

    return {
        "Metric": metric,
        "Group1": label1,
        "Group2": label2,
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "median1": median1,
        "median2": median2,
        "IQR1": f"{q1_1:.4f}–{q3_1:.4f}",
        "IQR2": f"{q1_2:.4f}–{q3_2:.4f}",
        "Mean_difference": mean_diff,
        "Welch_t": t_stat,
        "Welch_df": df_welch,
        "Welch_p": p_val,
        "Welch_CI_low": ci_low,
        "Welch_CI_high": ci_high,
        "Cohens_d": cohens_d,
    }

# Internal test 1 vs Internal test 2
results_dice = statistical_tests(
    dice_df, "dice",
    group1="Internal test set 1", group2="Internal test set 2",
    label1="Internal test 1", label2="Internal test 2",
)

results_hd95 = statistical_tests(
    hd95_df, "hd95",
    group1="Internal test set 1", group2="Internal test set 2",
    label1="Internal test 1", label2="Internal test 2",
)


stats_results = pd.DataFrame([results_dice, results_hd95])
stats_path = OUTPUT_DIR / "statistical_tests_internal1_vs_internal2.xlsx"
stats_results.to_excel(stats_path, index=False)
print(f"\nStatistical test results saved to:\n{stats_path}")


# ============================================================
# 12. COMMON PLOTTING FUNCTION
# ============================================================

def plot_segmentation_metric(
    df,
    metric,
    ylabel,
    output_name,
    ylim,
    yticks,
    panel_label
):

    """
    Publication-style boxplot.

    Box:
        Q1–Q3

    Center line:
        median

    Yellow triangle:
        mean

    Gray dots:
        individual patients

    Whiskers:
        2 × IQR

    Important:
        box fill is semi-transparent so patient-level
        points remain visible.
    """


    # --------------------------------------------------------
    # Figure size
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )


    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------

    sns.boxplot(
        data=df,

        x="dataset",
        y=metric,

        order=dataset_order,

        palette=[
            PALETTE[x]
            for x in dataset_order
        ],

        width=0.42,

        # whiskers = 2 × IQR
        whis=2.0,

        # all patient values are shown separately below
        showfliers=False,

        linewidth=1.2,

        saturation=1,

        # IMPORTANT:
        # alpha controls box transparency
        boxprops={
            "alpha": 0.6,
            "edgecolor": "black",
            "linewidth": 1.2
        },

        medianprops={
            "color": "black",
            "linewidth": 1.6
        },

        whiskerprops={
            "color": "black",
            "linewidth": 1.1
        },

        capprops={
            "color": "black",
            "linewidth": 1.1
        },

        ax=ax
    )


    # --------------------------------------------------------
    # Patient-level points
    #
    # zorder=5 makes the dots appear ABOVE the transparent box.
    # --------------------------------------------------------

    sns.stripplot(
        data=df,

        x="dataset",
        y=metric,

        order=dataset_order,

        color="#777777",

        alpha=0.60,

        jitter=0.20,

        size=4.3,

        zorder=5,

        ax=ax
    )


    # --------------------------------------------------------
    # Mean marker
    # --------------------------------------------------------

    means = (
        df
        .groupby("dataset")[metric]
        .mean()
    )


    for i, ds in enumerate(dataset_order):

        if ds in means.index:

            ax.scatter(
                i,
                means.loc[ds],

                marker="^",

                s=85,

                facecolor="#FFD700",

                edgecolor="black",

                linewidth=0.75,

                zorder=10
            )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel("")

    ax.set_ylabel(
        ylabel,
        fontsize=14,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # X labels
    # --------------------------------------------------------

    ax.set_xticks(
        range(
            len(dataset_order)
        )
    )

    ax.set_xticklabels(
        [
            dataset_labels[x]
            for x in dataset_order
        ],

        rotation=0,

        ha="center",

        fontsize=14,

        fontweight="bold"
    )


    # --------------------------------------------------------
    # Y-axis
    # --------------------------------------------------------

    ax.set_ylim(
        ylim
    )

    ax.set_yticks(
        yticks
    )


    # --------------------------------------------------------
    # Journal-style axes
    # --------------------------------------------------------

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_linewidth(
        1.1
    )

    ax.spines["bottom"].set_linewidth(
        1.1
    )

    ax.tick_params(
        axis="both",

        width=1,

        length=4,

        direction="out"
    )

    ax.grid(False)


    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.10,
        1.04,

        panel_label,

        transform=ax.transAxes,

        fontsize=16,

        fontweight="bold",

        va="top",
        ha="left"
    )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        pad=1.2
    )


    # ========================================================
    # SAVE PNG — 600 DPI
    # ========================================================

    png_path = (
        OUTPUT_DIR /
        f"{output_name}.png"
    )

    fig.savefig(
        png_path,

        dpi=600,

        bbox_inches="tight",

        facecolor="white",

        transparent=False
    )


    # ========================================================
    # SAVE TIFF — 600 DPI
    # ========================================================

    tiff_path = (
        OUTPUT_DIR /
        f"{output_name}.tiff"
    )

    fig.savefig(
        tiff_path,

        dpi=600,

        bbox_inches="tight",

        facecolor="white",

        transparent=False
    )


    # ========================================================
    # SAVE VECTOR PDF
    # ========================================================

    pdf_path = (
        OUTPUT_DIR /
        f"{output_name}.pdf"
    )

    fig.savefig(
        pdf_path,

        bbox_inches="tight",

        facecolor="white",

        transparent=False
    )


    print(
        "\nSaved:"
    )

    print(
        png_path
    )

    print(
        tiff_path
    )

    print(
        pdf_path
    )


    plt.show()

    plt.close(fig)


# ============================================================
# 13. FIGURE 3A — DICE
# ============================================================

plot_segmentation_metric(
    df=dice_df,

    metric="dice",

    ylabel="Dice coefficient",

    output_name="Figure3a_Dice",

    ylim=(
        0,
        1.02
    ),

    yticks=np.arange(
        0,
        1.01,
        0.2
    ),

    panel_label="a"
)


# ============================================================
# 14. FIGURE 3B — HD95
#
# Maximum external HD95 ≈ 152.87 mm,
# so the upper limit is fixed at 160 mm.
# ============================================================

plot_segmentation_metric(
    df=hd95_df,

    metric="hd95",

    ylabel="HD95 (mm)",

    output_name="Figure3b_HD95",

    ylim=(
        0,
        160
    ),

    yticks=np.arange(
        0,
        161,
        20
    ),

    panel_label="b"
)


# ============================================================
# 15. FINISHED
# ============================================================

print("\n====================================")
print("Figure 3 completed")
print("====================================")

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)