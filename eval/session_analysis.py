import os
import sys
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.multi_turn_evaluation import EVAL_DIR


def _holm_bonferroni_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    adjusted_values = np.empty(m, dtype=float)
    for i in range(m):
        adjusted_values[i] = ranked[i] * m / (i + 1)
    adjusted_values = np.minimum.accumulate(adjusted_values[::-1])[::-1]
    adjusted_values = np.clip(adjusted_values, 0.0, 1.0)
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_values
    return [float(v) for v in adjusted]


def _paired_test_and_effect_size(full_values: List[float], ablation_values: List[float]) -> Dict[str, object]:
    diffs = np.asarray([b - a for a, b in zip(full_values, ablation_values)], dtype=float)
    if diffs.size == 0:
        return {"test": "none", "p_value": 1.0, "effect_size": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    sem = float(np.std(diffs, ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else 0.0
    ci_low, ci_high = stats.t.interval(0.95, diffs.size - 1, loc=float(np.mean(diffs)), scale=sem) if diffs.size > 1 else (float(np.mean(diffs)), float(np.mean(diffs)))

    shapiro_stat, shapiro_p = stats.shapiro(diffs) if diffs.size >= 3 else (np.nan, 1.0)
    if np.isfinite(shapiro_p) and shapiro_p > 0.05 and diffs.size > 1:
        test_name = "paired_t_test"
        p_value = float(stats.ttest_rel(full_values, ablation_values).pvalue)
        pooled_std = np.sqrt((np.var(full_values, ddof=1) + np.var(ablation_values, ddof=1)) / 2.0) if len(full_values) > 1 and len(ablation_values) > 1 else 0.0
        effect_size = float(np.mean(diffs) / pooled_std) if pooled_std > 0 else 0.0
    else:
        test_name = "wilcoxon_signed_rank"
        p_value = float(stats.wilcoxon(full_values, ablation_values, zero_method="wilcox", correction=False).pvalue) if diffs.size > 1 else 1.0
        effect_size = float(np.mean(np.sign(diffs))) if diffs.size > 0 else 0.0

    return {
        "test": test_name,
        "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "effect_size": float(effect_size) if np.isfinite(effect_size) else 0.0,
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
    }


def analyze_sessions(results_csv: Optional[str] = None, output_dir: Optional[str] = None) -> pd.DataFrame:
    if results_csv is None:
        results_csv = os.path.join(EVAL_DIR, "multi_turn_results.csv")
    if output_dir is None:
        output_dir = EVAL_DIR

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    metric_cols = [
        "personalization",
        "adaptive_teaching",
        "difficulty_alignment",
        "weak_concept_recall",
        "pedagogical_quality",
        "learning_gain",
        "knowledge_retention",
        "tutor_consistency",
        "hallucination_rate",
        "latency_ms",
        "overall_quality",
    ]

    rows = []
    session_rows = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        for profile_name in sorted(df["profile"].dropna().unique()):
            full_group = df[(df["profile"] == profile_name) & (df["variant"] == "full")].copy()
            if full_group.empty:
                continue
            for variant in sorted(df["variant"].dropna().unique()):
                if variant == "full":
                    continue
                ablation_group = df[(df["profile"] == profile_name) & (df["variant"] == variant)].copy()
                if ablation_group.empty:
                    continue

                joined = full_group.merge(ablation_group[["session_id", metric]], on="session_id", suffixes=("_full", "_ablation"))
                if joined.empty:
                    continue

                full_values = joined[f"{metric}_full"].astype(float).tolist()
                ablation_values = joined[f"{metric}_ablation"].astype(float).tolist()
                diffs = np.asarray([b - a for a, b in zip(full_values, ablation_values)], dtype=float)
                mean_diff = float(np.mean(diffs))
                std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                sem_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
                ci_low, ci_high = stats.t.interval(0.95, len(diffs) - 1, loc=mean_diff, scale=sem_diff) if len(diffs) > 1 else (mean_diff, mean_diff)

                test_result = _paired_test_and_effect_size(full_values, ablation_values)
                p_value = test_result["p_value"]
                effect_size = test_result["effect_size"]
                rows.append({
                    "profile": profile_name,
                    "metric": metric,
                    "variant": variant,
                    "n_sessions": len(diffs),
                    "mean_full": round(float(np.mean(full_values)), 4),
                    "mean_ablation": round(float(np.mean(ablation_values)), 4),
                    "mean_difference": round(mean_diff, 4),
                    "std_difference": round(std_diff, 4),
                    "median_difference": round(float(np.median(diffs)), 4),
                    "iqr_difference": round(float(np.percentile(diffs, 75) - np.percentile(diffs, 25)), 4),
                    "std_error": round(sem_diff, 4),
                    "ci_lower": round(float(ci_low), 4),
                    "ci_upper": round(float(ci_high), 4),
                    "test": test_result["test"],
                    "p_value": round(float(p_value), 4),
                    "effect_size_cohens_d": round(float(effect_size), 4),
                })

                for _, row in joined.iterrows():
                    session_rows.append({
                        "profile": profile_name,
                        "variant": variant,
                        "metric": metric,
                        "session_id": int(row["session_id"]),
                        "full_value": float(row[f"{metric}_full"]),
                        "ablation_value": float(row[f"{metric}_ablation"]),
                        "difference": float(row[f"{metric}_ablation"] - row[f"{metric}_full"]),
                    })

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["adjusted_p_value"] = _holm_bonferroni_adjust(stats_df["p_value"].astype(float).tolist())
        stats_df.to_csv(os.path.join(output_dir, "multi_turn_statistical_tests.csv"), index=False)

    if session_rows:
        pd.DataFrame(session_rows).to_csv(os.path.join(output_dir, "multi_turn_raw_session_differences.csv"), index=False)

    summary_stats = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        for (variant, profile_name), group in df.groupby(["variant", "profile"]):
            values = group[metric].astype(float).tolist()
            if not values:
                continue
            summary_stats.append({
                "variant": variant,
                "profile": profile_name,
                "metric": metric,
                "n_sessions": len(values),
                "mean": round(float(np.mean(values)), 4),
                "median": round(float(np.median(values)), 4),
                "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
                "iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 4),
                "stderr": round(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0, 4),
                "ci_lower": round(float(stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))[0]) if len(values) > 1 else float(np.mean(values)), 4),
                "ci_upper": round(float(stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))[1]) if len(values) > 1 else float(np.mean(values)), 4),
            })

    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(os.path.join(output_dir, "multi_turn_session_statistics.csv"), index=False)

    # Publication-ready visuals.
    sns.set_theme(style="whitegrid")
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x="variant", y=metric, hue="profile")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"), dpi=200)
        plt.close()

    return stats_df


def main() -> None:
    analyze_sessions()


if __name__ == "__main__":
    main()
