
#!/usr/bin/env python3
"""
Inputs
------
data/raw/game_metadata.csv
data/raw/game_players_stats.csv
data/raw/champions_simple.json

Outputs
-------
data/processed/mid_players_2024.csv
data/processed/player_patch_summary.csv
data/processed/hypothesis_tests.csv
data/processed/overall_player_summary.csv
figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import entropy, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures"

PLAYERS = ["Faker", "Chovy", "ShowMaker", "Caps", "knight", "Bdd"]
PATCH_DATES = {
    # Effective dates are set to the day after the official patch-note publication
    # to give professional matches time to reflect the update in the public data.
    "14.5": pd.Timestamp("2024-03-06"),
    "14.13": pd.Timestamp("2024-06-26"),
    "14.16": pd.Timestamp("2024-08-14"),
}
MANUAL_TAGS = {
    "Hwei": ["Mage", "Support"],
    "Smolder": ["Marksman", "Mage"],
    "Zeri": ["Marksman"],
    "K'Sante": ["Tank", "Fighter"],
    "Vex": ["Mage"],
    "Aurora": ["Mage", "Assassin"],
}
METRICS = ["win", "kda", "cs_per_min", "dmg_per_min", "gold_per_min", "kp"]


def permutation_pvalue(
    faker_values: np.ndarray,
    peer_values: np.ndarray,
    *,
    alternative: str = "less",
    n_permutations: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Return observed Faker-peer mean difference and a permutation p-value.

    alternative="less" tests whether Faker's mean is smaller than peers.
    alternative="greater" tests whether Faker's mean is larger than peers.
    """
    rng = np.random.default_rng(seed)
    faker_values = np.asarray(faker_values, dtype=float)
    peer_values = np.asarray(peer_values, dtype=float)
    observed = float(faker_values.mean() - peer_values.mean())
    combined = np.concatenate([faker_values, peer_values])
    n_faker = len(faker_values)

    perm_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted = rng.permutation(combined)
        perm_diffs[i] = permuted[:n_faker].mean() - permuted[n_faker:].mean()

    if alternative == "less":
        p_value = float((np.sum(perm_diffs <= observed) + 1) / (n_permutations + 1))
    elif alternative == "greater":
        p_value = float((np.sum(perm_diffs >= observed) + 1) / (n_permutations + 1))
    else:
        p_value = float((np.sum(np.abs(perm_diffs) >= abs(observed)) + 1) / (n_permutations + 1))
    return observed, p_value


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the public professional match tables."""
    meta = pd.read_csv(RAW_DIR / "game_metadata.csv")
    stats = pd.read_csv(RAW_DIR / "game_players_stats.csv")
    return meta, stats


def load_champion_tags() -> pd.DataFrame:
    """Load champion tags and add a few manual mappings for newer champions."""
    with open(RAW_DIR / "champions_simple.json", "r", encoding="utf-8") as f:
        champions = json.load(f)

    rows = []
    for champ in champions:
        tags = champ.get("tags", [])
        rows.append(
            {
                "champion_name": champ["name"],
                "tags": ", ".join(tags),
                "primary_tag": tags[0] if tags else "Unknown",
            }
        )

    for name, tags in MANUAL_TAGS.items():
        rows.append(
            {"champion_name": name, "tags": ", ".join(tags), "primary_tag": tags[0]}
        )

    champ_df = pd.DataFrame(rows).drop_duplicates(subset=["champion_name"], keep="last")
    return champ_df


def prepare_match_level_data(meta: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Merge raw tables, filter the project sample, and create derived metrics."""
    df = stats.merge(meta, on="game_id", how="left", validate="many_to_one")
    df = df.rename(
        columns={
            "player_kills": "kills",
            "player_deaths": "deaths",
            "player_assists": "assists",
            "total_minions_killed": "cs",
            "gold_earned": "gold",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Focus the milestone on a manageable comparison set and one role.
    df = df[
        (df["player_name"].isin(PLAYERS))
        & (df["role"].eq("Mid"))
        & (df["date"] >= pd.Timestamp("2024-01-01"))
        & (df["date"] < pd.Timestamp("2024-09-15"))
    ].copy()

    # Remove potential repeated player-game rows if they appear in the public export.
    df = df.drop_duplicates(subset=["game_id", "player_id"]).copy()

    # Derived per-match metrics.
    df["minutes"] = df["game_length"] / 60.0
    df["deaths_safe"] = df["deaths"].replace(0, 1)
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths_safe"]
    df.loc[df["deaths"].eq(0), "kda"] = df.loc[df["deaths"].eq(0), "kills"] + df.loc[df["deaths"].eq(0), "assists"]

    df["cs_per_min"] = df["cs"] / df["minutes"]
    df["dmg_per_min"] = df["total_damage_dealt_to_champions"] / df["minutes"]
    df["gold_per_min"] = df["gold"] / df["minutes"]
    df["kp"] = (df["kills"] + df["assists"]) / df["team_kills"].replace(0, np.nan)
    df["kp"] = df["kp"].fillna(0)

    # Convert win to numeric to make the composite metric explicit and reproducible.
    df["win"] = df["win"].astype(int)

    # Standardize the selected metrics and create a composite performance index.
    for metric in METRICS:
        series = df[metric].astype(float)
        std = series.std(ddof=0)
        if std == 0:
            df[f"z_{metric}"] = 0.0
        else:
            df[f"z_{metric}"] = (series - series.mean()) / std
    z_cols = [f"z_{metric}" for metric in METRICS]
    df["performance_index"] = df[z_cols].mean(axis=1)

    return df.sort_values("date").reset_index(drop=True)


def enrich_with_champion_metadata(df: pd.DataFrame, champ_df: pd.DataFrame) -> pd.DataFrame:
    """Attach champion play-style metadata."""
    out = df.merge(champ_df, on="champion_name", how="left")
    out["primary_tag"] = out["primary_tag"].fillna("Unknown")
    out["tags"] = out["tags"].fillna("Unknown")
    return out


def shannon_entropy(values: Iterable[str]) -> float:
    """Return Shannon entropy of a discrete set of observations."""
    counts = pd.Series(list(values)).value_counts()
    if counts.empty:
        return float("nan")
    return float(entropy(counts.values, base=2))


def player_patch_summary(df: pd.DataFrame, window_days: int = 28) -> pd.DataFrame:
    """
    Build player-patch summaries.

    Each summary compares one player's average performance in the pre-patch and
    post-patch windows.
    """
    rows: List[Dict] = []
    for patch, patch_date in PATCH_DATES.items():
        pre_start = patch_date - pd.Timedelta(days=window_days)
        post_end = patch_date + pd.Timedelta(days=window_days)

        for player in PLAYERS:
            subset = df[df["player_name"].eq(player)].copy()
            pre = subset[(subset["date"] >= pre_start) & (subset["date"] < patch_date)].copy()
            post = subset[(subset["date"] >= patch_date) & (subset["date"] < post_end)].copy()
            if pre.empty or post.empty:
                continue

            pre_champs = set(pre["champion_name"].astype(str))
            post_champs = set(post["champion_name"].astype(str))
            new_post = post["champion_name"].isin(post_champs - pre_champs).mean()

            rows.append(
                {
                    "patch": patch,
                    "patch_date": patch_date,
                    "player_name": player,
                    "pre_games": len(pre),
                    "post_games": len(post),
                    "pre_perf": pre["performance_index"].mean(),
                    "post_perf": post["performance_index"].mean(),
                    "delta_perf": post["performance_index"].mean() - pre["performance_index"].mean(),
                    "abs_delta_perf": abs(post["performance_index"].mean() - pre["performance_index"].mean()),
                    "pre_winrate": pre["win"].mean(),
                    "post_winrate": post["win"].mean(),
                    "pre_diversity_unique": pre["champion_name"].nunique(),
                    "post_diversity_unique": post["champion_name"].nunique(),
                    "pre_diversity_entropy": shannon_entropy(pre["champion_name"]),
                    "post_diversity_entropy": shannon_entropy(post["champion_name"]),
                    "new_champion_rate": new_post,
                    "pre_primary_styles": ", ".join(sorted(pre["primary_tag"].dropna().unique())),
                    "post_primary_styles": ", ".join(sorted(post["primary_tag"].dropna().unique())),
                }
            )

    return pd.DataFrame(rows).sort_values(["patch_date", "player_name"]).reset_index(drop=True)


def overall_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create one concise summary table per player."""
    summary = (
        df.groupby("player_name")
        .agg(
            games=("game_id", "count"),
            leagues=("league_name", lambda s: ", ".join(sorted(set(s.dropna())))),
            winrate=("win", "mean"),
            avg_kda=("kda", "mean"),
            avg_perf=("performance_index", "mean"),
            perf_std=("performance_index", "std"),
            unique_champions=("champion_name", "nunique"),
            champion_entropy=("champion_name", shannon_entropy),
        )
        .reset_index()
        .sort_values("perf_std")
    )
    return summary


def run_hypothesis_tests(summary: pd.DataFrame) -> pd.DataFrame:
    """Run the milestone hypothesis tests and return a tidy results table."""
    faker = summary[summary["player_name"].eq("Faker")]
    peers = summary[~summary["player_name"].eq("Faker")]

    # H1: Faker has smaller average absolute post-patch change than peers.
    stat1, p1 = permutation_pvalue(
        faker["abs_delta_perf"].to_numpy(),
        peers["abs_delta_perf"].to_numpy(),
        alternative="less",
    )

    # H2: Faker has higher pre-patch diversity than peers.
    stat2, p2 = permutation_pvalue(
        faker["pre_diversity_entropy"].to_numpy(),
        peers["pre_diversity_entropy"].to_numpy(),
        alternative="greater",
    )

    # H3: Higher pre-patch diversity predicts smaller post-patch change.
    rho, p3 = spearmanr(summary["pre_diversity_entropy"], summary["abs_delta_perf"])

    # H4: Faker uses fewer truly new champions immediately after patches.
    stat4, p4 = permutation_pvalue(
        faker["new_champion_rate"].to_numpy(),
        peers["new_champion_rate"].to_numpy(),
        alternative="less",
    )

    results = pd.DataFrame(
        [
            {
                "hypothesis": "H1 Faker has smaller average absolute post-patch performance changes than peers",
                "test": "Permutation test (one-sided)",
                "statistic": "mean difference (Faker - peers)",
                "value": stat1,
                "p_value": p1,
                "faker_mean": faker["abs_delta_perf"].mean(),
                "peer_mean": peers["abs_delta_perf"].mean(),
            },
            {
                "hypothesis": "H2 Faker has higher pre-patch champion-pool diversity than peers",
                "test": "Permutation test (one-sided)",
                "statistic": "mean entropy difference (Faker - peers)",
                "value": stat2,
                "p_value": p2,
                "faker_mean": faker["pre_diversity_entropy"].mean(),
                "peer_mean": peers["pre_diversity_entropy"].mean(),
            },
            {
                "hypothesis": "H3 Higher pre-patch champion-pool diversity is associated with smaller absolute post-patch changes",
                "test": "Spearman correlation",
                "statistic": "rho",
                "value": rho,
                "p_value": p3,
                "faker_mean": np.nan,
                "peer_mean": np.nan,
            },
            {
                "hypothesis": "H4 Faker uses fewer truly new champions immediately after patches than peers",
                "test": "Permutation test (one-sided)",
                "statistic": "mean difference (Faker - peers)",
                "value": stat4,
                "p_value": p4,
                "faker_mean": faker["new_champion_rate"].mean(),
                "peer_mean": peers["new_champion_rate"].mean(),
            },
        ]
    )
    return results


def save_outputs(mid_df: pd.DataFrame, patch_summary: pd.DataFrame, tests: pd.DataFrame, overall: pd.DataFrame) -> None:
    """Write processed datasets to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    exported_mid_cols = [
        "game_id",
        "date",
        "league_name",
        "series_name",
        "tournament_name",
        "player_name",
        "team_name",
        "team_acronym",
        "role",
        "champion_name",
        "primary_tag",
        "tags",
        "win",
        "game_length",
        "player_kills",
        "player_deaths",
        "player_assists",
        "total_minions_killed",
        "gold_earned",
        "total_damage_dealt_to_champions",
        "team_kills",
        "kda",
        "cs_per_min",
        "dmg_per_min",
        "gold_per_min",
        "kp",
        "performance_index",
    ]
    export_mid = mid_df.rename(
        columns={
            "kills": "player_kills",
            "deaths": "player_deaths",
            "assists": "player_assists",
            "cs": "total_minions_killed",
            "gold": "gold_earned",
        }
    )
    export_mid[exported_mid_cols].to_csv(PROCESSED_DIR / "mid_players_2024.csv", index=False)
    patch_summary.to_csv(PROCESSED_DIR / "player_patch_summary.csv", index=False)
    tests.to_csv(PROCESSED_DIR / "hypothesis_tests.csv", index=False)
    overall.to_csv(PROCESSED_DIR / "overall_player_summary.csv", index=False)


def make_figures(mid_df: pd.DataFrame, patch_summary: pd.DataFrame) -> None:
    """Create milestone figures for the report."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: 21-day rolling performance.
    plt.figure(figsize=(12, 7))
    for player in PLAYERS:
        sub = mid_df[mid_df["player_name"].eq(player)].sort_values("date").copy()
        daily = sub.set_index("date")["performance_index"].rolling("21D", min_periods=5).mean()
        plt.plot(daily.index, daily.values, linewidth=2, label=player)
    for patch, patch_date in PATCH_DATES.items():
        plt.axvline(patch_date, linestyle="--", alpha=0.5)
        plt.text(patch_date, 0.72, patch, rotation=90, va="bottom", ha="left")
    plt.title("21-day rolling performance index for Faker and benchmark mid laners (2024)")
    plt.xlabel("Date")
    plt.ylabel("Rolling performance index")
    plt.legend(frameon=False, ncols=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_performance_2024.png", dpi=200)
    plt.close()

    # Figure 2: average absolute change by player.
    avg_abs = (
        patch_summary.groupby("player_name")["abs_delta_perf"].mean().sort_values().rename("mean_abs_change")
    )
    plt.figure(figsize=(10, 6))
    ax = avg_abs.plot(kind="bar")
    ax.set_title("Average absolute change in performance index across patch windows")
    ax.set_xlabel("Player")
    ax.set_ylabel("Mean |post - pre| performance index")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "avg_abs_patch_change.png", dpi=200)
    plt.close()

    # Figure 3: diversity vs stability.
    plt.figure(figsize=(8, 6))
    plt.scatter(
        patch_summary["pre_diversity_entropy"],
        patch_summary["abs_delta_perf"],
        s=80,
    )
    for _, row in patch_summary.iterrows():
        plt.annotate(
            f"{row['player_name']} {row['patch']}",
            (row["pre_diversity_entropy"], row["abs_delta_perf"]),
            fontsize=8,
            alpha=0.8,
        )
    plt.title("Champion-pool diversity vs post-patch stability")
    plt.xlabel("Pre-patch champion entropy")
    plt.ylabel("Absolute performance change")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "diversity_vs_stability.png", dpi=200)
    plt.close()

    # Figure 4: new champion rate after patches.
    new_rates = patch_summary.groupby("player_name")["new_champion_rate"].mean().sort_values()
    plt.figure(figsize=(10, 6))
    ax = new_rates.plot(kind="bar")
    ax.set_title("Average share of post-patch games on champions not used in the pre-patch window")
    ax.set_xlabel("Player")
    ax.set_ylabel("New champion rate")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "new_champion_rate.png", dpi=200)
    plt.close()

    # Figure 5: style shares for Faker vs peers.
    style_share = (
        mid_df.assign(group=np.where(mid_df["player_name"].eq("Faker"), "Faker", "Peer average"))
        .groupby(["group", "primary_tag"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = style_share.groupby("group")["count"].transform("sum")
    style_share["share"] = style_share["count"] / totals
    pivot = style_share.pivot(index="primary_tag", columns="group", values="share").fillna(0)
    pivot = pivot.sort_values("Faker", ascending=False)
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Champion style distribution: Faker versus peer pool")
    ax.set_xlabel("Champion style")
    ax.set_ylabel("Share of matches")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "style_share_faker_vs_peers.png", dpi=200)
    plt.close()

    # Figure 6: heatmap of post-pre changes.
    heat = patch_summary.pivot(index="player_name", columns="patch", values="delta_perf")
    plt.figure(figsize=(7, 5))
    plt.imshow(heat.values, aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat.index)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Post - pre performance change")
    plt.title("Patch-to-patch change in performance index")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "patch_delta_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meta, stats = load_raw_data()
    champion_tags = load_champion_tags()
    mid_df = prepare_match_level_data(meta, stats)
    mid_df = enrich_with_champion_metadata(mid_df, champion_tags)

    patch_summary = player_patch_summary(mid_df)
    overall = overall_player_summary(mid_df)
    tests = run_hypothesis_tests(patch_summary)

    save_outputs(mid_df, patch_summary, tests, overall)
    make_figures(mid_df, patch_summary)

    print("Processed rows:", len(mid_df))
    print("Patch summary rows:", len(patch_summary))
    print("Done. Outputs written to:", PROJECT_ROOT)


if __name__ == "__main__":
    main()
