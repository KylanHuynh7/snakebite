"""Phase 1: pull nflverse weekly stats, build full-PPR baseline rankings with VOR.

VOR (value over replacement) = player's PPR points per game minus the PPG of
the "replacement level" player at his position -- the best player you could
grab off waivers in a 10-team league. Replacement ranks reflect a standard
ESPN 10-team lineup (1QB / 2RB / 2WR / 1TE / 1FLEX) plus typical bench depth.
"""

from pathlib import Path

import nflreadpy as nfl
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

SEASONS = [2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]

# 10-team replacement level: starters league-wide (QB 10, TE 10, RB 20+flex,
# WR 20+flex) -- flex skews WR/RB, so replacement is the ~13th QB/TE and the
# ~28th RB/WR once bye-week streaming is accounted for.
REPLACEMENT_RANK = {"QB": 13, "RB": 28, "WR": 28, "TE": 13}

MIN_GAMES = 6  # PPG over fewer games is too noisy to rank on


def load_weekly() -> pl.DataFrame:
    cache = DATA / "weekly_stats.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    df = nfl.load_player_stats(SEASONS)
    df = df.filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(POSITIONS)
    )
    keep = [
        "player_id", "player_display_name", "position", "team", "season", "week",
        "fantasy_points_ppr", "fantasy_points",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "carries", "rushing_yards", "rushing_tds",
        "attempts", "passing_yards", "passing_tds", "passing_interceptions",
        "target_share", "air_yards_share",
    ]
    df = df.select([c for c in keep if c in df.columns])
    df.write_parquet(cache)
    return df


def season_totals(weekly: pl.DataFrame) -> pl.DataFrame:
    return (
        weekly.group_by(["player_id", "player_display_name", "position", "season"])
        .agg(
            games=pl.col("fantasy_points_ppr").count(),
            ppr_total=pl.col("fantasy_points_ppr").sum().round(1),
            ppr_pg=pl.col("fantasy_points_ppr").mean().round(2),
            team=pl.col("team").last(),
            targets_pg=pl.col("targets").mean().round(1),
            carries_pg=pl.col("carries").mean().round(1),
            tgt_share=pl.col("target_share").mean().round(3),
        )
        .sort("ppr_total", descending=True)
    )


def add_vor(season_df: pl.DataFrame) -> pl.DataFrame:
    qual = season_df.filter(pl.col("games") >= MIN_GAMES)
    frames = []
    for pos, rank in REPLACEMENT_RANK.items():
        pos_df = qual.filter(pl.col("position") == pos).sort(
            "ppr_pg", descending=True
        )
        repl_ppg = pos_df["ppr_pg"][min(rank, pos_df.height) - 1]
        pos_df = pos_df.with_columns(
            vor_pg=(pl.col("ppr_pg") - repl_ppg).round(2),
            pos_rank=pl.arange(1, pos_df.height + 1),
        )
        frames.append(pos_df)
    out = pl.concat(frames).sort("vor_pg", descending=True)
    return out.with_columns(overall_rank=pl.arange(1, out.height + 1))


def main() -> None:
    weekly = load_weekly()
    totals = season_totals(weekly)
    totals.write_parquet(DATA / "season_totals.parquet")

    ranked = add_vor(totals.filter(pl.col("season") == 2025))
    ranked.write_csv(OUT / "baseline_rankings_2025.csv")

    cols = ["overall_rank", "player_display_name", "position", "team",
            "games", "ppr_pg", "vor_pg", "pos_rank"]
    with pl.Config(tbl_rows=40, tbl_cols=-1, fmt_str_lengths=30):
        print("=== Top 40 by VOR per game (2025 season, full PPR) ===")
        print(ranked.select(cols).head(40))
        for pos in POSITIONS:
            print(f"\n=== Top 12 {pos} ===")
            print(ranked.filter(pl.col("position") == pos).select(cols).head(12))


if __name__ == "__main__":
    main()
