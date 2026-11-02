"""Figures for RESULTS.md.

Every number below was copied from the named script's own output, run on the
committed data (the league is anonymised, so the aggregates are unchanged).
They are pasted here rather than recomputed because the evidence scripts take
minutes each; to check one, run the script named next to it.

usage:  python scripts/results_figures.py       (needs requirements-docs.txt)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, DIM, GRID = "#172019", "#5f6b62", "#dde2d9"
GREEN, GOLD, RED, SLATE = "#1d7a4b", "#e0a21b", "#d4584a", "#8a948d"
plt.rcParams.update({
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": DIM, "xtick.color": DIM,
    "ytick.color": DIM, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# --- 1. FFC vs ESPN as the prior ------------------------------- replay_market.py
seasons = ["2023", "2024", "2025", "pooled"]
ffc = [0.0799, 0.1144, 0.1021, 0.0998]
espn = [0.0901, 0.1232, 0.1238, 0.1133]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
x = range(len(seasons))
ax.bar([i - .2 for i in x], ffc, .38, color=GREEN, label="FFC consensus")
ax.bar([i + .2 for i in x], espn, .38, color=SLATE, label="ESPN-shaped")
for i, (a, b) in enumerate(zip(ffc, espn)):
    ax.text(i - .2, a + .002, f"{a:.3f}", ha="center", fontsize=8, color=DIM)
    ax.text(i + .2, b + .002, f"{b:.3f}", ha="center", fontsize=8, color=DIM)
ax.set_xticks(list(x), seasons)
ax.set_ylabel("Brier score (lower is better)")
ax.set_ylim(0, .15)
ax.set_title("The league drafts on ESPN, but ESPN's market predicts it worse",
             loc="left", fontsize=11, fontweight="bold")
ax.legend(frameon=False, loc="upper left", ncol=2)
save(fig, "brier_markets.png")

# --- 2. The model ladder ------------------------------------------ replay_brier.py
steps = [("A  raw market spread", .1200, SLATE),
         ("B  league spread multiplier", .1003, GREEN),
         ("B2  rank-dependent curve", .1008, GREEN),
         ("B3  curve + chalk drift  (shipped)", .0998, GREEN),
         ("Cm  simulate the picks between", .1011, RED),
         ("C  + roster need and manager leans", .1033, RED)]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ys = list(range(len(steps)))[::-1]
for y, (lab, v, c) in zip(ys, steps):
    ax.barh(y, v, color=c, height=.62)
    ax.text(v + .0015, y, f"{v:.4f}", va="center", fontsize=9, color=DIM)
ax.set_yticks(ys, [s[0] for s in steps])
ax.set_xlim(.08, .125)
ax.set_xlabel("pooled Brier, leave-one-season-out (lower is better)")
ax.set_title("Calibration helped; modelling the opponents did not",
             loc="left", fontsize=11, fontweight="bold")
ax.grid(axis="y", visible=False)
save(fig, "brier_ladder.png")

# --- 3. Opponent weight grid ------------------------------------ need_redesign.py
need = ["0.00", "0.50", "1.00"]
grid = [[.10017, .09966, .10116],      # run 0.00
        [.10323, .10123, .10199],      # run 0.50
        [.10977, .10716, .10667]]      # run 1.00
shipped = .09976
fig, ax = plt.subplots(figsize=(4.6, 3.3))
im = ax.imshow(grid, cmap="RdYlGn_r", vmin=.0995, vmax=.1100)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{grid[i][j]:.4f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(3), need)
ax.set_yticks(range(3), need)
ax.set_xlabel("roster-need weight")
ax.set_ylabel("run weight")
ax.grid(False)
ax.set_title(f"Every opponent term is a tie or worse\n(shipped model: {shipped:.4f})",
             loc="left", fontsize=10.5, fontweight="bold")
save(fig, "opponent_grid.png")

# --- 4. Do runs continue? ---------------------------------------- need_redesign.py
runs = [("2 of last 3 were RB", .322, .350, 90),
        ("3 of last 3 were WR", .458, .424, 24),
        ("4 of last 5 were WR", .382, .424, 34)]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for i, (lab, p, base, n) in enumerate(runs[::-1]):
    ax.plot([base, p], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter([base], [i], color=SLATE, s=45, zorder=2)
    ax.scatter([p], [i], color=GREEN, s=45, zorder=3)
    ax.text(.49, i, f"lift {p / base:.2f}   n={n}", va="center", fontsize=9, color=DIM)
ax.set_yticks(range(3), [r[0] for r in runs[::-1]])
ax.set_xlim(.28, .56)
ax.set_xlabel("P(next pick is that position)   grey = base rate, green = after the run")
ax.set_title("Runs do not continue in this league", loc="left", fontsize=11,
             fontweight="bold")
ax.grid(axis="y", visible=False)
save(fig, "runs.png")

# --- 5. Disagreement test -------------------------------------- board_backtest.py
labels = ["2024", "2025"]
likes = [52.0, 61.9]
fades = [39.0, 33.3]
fig, ax = plt.subplots(figsize=(5.4, 3.1))
x = range(2)
ax.bar([i - .2 for i in x], likes, .38, color=GREEN, label="board LIKES (24+ ranks)")
ax.bar([i + .2 for i in x], fades, .38, color=RED, label="board FADES (24+ ranks)")
for i, (a, b) in enumerate(zip(likes, fades)):
    ax.text(i - .2, a + 1, f"{a:.0f}%", ha="center", fontsize=9, color=DIM)
    ax.text(i + .2, b + 1, f"{b:.0f}%", ha="center", fontsize=9, color=DIM)
ax.axhline(50, color=INK, lw=.8, ls="--")
ax.text(1.46, 50.8, "coin flip", ha="right", fontsize=8, color=DIM)
ax.set_xticks(list(x), labels)
ax.set_ylabel("% beating their ADP price")
ax.set_ylim(0, 75)
ax.set_title("Only large disagreements with the market paid off", loc="left",
             fontsize=11, fontweight="bold")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
save(fig, "disagreement.png")

# --- 6. Quantile coverage -------------------------------- calibrate_quantiles.py
req = [.06, .08, .10, .12, .14, .16, .20]
obs = [13.6, 16.5, 18.4, 20.9, 22.8, 25.5, 29.8]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.plot([0, 35], [0, 35], color=INK, lw=.8, ls="--")
ax.plot([r * 100 for r in req], obs, "o-", color=RED, lw=2)
ax.scatter([12], [20.9], s=120, facecolor="none", edgecolor=GREEN, lw=2, zorder=5)
ax.annotate("ask for 12%\nto get a true 20%", (12, 20.9), (15.5, 11.5), fontsize=8.5,
            color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN))
ax.set_xlim(4, 22)
ax.set_ylim(4, 33)
ax.set_xlabel("quantile requested from the model (%)")
ax.set_ylabel("share of players actually below it (%)")
ax.set_title('The "floor" was a 30th percentile', loc="left", fontsize=11,
             fontweight="bold")
save(fig, "floor_calibration.png")
