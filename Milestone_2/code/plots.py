"""Milestone 2 figures for the ownership-overlap graph.

All figures use a single sequential blue ramp for magnitude, two fixed
categorical slots (blue, orange) where two series must be told apart, and
text-colour tokens for every label. Nothing is encoded by colour alone: cells
carry values, edges carry widths, and every two-series plot has a legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Palette (validated: scripts/validate_palette.js, light surface) -------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SEQ_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _strip(ax, keep=("left", "bottom")):
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)
        sp.set_linewidth(0.8)
    ax.tick_params(length=0)


def _heatmap(ax, M: np.ndarray, labels_x, labels_y, vmax, fmt="{:.2f}", skip_diag=True, skip_zero=False,
             fontsize=8):
    im = ax.imshow(M, cmap=CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels_x)), labels_x, rotation=45, ha="right", color=INK)
    ax.set_yticks(range(len(labels_y)), labels_y, color=INK)
    _strip(ax, keep=())
    ax.set_xticks(np.arange(-0.5, len(labels_x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_y), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if skip_diag and M.shape[0] == M.shape[1] and i == j:
                continue
            v = M[i, j]
            if skip_zero and v == 0:
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="white" if v > 0.55 * vmax else INK_2)
    return im


# ----------------------------------------------------------------------------
# Figure 1: Jaccard adjacency before / after passive filtering
# ----------------------------------------------------------------------------

def fig_adjacency_pair(A_all: pd.DataFrame, A_active: pd.DataFrame | None, path: Path, period: str) -> None:
    panels = [("All institutions", A_all)]
    if A_active is not None:
        panels.append(("Passive indexers removed", A_active))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels) + 0.6, 5.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    labels = [str(c) for c in A_all.columns]
    for ax, (title, A) in zip(axes, panels):
        im = _heatmap(ax, A.to_numpy(), labels, labels, vmax=1.0)
        if len(panels) > 1:
            ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
    cbar.set_label("Jaccard overlap of holder sets", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Figure 1. Ownership-overlap adjacency A_own\n13F report period {period} (pilot sample)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: institution x stock incidence (the extracted 13F records)
# ----------------------------------------------------------------------------

def fig_incidence(weighted: pd.DataFrame, names: dict, passive_ciks: set, path: Path, period: str) -> None:
    W = weighted.copy()
    order = (W > 0).sum(axis=1).sort_values(ascending=False, kind="stable").index
    W = W.loc[order]
    ylabels = []
    for cik in W.index:
        tag = "  (passive)" if cik in passive_ciks else ""
        ylabels.append(f"{names[cik].title()}{tag}")
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    vmax = float(np.ceil(W.to_numpy().max()))
    im = _heatmap(ax, W.to_numpy(), [str(c) for c in W.columns], ylabels, vmax=vmax,
                  fmt="{:.1f}", skip_diag=False, skip_zero=True)
    ax.set_title("Institution x stock incidence, weight = % of shares outstanding held",
                 loc="left", color=INK, fontsize=11, pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% of shares outstanding", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Figure 2. Extracted 13F holding records, report period {period}\n"
                 f"Top-5 holders per stock; GOOG and GOOGL kept as separate nodes",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: node-link view of the stock graph (circular layout, no networkx)
# ----------------------------------------------------------------------------

def _circular_positions(n: int, r: float = 1.0) -> np.ndarray:
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return np.c_[r * np.cos(ang), r * np.sin(ang)]


def _draw_graph(ax, A: pd.DataFrame, binary: pd.DataFrame, title: str) -> None:
    labels = [str(c) for c in A.columns]
    n = len(labels)
    pos = _circular_positions(n)
    M = A.to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            w = M[i, j]
            if w <= 0:
                continue
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=CMAP(0.25 + 0.75 * w), linewidth=1.0 + 5.0 * w,
                    solid_capstyle="round", zorder=1)
    deg = (M > 0).sum(axis=1)
    holders = binary.sum(axis=0).reindex(A.columns).to_numpy()
    for k in range(n):
        face = SERIES_1 if deg[k] > 0 else SURFACE
        ax.scatter(pos[k, 0], pos[k, 1], s=1500, color=face, edgecolor=SERIES_1 if deg[k] == 0 else SURFACE,
                   linewidth=2, zorder=3)
        ax.text(pos[k, 0], pos[k, 1], labels[k], ha="center", va="center", fontsize=9.5,
                color="white" if deg[k] > 0 else INK, zorder=4, fontweight="bold")
        # holder count just outside the node, pushed away from the centre
        x, y = pos[k]
        ha = "center" if abs(x) < 0.3 else ("left" if x > 0 else "right")
        ax.text(x * 1.22 + np.sign(x) * 0.08, y * 1.22, f"{holders[k]} holders", ha=ha, va="center",
                fontsize=8, color=MUTED, zorder=4)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.45, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", color=INK, fontsize=11)


def fig_network(results: dict, path: Path, period: str, with_active: bool = True) -> None:
    panels = [("all", "All institutions")]
    if with_active:
        panels.append(("active", "Passive indexers removed"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 5.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, title) in zip(axes, panels):
        m = results[name]["matrices"]
        g = results[name]["jaccard"]
        head = f"{title}\n" if with_active else ""
        _draw_graph(ax, m["jaccard"], m["binary"],
                    f"{head}{g['n_edges']} of {g['n_possible_edges']} edges, density {g['density']:.2f}")
    # weight key (single hue, so a width/shade key rather than a series legend)
    handles = [Line2D([0], [0], color=CMAP(0.25 + 0.75 * w), linewidth=1.0 + 5.0 * w, label=f"Jaccard {w:.2f}")
               for w in (0.25, 0.5, 1.0)]
    any_isolated = any(results[name]["jaccard"]["isolated_nodes"] for name, _ in panels)
    if any_isolated:
        handles.append(Line2D([0], [0], marker="o", color=SURFACE, markeredgecolor=SERIES_1, markersize=10,
                              markeredgewidth=2, linewidth=0, label="isolated stock"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, labelcolor=INK_2)
    fig.suptitle(f"Figure 3. Stock ownership-overlap graph, report period {period}\n"
                 "Edge width and shade = Jaccard overlap of holder sets",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: binary vs weighted Jaccard, per stock pair (active graph)
# ----------------------------------------------------------------------------

def fig_edge_weights(edges: pd.DataFrame, path: Path, title_suffix: str) -> None:
    e = edges.sort_values(["weighted_jaccard", "jaccard"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(e) + 1.8), constrained_layout=True)
    ax.hlines(y, e["jaccard"], e["weighted_jaccard"], color=GRID, linewidth=2, zorder=1)
    ax.scatter(e["jaccard"], y, s=70, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Binary Jaccard (holder sets)")
    ax.scatter(e["weighted_jaccard"], y, s=70, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Weighted Jaccard (% shares outstanding)")
    ax.set_yticks(y, [f"{a} - {b}" for a, b in zip(e["stock_a"], e["stock_b"])], color=INK)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("Edge weight")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _strip(ax, keep=("bottom",))
    # Direct-label only the extremes
    top = e.iloc[-1]
    ax.annotate(f"{top['weighted_jaccard']:.2f}", (top["weighted_jaccard"], len(e) - 1),
                xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    ax.set_title(f"Figure 4. Edge weight per stock pair, {title_suffix}", loc="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: filing-lag timeline (the look-ahead control)
# ----------------------------------------------------------------------------

def fig_filing_timeline(holdings: pd.DataFrame, path: Path) -> None:
    h = (holdings.groupby(["owner_cik", "owner_name"], as_index=False)
         .agg(filing_date=("filing_date", "max"), available_from=("available_from", "max"),
              n_stocks=("node", "nunique")))
    h = h.sort_values(["filing_date", "owner_name"]).reset_index(drop=True)
    period_end = holdings["report_period"].max()
    deadline = holdings["statutory_deadline"].max()
    y = np.arange(len(h))

    fig, ax = plt.subplots(figsize=(10, 0.38 * len(h) + 2.0), constrained_layout=True)
    ax.axvline(period_end, color=AXIS, linewidth=1)
    ax.axvline(deadline, color=INK_2, linewidth=1)
    ax.text(period_end, len(h) - 0.3, "  quarter end\n  (holdings as of)", va="bottom", ha="left",
            fontsize=8, color=MUTED)
    ax.text(deadline, len(h) - 0.3, "  statutory deadline\n  (quarter end + 45 days)", va="bottom",
            ha="left", fontsize=8, color=INK_2)
    ax.hlines(y, period_end, h["available_from"], color=GRID, linewidth=2, zorder=1)
    ax.scatter(h["filing_date"], y, s=70, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Form 13F filed (public)")
    ax.scatter(h["available_from"], y, s=70, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=3,
               marker="D", label="First date usable as a graph prior")
    ax.set_yticks(y, [f"{n.title()}  ({k} stock{'s' if k != 1 else ''})"
                      for n, k in zip(h["owner_name"], h["n_stocks"])], color=INK)
    ax.set_ylim(-0.8, len(h) + 1.4)
    ax.set_xlim(period_end - pd.Timedelta(days=4), deadline + pd.Timedelta(days=8))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax.grid(axis="x", which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _strip(ax, keep=("bottom",))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.legend(loc="upper center", ncol=2, frameon=False, labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Figure 5. Filing-lag alignment\nQ4 2025 holdings only enter the model after they are public",
                 loc="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6: the normalised prior that enters the DGT attention head
# ----------------------------------------------------------------------------

def fig_prior(P_active: pd.DataFrame | None, P_all: pd.DataFrame, path: Path) -> None:
    panels = [("All institutions", P_all)]
    if P_active is not None:
        panels.append(("Passive indexers removed", P_active))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels) + 0.6, 5.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    labels = [str(c) for c in P_all.columns]
    vmax = 1.0
    for ax, (title, P) in zip(axes, panels):
        im = _heatmap(ax, P.to_numpy(), labels, labels, vmax=vmax, skip_diag=False)
        if len(panels) > 1:
            ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
    cbar.set_label("Attention prior weight (each row sums to 1)", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle("Figure 6. Row-normalised prior D^-1 (A_own + I)\npassed to the differential graph attention head",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------

def make_all_figures(holdings: pd.DataFrame, results: dict, stats: dict, figdir: Path) -> list[Path]:
    from build_ownership_graph import edge_list

    period = stats["report_periods"][-1]
    names = {int(k): v for k, v in stats["institutions"].items()}
    passive_ciks = set(stats.get("passive_ciks_used", []))
    filtered = stats["passive_records_removed"] > 0
    m_all, m_act = results["all"]["matrices"], results["active"]["matrices"]
    # When no passive filter is applied, "active" equals "all"; plot once.
    m_graph = m_act if filtered else m_all
    suffix = "passive indexers removed" if filtered else "all institutions in the CSV"
    paths = []

    p = figdir / "fig1_adjacency_jaccard_before_after.png"
    fig_adjacency_pair(m_all["jaccard"], m_act["jaccard"] if filtered else None, p, period); paths.append(p)

    p = figdir / "fig2_incidence_institution_x_stock.png"
    fig_incidence(m_all["weighted"], names, passive_ciks, p, period); paths.append(p)

    p = figdir / "fig3_ownership_network.png"
    fig_network(results, p, period, with_active=filtered); paths.append(p)

    edges = edge_list(m_graph["jaccard"], m_graph["counts"], m_graph["weighted_jaccard"], m_graph["binary"], names)
    p = figdir / "fig4_edge_weights_binary_vs_weighted.png"
    fig_edge_weights(edges, p, suffix); paths.append(p)

    p = figdir / "fig5_filing_lag_timeline.png"
    fig_filing_timeline(holdings, p); paths.append(p)

    p = figdir / "fig6_dgt_prior_rownorm.png"
    fig_prior(m_act["prior"] if filtered else None, m_all["prior"], p); paths.append(p)
    return paths
