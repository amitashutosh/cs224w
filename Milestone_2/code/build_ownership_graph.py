"""Milestone 2: pilot SEC Form 13F ingestion and ownership-overlap graph construction.

Pipeline
--------
1. Load institutional holding records (one row = one institution's position in
   one stock for one report period) and normalise column names / units.
2. Assign each ticker to a stock node. GOOG and GOOGL are kept as separate
   nodes (share-class merging is available but switched off).
3. Enforce the statutory 45-day post-quarter 13F filing lag so holdings are only
   usable for prediction after they are public (no look-ahead).
4. Split holders into active managers and broad passive indexers that would
   trivially connect every constituent.
5. Build the institution x stock incidence matrix (binary and weighted by
   percent of shares outstanding).
6. Compute the stock x stock ownership-overlap adjacency:
       jaccard          : |N(i) & N(j)| / |N(i) | N(j)|
       weighted_jaccard : sum_k min(w_ki, w_kj) / sum_k max(w_ki, w_kj)
7. Produce the normalised adjacency used as the DGT graph prior A_own:
   row-normalised (A + I) so every stock attends to itself and each row is a
   proper attention mask that sums to 1. A symmetric GCN-style normalisation
   is saved as well.
8. Write dataset statistics and the Milestone 2 figures.

Usage
-----
    python code/build_ownership_graph.py \
        --input 1_input_13f_holdings.csv \
        --outdir out --no-passive-filter

The same script is intended to run unchanged on bulk EDGAR 13F data once the
INFOTABLE / SUBMISSION tables are joined into this record schema.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

#: Share-class collapsing. Maps a ticker to the canonical node id it belongs to.
#: Left empty for the milestone: GOOG and GOOGL are kept as separate stock
#: nodes, matching the DGT price dataset where each ticker is its own series.
#: To merge classes, add e.g. {"GOOG": "GOOGL", "GOOGL": "GOOGL"}.
SHARE_CLASS_MAP: dict[str, str] = {}

#: Broad passive indexers / index-fund managers, keyed by SEC CIK. These hold
#: essentially every S&P 500 constituent in cap-weighted proportion and would
#: otherwise connect every pair of stocks. Vanguard and BlackRock are listed for
#: when the bulk data is used; State Street and Geode (Fidelity's index-fund
#: arm) are the ones present in the pilot sample.
PASSIVE_INDEXER_CIKS = {
    102909: "VANGUARD GROUP INC",
    1364742: "BLACKROCK INC",
    93751: "STATE STREET CORP",
    1214717: "GEODE CAPITAL MANAGEMENT, LLC",
}

#: Statutory 13F filing deadline: 45 days after the end of the calendar quarter.
FILING_LAG_DAYS = 45

#: Alternative spellings of the input columns seen across the team's CSV
#: exports. Values given in billions are rescaled to USD.
COLUMN_ALIASES = {
    "reported_position_value_billion_usd": ("reported_position_value_usd", 1e9),
    "market_cap_billion usd": ("market_cap_usd", 1e9),
    "market_cap_billion_usd": ("market_cap_usd", 1e9),
    "implied_market_cap_usd": ("market_cap_usd", 1.0),
}

REQUIRED_COLUMNS = [
    "ticker",
    "company_name",
    "cusip",
    "owner_cik",
    "owner_name",
    "filing_date",
    "report_period",
    "shares_held",
    "reported_position_value_usd",
    "shares_outstanding",
    "pct_shares_outstanding",
]


# ----------------------------------------------------------------------------
# Step 1-4: load, collapse share classes, enforce filing lag, split passive
# ----------------------------------------------------------------------------

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased columns to the canonical schema and rescale units."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for src, (dst, scale) in COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce") * scale
            df = df.drop(columns=[src])
    return df


def load_holdings(path: str | Path) -> pd.DataFrame:
    """Read raw 13F holding records and return a cleaned, typed DataFrame."""
    df = normalise_columns(pd.read_csv(path))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["cusip"] = df["cusip"].astype(str).str.strip().str.zfill(9)
    df["owner_cik"] = df["owner_cik"].astype(int)
    df["owner_name"] = df["owner_name"].astype(str).str.strip()
    df["filing_date"] = pd.to_datetime(df["filing_date"], format="mixed", dayfirst=True)
    df["report_period"] = pd.to_datetime(df["report_period"], format="mixed", dayfirst=True)
    for c in ["shares_held", "reported_position_value_usd", "shares_outstanding", "pct_shares_outstanding"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["ticker", "owner_cik", "shares_held"]].isna().any().any():
        raise ValueError("Blank ticker, owner_cik or shares_held in input")
    return df


def collapse_share_classes(df: pd.DataFrame, share_class_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Assign each row to a stock node, optionally merging share classes.

    With an empty map (the default) every ticker is its own node. When a map
    is given, an institution holding both classes gets its shares and position
    value summed and percent of shares outstanding is recomputed against the
    issuer's total shares outstanding (the same figure is reported for every
    class).
    """
    share_class_map = SHARE_CLASS_MAP if share_class_map is None else share_class_map
    df = df.copy()
    df["node"] = df["ticker"].map(lambda t: share_class_map.get(t, t))

    agg = (
        df.groupby(["node", "owner_cik", "report_period"], as_index=False)
        .agg(
            company_name=("company_name", "first"),
            owner_name=("owner_name", "first"),
            tickers=("ticker", lambda s: "+".join(sorted(set(s)))),
            filing_date=("filing_date", "max"),
            shares_held=("shares_held", "sum"),
            reported_position_value_usd=("reported_position_value_usd", "sum"),
            shares_outstanding=("shares_outstanding", "max"),
        )
    )
    agg["pct_shares_outstanding"] = 100.0 * agg["shares_held"] / agg["shares_outstanding"]
    return agg


def add_availability_date(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the first date on which each holding may be used for prediction.

    We take the later of the actual filing date and the statutory deadline
    (report period + 45 days). Using the deadline rather than the filing date
    alone is deliberately conservative: it keeps the alignment rule identical
    for every filer regardless of how early they filed.
    """
    df = df.copy()
    deadline = df["report_period"] + timedelta(days=FILING_LAG_DAYS)
    df["statutory_deadline"] = deadline
    df["available_from"] = np.maximum(df["filing_date"], deadline)
    return df


def filter_passive_indexers(df: pd.DataFrame, passive_ciks=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split records into (active, passive) by owner CIK."""
    passive_ciks = set(PASSIVE_INDEXER_CIKS if passive_ciks is None else passive_ciks)
    is_passive = df["owner_cik"].isin(passive_ciks)
    return df.loc[~is_passive].copy(), df.loc[is_passive].copy()


# ----------------------------------------------------------------------------
# Step 5: incidence matrices
# ----------------------------------------------------------------------------

def build_incidence(df: pd.DataFrame, nodes: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (binary, weighted) institution x stock incidence matrices.

    Rows are institutions (owner_cik), columns are stock nodes. The weighted
    version uses percent of shares outstanding held, so weights are comparable
    across stocks of very different size.
    """
    weighted = df.pivot_table(
        index="owner_cik", columns="node", values="pct_shares_outstanding", aggfunc="sum", fill_value=0.0
    )
    if nodes is not None:
        weighted = weighted.reindex(columns=nodes, fill_value=0.0)
    binary = (weighted > 0).astype(int)
    return binary, weighted


# ----------------------------------------------------------------------------
# Step 6-7: overlap adjacency and normalisation
# ----------------------------------------------------------------------------

def jaccard_adjacency(binary: pd.DataFrame) -> pd.DataFrame:
    """Stock x stock Jaccard overlap of holder sets. Diagonal is set to 0."""
    B = binary.to_numpy(dtype=float)
    inter = B.T @ B
    counts = B.sum(axis=0)
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(J, 0.0)
    return pd.DataFrame(J, index=binary.columns, columns=binary.columns)


def shared_holder_counts(binary: pd.DataFrame) -> pd.DataFrame:
    """Stock x stock count of common holders (the Jaccard numerator)."""
    B = binary.to_numpy(dtype=float)
    C = B.T @ B
    np.fill_diagonal(C, 0.0)
    return pd.DataFrame(C.astype(int), index=binary.columns, columns=binary.columns)


def weighted_jaccard_adjacency(weighted: pd.DataFrame) -> pd.DataFrame:
    """Stock x stock weighted Jaccard (Ruzicka) overlap. Diagonal is set to 0."""
    W = weighted.to_numpy(dtype=float)
    n = W.shape[1]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            num = np.minimum(W[:, i], W[:, j]).sum()
            den = np.maximum(W[:, i], W[:, j]).sum()
            A[i, j] = A[j, i] = num / den if den > 0 else 0.0
    return pd.DataFrame(A, index=weighted.columns, columns=weighted.columns)


def row_normalize_with_self_loops(A: pd.DataFrame) -> pd.DataFrame:
    """Return D^{-1}(A + I): the attention-mask form of the graph prior."""
    M = A.to_numpy(dtype=float) + np.eye(len(A))
    M = M / M.sum(axis=1, keepdims=True)
    return pd.DataFrame(M, index=A.index, columns=A.columns)


def sym_normalize_with_self_loops(A: pd.DataFrame) -> pd.DataFrame:
    """Return D^{-1/2}(A + I)D^{-1/2}: the GCN-style normalisation."""
    M = A.to_numpy(dtype=float) + np.eye(len(A))
    d = M.sum(axis=1)
    Dm12 = np.diag(1.0 / np.sqrt(d))
    return pd.DataFrame(Dm12 @ M @ Dm12, index=A.index, columns=A.columns)


def edge_list(A: pd.DataFrame, counts: pd.DataFrame, A_w: pd.DataFrame, binary: pd.DataFrame,
              names: dict[int, str]) -> pd.DataFrame:
    """Long-format table of every stock pair with its overlap statistics."""
    rows = []
    cols = list(A.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            shared = binary.index[(binary[a] > 0) & (binary[b] > 0)]
            rows.append({
                "stock_a": a,
                "stock_b": b,
                "shared_holders": int(counts.loc[a, b]),
                "jaccard": float(A.loc[a, b]),
                "weighted_jaccard": float(A_w.loc[a, b]),
                "shared_holder_names": "; ".join(sorted(names[c] for c in shared)),
            })
    return pd.DataFrame(rows).sort_values(["jaccard", "stock_a", "stock_b"], ascending=[False, True, True])


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def graph_stats(A: pd.DataFrame, binary: pd.DataFrame) -> dict:
    """Basic statistics for one adjacency matrix and its incidence matrix."""
    n = len(A)
    upper = A.to_numpy()[np.triu_indices(n, k=1)]
    n_edges = int((upper > 0).sum())
    possible = n * (n - 1) // 2
    deg = (A.to_numpy() > 0).sum(axis=1)
    return {
        "n_stocks": int(n),
        "n_institutions": int(binary.shape[0]),
        "n_holding_records": int(binary.to_numpy().sum()),
        "n_edges": n_edges,
        "n_possible_edges": possible,
        "density": float(n_edges / possible) if possible else 0.0,
        "mean_degree": float(deg.mean()),
        "isolated_nodes": [str(s) for s in A.index[deg == 0]],
        "edge_weight_mean": float(upper[upper > 0].mean()) if n_edges else 0.0,
        "edge_weight_max": float(upper.max()) if n_edges else 0.0,
        "holders_per_stock": {str(k): int(v) for k, v in binary.sum(axis=0).items()},
        "stocks_per_institution": {str(k): int(v) for k, v in binary.sum(axis=1).items()},
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(input_path: Path, outdir: Path, passive_ciks=None, make_figures: bool = True) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    raw = load_holdings(input_path)
    holdings = add_availability_date(collapse_share_classes(raw))
    holdings.to_csv(outdir / "holdings_clean.csv", index=False)

    active, passive = filter_passive_indexers(holdings, passive_ciks)
    nodes = sorted(holdings["node"].unique())
    names = holdings.drop_duplicates("owner_cik").set_index("owner_cik")["owner_name"].to_dict()

    results = {}
    for name, frame in [("all", holdings), ("active", active)]:
        binary, weighted = build_incidence(frame, nodes)
        A_j = jaccard_adjacency(binary)
        A_wj = weighted_jaccard_adjacency(weighted)
        counts = shared_holder_counts(binary)
        A_prior = row_normalize_with_self_loops(A_j)
        A_sym = sym_normalize_with_self_loops(A_j)

        binary.to_csv(outdir / f"incidence_binary_{name}.csv")
        weighted.to_csv(outdir / f"incidence_weighted_{name}.csv")
        A_j.to_csv(outdir / f"adjacency_jaccard_{name}.csv")
        A_wj.to_csv(outdir / f"adjacency_weighted_jaccard_{name}.csv")
        counts.to_csv(outdir / f"shared_holder_counts_{name}.csv")
        A_prior.to_csv(outdir / f"adjacency_prior_rownorm_{name}.csv")
        A_sym.to_csv(outdir / f"adjacency_prior_symnorm_{name}.csv")
        np.save(outdir / f"A_own_{name}.npy", A_prior.to_numpy(dtype=np.float32))
        edge_list(A_j, counts, A_wj, binary, names).to_csv(outdir / f"edge_list_{name}.csv", index=False)

        results[name] = {
            "jaccard": graph_stats(A_j, binary),
            "weighted_jaccard": graph_stats(A_wj, binary),
            "matrices": {
                "binary": binary, "weighted": weighted, "jaccard": A_j,
                "weighted_jaccard": A_wj, "counts": counts, "prior": A_prior,
            },
        }

    report_periods = sorted(holdings["report_period"].dt.strftime("%Y-%m-%d").unique())
    concentration = (
        holdings.groupby("node")["pct_shares_outstanding"].sum().round(3).to_dict()
    )
    stats = {
        "input_file": str(input_path),
        "raw_rows": int(len(raw)),
        "raw_tickers": sorted(raw["ticker"].unique()),
        "nodes_after_share_class_collapse": nodes,
        "share_class_merges": {k: v for k, v in SHARE_CLASS_MAP.items() if k != v},
        "n_institutions": int(holdings["owner_cik"].nunique()),
        "institutions": {int(k): v for k, v in names.items()},
        "report_periods": report_periods,
        "filing_dates": sorted(holdings["filing_date"].dt.strftime("%Y-%m-%d").unique()),
        "filing_lag_days": FILING_LAG_DAYS,
        "statutory_deadline": sorted(holdings["statutory_deadline"].dt.strftime("%Y-%m-%d").unique()),
        "available_from": sorted(holdings["available_from"].dt.strftime("%Y-%m-%d").unique()),
        "passive_ciks_used": sorted(int(c) for c in (PASSIVE_INDEXER_CIKS if passive_ciks is None else passive_ciks)),
        "passive_indexers_removed": sorted(passive["owner_name"].unique()),
        "passive_records_removed": int(len(passive)),
        "top5_pct_shares_outstanding_per_stock": concentration,
        "graphs": {
            name: {k: v for k, v in r.items() if k != "matrices"} for name, r in results.items()
        },
    }
    with open(outdir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if make_figures:
        from plots import make_all_figures  # local module, same directory
        make_all_figures(holdings, results, stats, figdir)

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="CSV of 13F holding records")
    p.add_argument("--outdir", default=Path("out"), type=Path)
    p.add_argument(
        "--passive-ciks", nargs="*", type=int, default=None,
        help="Owner CIKs to treat as passive indexers (default: built-in list)",
    )
    p.add_argument("--no-passive-filter", action="store_true",
                   help="Use every institution in the CSV as-is; no passive-indexer list is applied")
    p.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = p.parse_args()
    passive = [] if args.no_passive_filter else args.passive_ciks
    stats = run(args.input, args.outdir, passive, make_figures=not args.no_figures)

    print(f"Raw rows: {stats['raw_rows']}  tickers: {stats['raw_tickers']}")
    print(f"Stocks (after share-class collapse): {stats['nodes_after_share_class_collapse']}")
    print(f"Institutions: {stats['n_institutions']}")
    print(f"Report periods: {stats['report_periods']}  ->  usable from {stats['available_from']}")
    print(f"Passive indexers removed: {stats['passive_indexers_removed']} ({stats['passive_records_removed']} records)")
    for name, g in stats["graphs"].items():
        j = g["jaccard"]
        print(
            f"[{name:6s}] institutions={j['n_institutions']:2d} edges={j['n_edges']:2d}/{j['n_possible_edges']} "
            f"density={j['density']:.2f} mean_deg={j['mean_degree']:.2f} isolated={j['isolated_nodes']}"
        )
    print(f"Outputs written to {args.outdir}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
