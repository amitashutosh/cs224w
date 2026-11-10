# Milestone 2 submission: pilot 13F ingestion and ownership-overlap graph

Everything in this folder is computed only from the 40 rows in
`1_input_13f_holdings.csv` (top-5 institutional holders of 8 stocks, SEC Form
13F, report period 31 Dec 2025). No institutions are filtered out and GOOG and
GOOGL are treated as separate stocks.

## CSV files

| File | What it is |
|---|---|
| `1_input_13f_holdings.csv` | The raw input: one row per (stock, institution) holding. |
| `2_extracted_holding_records.csv` | Deliverable 1. The same records typed and cleaned, with the date each holding becomes usable (later of filing date and quarter end + 45 days). |
| `3_institution_x_stock_incidence.csv` | 12 x 8 matrix, 1 if the institution holds the stock. |
| `4_shared_holder_counts.csv` | 8 x 8 matrix, number of institutions two stocks have in common. |
| `5_ownership_adjacency_jaccard.csv` | Deliverable 2. 8 x 8 Jaccard overlap: shared holders / union of holders. Diagonal is 0. |
| `6_ownership_adjacency_weighted_jaccard.csv` | Same, weighted by % of shares outstanding held. |
| `7_ownership_adjacency_normalized_prior.csv` | Row-normalised (A + I): each row sums to 1. This is the form handed to the DGT attention head. |
| `8_stock_pair_edge_list.csv` | All 28 stock pairs with shared-holder count, both Jaccard scores, and the names of the shared holders. |

## Plots

| File | What it shows |
|---|---|
| `plots/plot1_adjacency_heatmap.png` | The Jaccard adjacency matrix (file 5). |
| `plots/plot2_institution_x_stock_incidence.png` | Which institution holds which stock, shaded by % of shares outstanding. |
| `plots/plot3_ownership_network.png` | The stock graph; edge width and shade = Jaccard weight. |
| `plots/plot4_edge_weights.png` | Binary vs weighted Jaccard for every stock pair. |
| `plots/plot5_filing_lag_timeline.png` | Filing date vs first usable date per institution (the 45-day look-ahead rule). |
| `plots/plot6_normalized_prior.png` | The row-normalised prior (file 7). |

## Key numbers

| Property | Value |
|---|---|
| Holding records | 40 |
| Stocks | 8 |
| Institutions | 12 |
| Edges | 28 of 28 (every pair shares at least one holder) |
| Jaccard range | 0.25 (TSLA pairs) to 1.00 (AMZN-MSFT, AMZN-NVDA, MSFT-NVDA) |
| Holdings usable from | 14 Feb 2026, or 17 Feb 2026 for the five late filers |

The graph is fully connected because State Street and Geode appear in every
stock's top 5. Filtering such passive index-fund managers, as the proposal
describes, is deferred to the full project with complete 13F data.

## Reproduce

```bash
python3 code/build_ownership_graph.py --input 1_input_13f_holdings.csv --outdir out --no-passive-filter
```

## Handoff to Milestone 3 (DGT forward pass)

Milestone 3 reproduces the differential graph attention layer and runs a
forward pass to confirm tensor shapes. From this folder you need one thing:
the ownership prior `A_own`.

- **File:** `7_ownership_adjacency_normalized_prior.csv`.
- **Shape:** `(8, 8)`, row-normalised so each row sums to 1, with a
  self-loop on the diagonal.
- **Node order (row = column index):** AAPL, AMZN, GOOG, GOOGL, META, MSFT,
  NVDA, TSLA. The price tensor for the forward pass must use the same order.
- **Where it plugs in:** the `A` argument of the DGT `MultiheadDiffAttn`
  forward, which multiplies the first softmax map elementwise: `softmax(Q1 K1ᵀ) ⊙ A`.
  Broadcast it to `(batch, heads, N, N)`, or pass `(N, N)` and let PyTorch
  broadcast. Use `A = I` (identity) for the temporal-only arm of the ablation.

```python
import pandas as pd, torch
P = pd.read_csv("7_ownership_adjacency_normalized_prior.csv", index_col=0)
nodes = list(P.index)                                              # stock order
A_own = torch.tensor(P.to_numpy(), dtype=torch.float32)            # (8, 8)
A = A_own[None, None]                                              # (1, 1, N, N) for broadcast
```

Two things Milestone 3 must supply itself:

1. **Price data.** Milestone 1 (price preprocessing) is still open. For a
   shape check, a random tensor of shape `(batch, T=64, N=8, 1)` is enough.
   For a real run, pull daily closes for the eight tickers above from Yahoo
   Finance and z-score them.
2. **Date alignment.** This prior is from the Q4 2025 filings and is only
   valid for prices on or after 14 Feb 2026 (17 Feb for five late filers,
   see file 2). It cannot be used with the 2014 to 2024 DGT price window;
   that needs historical 13F quarters, which is final-project work.
