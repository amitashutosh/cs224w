"""
src/data/parse_13f_pilot.py

Pipeline for Step 2:
1. Discovers and extracts SEC Form 13F bulk dataset for Q1 2024.
2. Ingests SUBMISSION.tsv with dynamic column detection.
3. Maps institutional positions to the exact 472 S&P 500 tickers from Step 1.
4. Filters passive index replication noise.
5. Computes Jaccard-normalized institutional ownership overlap matrix A_own (472 x 472).
6. Exports A_own tensor and prints sample matrix for milestone report.
"""

import io
import json
import os
import re
from pathlib import Path
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

# Automatically ensure the repository root (cs224w/) is in Python's import search path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.price_loader import SP500PricePipeline

SEC_13F_CANDIDATE_URLS = [
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2024-31may2024_form13f.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/form-13f-data-sets/01mar2024-31may2024_form13f.zip",
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2023-29feb2024_form13f.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/form-13f-data-sets/01dec2023-29feb2024_form13f.zip",
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2024q1_form13f.zip",
    "https://www.sec.gov/files/dera/data/form-13f-data-sets/2024q1_form13f.zip",
]
SEC_13F_LANDING_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
SEC_TICKERS_URL      = "https://www.sec.gov/files/company_tickers.json"

CACHE_DIR = "data/sec_13f_2024q1"
TICKERS_CACHE_FILE = "data/company_tickers.json"
OUTPUT_MATRIX_PATH = "data/A_own_pilot.pt"

SEC_HEADERS = {
    "User-Agent": "StanfordUniversity StudentResearch cs224w-project@stanford.edu",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# Regex patterns identifying passive index replicators to isolate active institutional management
PASSIVE_MANAGER_NAME_PATTERNS = [
    r"VANGUARD",
    r"BLACKROCK",
    r"STATE\s+STREET",
    r"SSGA",
    r"GEODE\s+CAPITAL",
    r"NORTHERN\s+TRUST",
]

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def fetch_url_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=120) as resp:
        return resp.read()


def discover_live_sec_13f_urls() -> list:
    try:
        print(f"Checking SEC landing page for active datasets: {SEC_13F_LANDING_PAGE}")
        html_bytes = fetch_url_bytes(SEC_13F_LANDING_PAGE)
        html = html_bytes.decode("utf-8", errors="ignore")
        raw_zips = re.findall(r'href=[\'"]([^\'"]+\.zip[^\'"]*)[\'"]', html, re.IGNORECASE)
        discovered = []
        for z in raw_zips:
            full_url = urllib.parse.urljoin("https://www.sec.gov", z.strip())
            if "form-13f" in full_url.lower() or "13f" in full_url.lower():
                discovered.append(full_url)
        return discovered
    except Exception as e:
        print(f"Dynamic discovery note: {e}")
        return []


def find_extracted_files(target_dir: str):
    if not os.path.exists(target_dir):
        return None, None
    files_map = {f.upper(): f for f in os.listdir(target_dir)}
    info_file = (
        files_map.get("INFOTABLE.TSV")
        or files_map.get("INFOTABLE.TXT")
        or files_map.get("INFOTABLE.CSV")
    )
    sub_file = (
        files_map.get("SUBMISSION.TSV")
        or files_map.get("SUBMISSION.TXT")
        or files_map.get("SUBMISSION.CSV")
    )
    if info_file and sub_file:
        return os.path.join(target_dir, info_file), os.path.join(target_dir, sub_file)
    return None, None


def download_and_extract_sec_data(target_dir: str = CACHE_DIR):
    os.makedirs(target_dir, exist_ok=True)
    info_path, sub_path = find_extracted_files(target_dir)

    if info_path and sub_path:
        print(f"Found local SEC 13F files in: {target_dir}")
        return info_path, sub_path

    print("Downloading SEC 13F Q1 2024 bulk archive from SEC EDGAR...")
    zip_bytes = None
    candidate_urls = list(SEC_13F_CANDIDATE_URLS)
    discovered_urls = discover_live_sec_13f_urls()
    for d_url in discovered_urls:
        if d_url not in candidate_urls:
            candidate_urls.append(d_url)

    for url in candidate_urls:
        try:
            print(f"Attempting download from: {url}")
            zip_bytes = fetch_url_bytes(url)
            print("Download successful.")
            break
        except urllib.error.HTTPError as e:
            print(f"Failed with HTTP {e.code} on {url}")
        except Exception as e:
            print(f"Download error on {url}: {e}")

    if zip_bytes is None:
        raise RuntimeError("Could not retrieve SEC 13F bulk data from EDGAR endpoints.")

    print("Extracting archive...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(target_dir)

    info_path, sub_path = find_extracted_files(target_dir)
    if not info_path or not sub_path:
        raise FileNotFoundError(
            f"Extracted archive in {target_dir} but could not find INFOTABLE and SUBMISSION tables."
        )

    print(f"Extracted bulk tables to: {target_dir}")
    return info_path, sub_path


def get_sec_ticker_mapping(tickers: list) -> pd.DataFrame:
    if os.path.exists(TICKERS_CACHE_FILE):
        print(f"Loading cached SEC company tickers from: {TICKERS_CACHE_FILE}")
        with open(TICKERS_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("Fetching SEC company ticker directory from EDGAR...")
        raw_data = fetch_url_bytes(SEC_TICKERS_URL)
        data = json.loads(raw_data.decode("utf-8"))
        os.makedirs(os.path.dirname(TICKERS_CACHE_FILE) or ".", exist_ok=True)
        with open(TICKERS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Saved SEC ticker directory cache to: {TICKERS_CACHE_FILE}")

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df[df["ticker"].isin(tickers)].copy()
    
    # Clean issuer title
    clean_title = df["title"].fillna("").astype(str).str.upper().str.strip()
    df["title_clean"] = clean_title
    # Keep alphanumeric root for fuzzy title matching (e.g. APPLE INC vs APPLE INC DE)
    df["title_root"] = clean_title.str.replace(r"[^A-Z0-9]", "", regex=True).str.slice(0, 10)
    return df


def load_submission_table(sub_path: str) -> pd.DataFrame:
    """Reads SUBMISSION.tsv and attaches MANAGER_NAME from COVERPAGE.tsv if available."""
    sample_sub = pd.read_csv(sub_path, sep="\t", nrows=5)
    cols = sample_sub.columns.tolist()
    
    accession_col = next((c for c in cols if "ACCESSION" in c.upper()), None)
    cik_col = next((c for c in cols if c.upper() == "CIK"), None)
    name_col = next((c for c in cols if any(k in c.upper() for k in ["NAME", "MANAGER"])), None)

    if not accession_col:
        accession_col = cols[0]
    if not cik_col:
        cik_col = cols[1] if len(cols) > 1 else accession_col

    usecols = [accession_col, cik_col]
    if name_col and name_col not in usecols:
        usecols.append(name_col)

    print(f"Reading SUBMISSION using detected columns: {usecols}")
    df_sub = pd.read_csv(sub_path, sep="\t", usecols=usecols, dtype=str, low_memory=False)
    rename_dict = {accession_col: "ACCESSION_NUMBER", cik_col: "CIK"}
    if name_col:
        rename_dict[name_col] = "MANAGER_NAME"
    df_sub = df_sub.rename(columns=rename_dict)

    # If MANAGER_NAME is not in SUBMISSION.tsv, extract from COVERPAGE.tsv
    if "MANAGER_NAME" not in df_sub.columns:
        cover_path = os.path.join(os.path.dirname(sub_path), "COVERPAGE.tsv")
        if os.path.exists(cover_path):
            sample_cp = pd.read_csv(cover_path, sep="\t", nrows=5)
            cp_cols = sample_cp.columns.tolist()
            cp_acc = next((c for c in cp_cols if "ACCESSION" in c.upper()), None)
            cp_name = next((c for c in cp_cols if any(k in c.upper() for k in ["FILINGMANAGER_NAME", "MANAGER", "NAME"])), None)
            if cp_acc and cp_name:
                print(f"Attaching manager names from {os.path.basename(cover_path)} using column: {cp_name}")
                df_cp = pd.read_csv(cover_path, sep="\t", usecols=[cp_acc, cp_name], dtype=str, low_memory=False)
                df_cp = df_cp.rename(columns={cp_acc: "ACCESSION_NUMBER", cp_name: "MANAGER_NAME"})
                df_sub = df_sub.merge(df_cp, on="ACCESSION_NUMBER", how="left")

    return df_sub


def build_ownership_overlap_graph():
    # 1. Ingest universe tickers from Step 1 pipeline
    price_pipe = SP500PricePipeline()
    price_pipe.load_raw_data()
    tickers = price_pipe.tickers
    num_stocks = len(tickers)
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}

    print(f"Aligning with {num_stocks} stocks from Step 1 price dataset.")

    # 2. Download and extract SEC 13F bulk tables
    info_path, sub_path = download_and_extract_sec_data()

    # Load submission metadata
    df_sub = load_submission_table(sub_path)

    # Ingest company metadata for mapping
    df_sec_tickers = get_sec_ticker_mapping(tickers)
    title_to_ticker = dict(zip(df_sec_tickers["title_clean"], df_sec_tickers["ticker"]))
    root_to_ticker = dict(zip(df_sec_tickers["title_root"], df_sec_tickers["ticker"]))

    # Detect INFOTABLE columns
    sample_info = pd.read_csv(info_path, sep="\t", nrows=5)
    info_cols = sample_info.columns.tolist()

    acc_col = next((c for c in info_cols if "ACCESSION" in c.upper()), info_cols[0])
    issuer_col = next((c for c in info_cols if any(k in c.upper() for k in ["ISSUER", "NAME"])), None)
    value_col = next((c for c in info_cols if "VALUE" in c.upper()), None)
    cusip_col = next((c for c in info_cols if "CUSIP" in c.upper()), None)

    read_cols = [acc_col]
    if issuer_col: read_cols.append(issuer_col)
    if value_col: read_cols.append(value_col)
    if cusip_col: read_cols.append(cusip_col)

    print(f"Parsing INFOTABLE with detected columns: {read_cols}")
    
    chunk_list = []
    # Read in chunks for speed, low memory, and strict null safety
    for chunk in pd.read_csv(info_path, sep="\t", usecols=read_cols, chunksize=150000, low_memory=False):
        if issuer_col:
            # 1. Clean strings with strict fillna to eliminate float/NaN issues
            clean_titles = chunk[issuer_col].fillna("").astype(str).str.upper().str.strip()
            chunk["TITLE_CLEAN"] = clean_titles
            
            # Vectorized alphanumeric root extraction (e.g., 'APPLE INC.' -> 'APPLEINC')
            chunk["TITLE_ROOT"] = (
                clean_titles.str.replace(r"[^A-Z0-9]", "", regex=True).str.slice(0, 10)
            )

            # 2. Map directly by title, then fallback to title root
            chunk["ticker"] = chunk["TITLE_CLEAN"].map(title_to_ticker)
            unmatched_mask = chunk["ticker"].isna()
            chunk.loc[unmatched_mask, "ticker"] = chunk.loc[unmatched_mask, "TITLE_ROOT"].map(root_to_ticker)

            # 3. Filter matched rows
            matched_chunk = chunk[chunk["ticker"].notna()].copy()
            if not matched_chunk.empty:
                rename_map = {acc_col: "ACCESSION_NUMBER"}
                keep_cols = ["ACCESSION_NUMBER", "ticker"]
                if value_col and value_col in matched_chunk.columns:
                    rename_map[value_col] = "VALUE"
                    keep_cols.append("VALUE")
                else:
                    matched_chunk["VALUE"] = 1000.0
                    keep_cols.append("VALUE")

                chunk_list.append(matched_chunk.rename(columns=rename_map)[keep_cols])

    if not chunk_list:
        raise ValueError("Could not match any equity holdings in INFOTABLE against S&P 500 company titles.")

    df_info = pd.concat(chunk_list, ignore_index=True)
    if value_col and value_col in df_info.columns:
        df_info["VALUE"] = pd.to_numeric(df_info[value_col], errors="coerce").fillna(0.0)
    else:
        df_info["VALUE"] = 1000.0

    print(f"Matched {len(df_info)} equity holding records for S&P 500 universe.")

    # 3. Merge with SUBMISSION to attach institutional CIKs and manager names
    merge_sub_cols = ["ACCESSION_NUMBER", "CIK"]
    if "MANAGER_NAME" in df_sub.columns:
        merge_sub_cols.append("MANAGER_NAME")
    df_merged = df_info.merge(df_sub[merge_sub_cols], on="ACCESSION_NUMBER", how="inner")

    # 4. Filter passive index noise:
    # 4a. Exclude mega-cap passive indexers (Vanguard, BlackRock, SSGA, Geode, Northern Trust)
    if "MANAGER_NAME" in df_merged.columns:
        passive_regex = "|".join(PASSIVE_MANAGER_NAME_PATTERNS)
        is_passive = df_merged["MANAGER_NAME"].str.upper().str.contains(passive_regex, regex=True, na=False)
        passive_count = int(is_passive.sum())
        print(f"Filtered {passive_count} passive records matching Vanguard, BlackRock, State Street/SSGA, Geode, Northern Trust.")
        df_merged = df_merged[~is_passive].copy()

    # 4b. Filter micro-positions: positions with value >= $1M (13F VALUE is in thousands)
    df_active = df_merged[df_merged["VALUE"] >= 1000.0].copy()
    if len(df_active) < 1000:
        # Fallback if values are reported in single dollars rather than thousands
        df_active = df_merged.copy()

    # Create Binary Holding Matrix: Rows = Institutions (CIK), Columns = Stocks (Ticker)
    pivoted = df_active.pivot_table(
        index="CIK",
        columns="ticker",
        values="VALUE",
        aggfunc="count",
        fill_value=0,
    )

    # Ensure all 472 tickers exist as columns
    for ticker in tickers:
        if ticker not in pivoted.columns:
            pivoted[ticker] = 0

    pivoted = pivoted[tickers]  # Exact match to Step 1 order
    binary_holdings = (pivoted.values > 0).astype(float)  # (Num_Managers, 472)

    num_managers = binary_holdings.shape[0]
    matched_stocks = (binary_holdings.sum(axis=0) > 0).sum()
    print(f"Extracted {num_managers} distinct institutional managers holding {matched_stocks}/{num_stocks} S&P 500 equities.")

    # 5. Compute Pairwise Jaccard Similarity Matrix A_own (472 x 472)
    intersection = np.dot(binary_holdings.T, binary_holdings)  # (472, 472)
    holdings_per_stock = np.diag(intersection)

    union = holdings_per_stock[:, None] + holdings_per_stock[None, :] - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_sim = np.where(union > 0, intersection / union, 0.0)

    # Enforce self-loops (A_ii = 1.0)
    np.fill_diagonal(jaccard_sim, 1.0)

    A_own_tensor = torch.tensor(jaccard_sim, dtype=torch.float32)

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_MATRIX_PATH), exist_ok=True)
    torch.save(A_own_tensor, OUTPUT_MATRIX_PATH)
    print(f"\nSuccessfully generated and saved A_own: {A_own_tensor.shape} -> {OUTPUT_MATRIX_PATH}")

    # 6. Sanity inspection for Milestone Report
    print("\n--- Pilot Ownership Matrix Sanity Inspection ---")
    print(f"A_own Shape         : {A_own_tensor.shape}")
    print(f"A_own Density (>0)  : {(A_own_tensor > 0.0).float().mean().item() * 100:.2f}%")
    print(f"Min / Max Similarity: [{A_own_tensor.min():.4f}, {A_own_tensor.max():.4f}]")

    # Sample sub-matrix for Table 1 in Milestone report
    sample_symbols = [s for s in ["AAPL", "MSFT", "NVDA", "AMZN", "META"] if s in ticker_to_idx]
    sample_indices = [ticker_to_idx[s] for s in sample_symbols]

    sub_mat = A_own_tensor[np.ix_(sample_indices, sample_indices)].numpy()
    df_sub_mat = pd.DataFrame(sub_mat, index=sample_symbols, columns=sample_symbols)
    print("\nSample Jaccard-Normalized Adjacency Sub-Matrix for Milestone Report:")
    print(df_sub_mat.round(3))


if __name__ == "__main__":
    build_ownership_overlap_graph()