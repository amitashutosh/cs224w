# CS224W Project Architecture & Dependency Map

This document outlines the software architecture, module dependencies, data pipelines, class models, and tensor transformations across `src/data` and `src/models` for the **Stock Market Forecasting with Differential Graph Transformer (DGT)** system.

---

## 1. High-Level System Dependency Map

The diagram below maps all project modules, external ingestion endpoints, filesystem artifacts, and dependency flows across the pipeline:

```mermaid
flowchart TD
    %% External Endpoints
    subgraph External ["External Data Sources"]
        sec_edgar["SEC EDGAR Bulk Portal<br/>(01mar2024-31may2024_form13f.zip)"]
        sec_tickers["SEC Company Tickers API<br/>(company_tickers.json)"]
        remote_sp500["GitHub S&P 500 Historical Feed<br/>(sp500.csv)"]
    end

    %% Data Directory Artifacts
    subgraph DataDir ["data/ Directory (Cache & Serialized Artifacts)"]
        sp500_csv[("data/sp500.csv<br/>(2,517 days × 472 stocks)")]
        tickers_json[("data/company_tickers.json<br/>(CIK ↔ Ticker ↔ Title)")]
        sec_bulk_dir[("data/sec_13f_2024q1/<br/>(INFOTABLE.tsv & SUBMISSION.tsv)<br/>[Git-ignored cache ~300MB]")]
        a_own_pt[("data/A_own_pilot.pt<br/>(torch.Tensor 472 × 472)")]
    end

    %% src/data Subgraph
    subgraph SrcData ["src/data (Data Ingestion & Graph Construction)"]
        direction TB
        price_loader["price_loader.py<br/>• SP500PricePipeline<br/>• StockPriceDataset"]
        parse_13f["parse_13f_pilot.py<br/>• download_and_extract_sec_data()<br/>• get_sec_ticker_mapping()<br/>• build_ownership_overlap_graph()"]
    end

    %% src/models Subgraph
    subgraph SrcModels ["src/models (Model Architecture & Testing)"]
        direction TB
        dgt_mod["dgt.py<br/>• MultiheadDiffAttn<br/>• DifferentialGraphTransformer"]
        forward_test["forward_test.py<br/>• run_forward_pass_verification()"]
    end

    %% Root
    subgraph ProjectRoot ["Project Root"]
        main_py["main.py"]
    end

    %% Ingestion & Caching
    remote_sp500 -->|Downloads if absent| price_loader
    price_loader -->|Saves local cache| sp500_csv
    sp500_csv -->|Loads daily prices| price_loader

    sec_tickers -->|Fetches company directory| parse_13f
    parse_13f -->|Caches metadata| tickers_json
    tickers_json -->|Resolves CIK ↔ Ticker| parse_13f

    sec_edgar -->|Extracts bulk TSVs| parse_13f
    parse_13f -->|Unpacks raw tables| sec_bulk_dir
    sec_bulk_dir -->|Reads holdings & managers| parse_13f

    %% Cross-module dependencies
    price_loader -->|Provides S&P 500 tickers universe| parse_13f
    parse_13f -->|Serializes Jaccard overlap graph| a_own_pt

    %% Model flow
    price_loader -->|Yields batches X, y, A_global| forward_test
    a_own_pt -->|Loads institutional prior A_own| forward_test
    dgt_mod -->|Instantiates DGT model| forward_test
    main_py -.->|Optional Entrypoint| forward_test
```

---

## 2. UML Class Diagram (Code Entities & Interfaces)

This UML diagram captures the object-oriented structure, class inheritance, methods, attributes, and relations in `src/data` and `src/models`:

```mermaid
classDiagram
    %% Inheritance
    class Dataset {
        <<PyTorch Dataset>>
        +__len__() int
        +__getitem__(idx) tuple
    }

    class Module {
        <<PyTorch nn.Module>>
        +forward(*args)
        +parameters()
        +train(mode)
        +eval()
    }

    %% Data Pipeline Classes
    class StockPriceDataset {
        +torch.Tensor x
        +torch.Tensor y
        +np.ndarray dates
        +__len__() int
        +__getitem__(idx: int) Tuple~Tensor, Tensor, ndarray~
    }
    Dataset <|-- StockPriceDataset

    class SP500PricePipeline {
        +str csv_path
        +int window_size
        +int batch_size
        +float train_ratio
        +float val_ratio
        +float test_ratio
        +List~str~ tickers
        +DatetimeIndex dates
        +ndarray means
        +ndarray stds
        +Tensor A_global
        +load_raw_data() DataFrame
        +compute_global_correlation(train_prices: ndarray) Tensor
        +_create_chunks(data: ndarray, dates: ndarray) Tuple~Tensor, Tensor, ndarray~
        +build_loaders() Dict~str, DataLoader~
    }
    SP500PricePipeline ..> StockPriceDataset : creates

    %% Model Architecture Classes
    class MultiheadDiffAttn {
        +int embed_dim
        +int num_heads
        +int head_dim
        +float lambda_init
        +Linear q_proj
        +Linear k_proj
        +Linear v_proj
        +Linear out_proj
        +Parameter lambda_q1
        +Parameter lambda_k1
        +Parameter lambda_q2
        +Parameter lambda_k2
        +_build_prior_tensor(adj_priors, x, num_nodes) Tensor
        +forward(x: Tensor, adj_priors: Any) Tensor
    }
    Module <|-- MultiheadDiffAttn

    class DifferentialGraphTransformer {
        +int num_stocks
        +int window_size
        +int embed_dim
        +Linear input_proj
        +Parameter pos_embedding
        +TransformerEncoder temporal_encoder
        +MultiheadDiffAttn spatial_diff_attn
        +LayerNorm norm
        +Sequential head
        +forward(x: Tensor, adj_priors: Any) Tensor
    }
    Module <|-- DifferentialGraphTransformer
    DifferentialGraphTransformer *-- MultiheadDiffAttn : contains

    %% Step 2 Procedural Module
    class Parse13FPilotModule {
        <<module: parse_13f_pilot.py>>
        +fetch_url_bytes(url: str) bytes
        +discover_live_sec_13f_urls() list
        +find_extracted_files(target_dir: str) tuple
        +download_and_extract_sec_data(target_dir: str) tuple
        +get_sec_ticker_mapping(tickers: list) DataFrame
        +load_submission_table(sub_path: str) DataFrame
        +build_ownership_overlap_graph() None
    }
    Parse13FPilotModule ..> SP500PricePipeline : consumes tickers

    %% Step 3 Verification Harness
    class ForwardTestModule {
        <<module: forward_test.py>>
        +run_forward_pass_verification() None
    }
    ForwardTestModule ..> SP500PricePipeline : loads batches
    ForwardTestModule ..> DifferentialGraphTransformer : runs forward & backward
```

---

## 3. End-to-End Tensor Dimensionality & Data Flow

This chart traces tensor transformations chronologically through the three pipeline stages:

1. **Stage 1 (Price Pipeline)** establishes the 472-stock universe, extracts temporal chunks ($X_t$), and computes the **Global Correlation Prior ($A^{\text{global}}$)** for **Attention Head 1**.
2. **Stage 2 (SEC 13F Pipeline)** consumes the 472-stock universe from Stage 1, parses 851,250 institutional equity holdings, and constructs the **Jaccard Co-Holding Prior ($A^{\text{own}}$)** for **Attention Head 2**.
3. **Stage 3 (DGT Model Execution)** conditions multi-head differential spatial attention simultaneously on both graph topologies ($A^{\text{global}}$ and $A^{\text{own}}$) to forecast next-day cross-sectional prices ($\hat{y}_t$).

```mermaid
flowchart TD
    %% ==========================================
    %% STAGE 1: PRICE PIPELINE & PRIOR HEAD 1
    %% ==========================================
    subgraph Stage1 ["STAGE 1: Price Ingestion & Temporal Chunking (price_loader.py)"]
        direction TB
        raw_csv["sp500.csv<br/>Raw Matrix: 2,517 days × 472 stocks"]
        split["Chronological Partitioning<br/>Train: 1,949 days (80%)<br/>Val: 188 days (10%)<br/>Test: 188 days (10%)"]
        zscore["Z-Score Normalization<br/>μ, σ fitted strictly on Train slice"]
        tickers_out["472 S&P 500 Ticker Universe<br/>['A', 'AAL', ..., 'ZTS']"]

        a_global["Global Pearson Correlation Prior A_global<br/>A_ij = corr(P_i, P_j) on Train<br/>Shape: (472, 472)<br/>★ Prior Topology for Head 1 ★"]

        chunks["64-Day Rolling Window Chunking<br/>X_t: (B=32, T=64, N=472)<br/>y_t: (B=32, N=472)"]

        raw_csv --> split
        split --> zscore
        zscore --> tickers_out
        zscore --> a_global
        zscore --> chunks
    end

    %% ==========================================
    %% STAGE 2: SEC 13F GRAPH & PRIOR HEAD 2
    %% ==========================================
    subgraph Stage2 ["STAGE 2: Institutional Ownership Graph Construction (parse_13f_pilot.py)"]
        direction TB
        sec_raw["SEC Form 13F Bulk TSVs (Q1 2024)<br/>INFOTABLE.tsv & SUBMISSION.tsv<br/>851,250 equity records"]
        sec_filter["Filter & Align to 472 Tickers<br/>Threshold: Position Value ≥ $1M"]
        binary_m["Bipartite Binary Ownership Matrix M<br/>Shape: (6,377 managers, 472 stocks)"]

        a_own["Jaccard Co-Holding Overlap Prior A_own<br/>A_ij = |M_i ∩ M_j| / |M_i ∪ M_j|<br/>Shape: (472, 472) | Density: 88.04%<br/>★ Prior Topology for Head 2 ★"]

        sec_raw --> sec_filter
        sec_filter --> binary_m
        binary_m --> a_own
    end

    %% Stage 1 explicitly precedes Stage 2 to establish stock universe
    tickers_out ==>|1. Supplies Universe to Filter Holdings| sec_filter

    %% ==========================================
    %% STAGE 3: DGT MULTI-RELATIONAL MODEL
    %% ==========================================
    subgraph Stage3 ["STAGE 3: Differential Graph Transformer Forward Pass (dgt.py)"]
        direction TB

        %% Temporal Encoding
        input_x["Input Price Sequence Batch X_t<br/>Shape: (B=32, T=64, N=472)"]
        input_proj["1. Linear Projection + Positional Embedding<br/>(B, 64, 472, 1) → (B, 64, 472, d=64)"]
        temp_attn["2. Temporal Transformer Encoder<br/>Reshape: (B*N, 64, d)<br/>Causal Self-Attention across 64-day lookback"]
        h_final["Final Lookback State Slice (t = 64)<br/>h_final: (B=32, N=472, d=64)"]

        %% Multi-Head Spatial Differential Attention
        subgraph MultiHead ["3. MultiheadDiffAttn (Spatial Relational Conditioning)"]
            direction TB
            head1["Head 1: DiffAttn₁ ⊙ A_global<br/>(Macro Correlation Structure)"]
            head2["Head 2: DiffAttn₂ ⊙ A_own<br/>(Institutional Co-Holding Structure)"]
            heads_rest["Heads 3 & 4: Unconstrained DiffAttn<br/>(Latent High-Order Interactions)"]

            diff_combine["Differential Subtraction & Value Aggregation<br/>DiffAttn = Attn_map1 - λ · Attn_map2<br/>out: (B, 472, d=64)"]

            head1 --> diff_combine
            head2 --> diff_combine
            heads_rest --> diff_combine
        end

        %% Prediction
        residual["Residual Connection + LayerNorm<br/>h_out = LayerNorm(h_final + h_spatial)"]
        pred_head["4. Prediction Head (MLP)<br/>Linear(64 → 32) → GELU → Linear(32 → 1)<br/>Output: y_hat (B=32, N=472)"]

        input_x --> input_proj
        input_proj --> temp_attn
        temp_attn --> h_final
        h_final --> head1
        h_final --> head2
        h_final --> heads_rest
        diff_combine --> residual
        residual --> pred_head
    end

    %% Flowing from Stages 1 & 2 into Stage 3
    chunks ==>|2. Feeds Temporal Sequences X_t| input_x
    a_global ==>|3. Feeds Prior Topology to Head 1| head1
    a_own ==>|4. Feeds Prior Topology to Head 2| head2
```

---

## 4. Entity-Relationship Data Model (SEC 13F & Price Series)

The entity relationships connecting institutional filers, filings, equity holdings, and the S&P 500 universe:

```mermaid
erDiagram
    INSTITUTIONAL_MANAGER ||--o{ SUBMISSION : files
    SUBMISSION ||--|{ INFOTABLE : contains
    COMPANY_TICKER ||--o{ INFOTABLE : matches_cusip_or_title
    SP500_UNIVERSE ||--|| COMPANY_TICKER : identifies
    SP500_UNIVERSE ||--o{ DAILY_PRICE_RECORD : has_history
    SP500_UNIVERSE ||--o{ CO_HOLDING_PAIR : forms_overlap
    SP500_UNIVERSE ||--o{ CORRELATION_PAIR : forms_correlation

    INSTITUTIONAL_MANAGER {
        string CIK PK
        string FILING_MANAGER_NAME
    }

    SUBMISSION {
        string ACCESSION_NUMBER PK
        string CIK FK
        string FILING_DATE
    }

    INFOTABLE {
        string ACCESSION_NUMBER FK
        string CUSIP
        string NAMEOFISSUER
        float VALUE
        float SSHPRNAMT
    }

    COMPANY_TICKER {
        int CIK
        string TICKER PK
        string TITLE
    }

    SP500_UNIVERSE {
        string TICKER PK
        int MATRIX_INDEX
        float IN_SAMPLE_MEAN
        float IN_SAMPLE_STD
    }

    DAILY_PRICE_RECORD {
        date DATE PK
        string TICKER FK
        float RAW_PRICE
        float NORMALIZED_PRICE
    }

    CO_HOLDING_PAIR {
        string TICKER_I FK
        string TICKER_J FK
        float JACCARD_SIMILARITY
    }

    CORRELATION_PAIR {
        string TICKER_I FK
        string TICKER_J FK
        float PEARSON_R
    }
```

---

## 5. Module Execution & Verification Flow

| Step       | Script                                                                        | Input Files / Sources                                     | Core Function / Class                                  | Output Produced                                   | Verified Dimensions                                                       |
| :--------- | :---------------------------------------------------------------------------- | :-------------------------------------------------------- | :----------------------------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------ |
| **Step 1** | [price_loader.py](src/data/price_loader.py)                                   | `data/sp500.csv`                                          | `SP500PricePipeline`                                   | Train/Val/Test DataLoaders + $A^{\text{global}}$  | $X: [32, 64, 472]$<br/>$y: [32, 472]$<br/>$A^{\text{global}}: [472, 472]$ |
| **Step 2** | [parse_13f_pilot.py](src/data/parse_13f_pilot.py)                             | SEC EDGAR Q1 bulk TSVs, `company_tickers.json`            | `build_ownership_overlap_graph()`                      | `data/A_own_pilot.pt` (Jaccard similarity matrix) | $A^{\text{own}}: [472, 472]$<br/>Density: $88.04\%$                       |
| **Step 3** | [dgt.py](src/models/dgt.py)<br/>[forward_test.py](src/models/forward_test.py) | Step 1 DataLoaders, $A^{\text{global}}$, $A^{\text{own}}$ | `DifferentialGraphTransformer`<br/>`MultiheadDiffAttn` | End-to-end forward/backward gradient check        | Forward $\hat{y}: [32, 472]$<br/>Backward MSE loss: $0.8933$              |
