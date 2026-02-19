# 🧬 FusFun — Functional Interpretation of Gene Fusions

**FusFun** is a modular bioinformatics framework for the functional characterization of gene fusions.  
It reconstructs fusion peptides, retrieves protein domains using a hybrid **InterProScan OR FAISS** pipeline, integrates expression and structural data, 
and scores oncogenic potential through diffusion-based PU learning.

---

## Installation guide

Create and activate your environment, then install dependencies:

```bash
# (Optional) create venv
python3 -m venv fusfun_env
source fusfun_env/bin/activate

# install all packages
pip install -r requirements.txt
```


---

## 📂 Data Setup (Required)

To keep the repository lightweight, the large resource files (FAISS indices, lookups, and genomic references) are hosted externally. **These files (~8.5GB) are mandatory for the pipeline to run.**

1. **Download the resources:** Access the Google Drive folder [here](https://drive.google.com/drive/folders/1kcgopzXEs2E--cce5o9yz_gv297p_2t0?usp=sharing).
2. **Placement:** Download the contents and place them inside the `resources/` directory in the project root.
3. **Structure Check:** After downloading, your folder structure must look like this:

```bash
FusFun/
├── resources/
│   ├── FAISS/          # Index files and embeddings
│   ├── lookups/        # Peptide and CDS lookup tables
│   ├── ensembl/        # Genomic references and HGNC data
│   └── gene_features/  # Known driver fusions
```
---

## OVERVIEW

What you will find in this framework:

1. Preprocessing step: Map fusion breakpoints to coding sequences and reconstruct the retained and lost peptide fragments for each fusion partner. 
2. Domain retrieval (FAISS-based): Retrieve protein domains from peptide fragments using vector similarity against a prebuilt FAISS index of domain embeddings.
2b. Domain retrieval (InterProScan-based): Retrieve protein domains from peptide fragments using the standard tool InterProScan.
3. Arriba expression analysis: Extract retained vs lost expression signal from Arriba-like PDF plots by locating the breakpoint in each histogram and quantifying exon-level bar heights.
4. Feature analysis: Derive fusion-level feature flags from lookups and domain outputs (strand coherence, UTR retention, truncation, frame status, and class-specific domain loss).
5. Functional analysis: Add biological interpretation flags per fusion (same-gene, read-through, promoter hijacking, true-fusion thresholds).

Note: Step 3 and the “true fusion” category rely on Arriba detection tool PDFs.
If such PDFs are missing, the corresponding analyses will be skipped, but all other steps will still run.

### INPUT FILES

This framework requires:
1. Fusion description table (TSV/CSV format)
2. Arriba PDF folder (optional for expression analysis)

The input CSV/TSV must contain at least these columns:
["gene1", "gene2", "transcript_id1", "transcript_id2", "chromosome1", "chromosome2", "breakpoint1", "breakpoint2"]

Extra columns are preserved automatically.
All Arriba PDFs should be located in a single folder.

### MAIN OUTPUT FILES
Results are written under output/.
- output/final/ comprehensive analysis (<input>_complete_analysis.tsv)
- output/interim/ — intermediate files for each processing stage
- output/arriba_analysis/ — Arriba expression results
    - csv/<pdf_dir_stem>_coverage_metrics_<itr>.tsv
    - jsons/<pdf_dir_stem>_in_depth_<itr>.json
    - debug_images/<pdf_dir_stem>_page<N>.png

Arriba output is generated only if the corresponding flag --arriba is passed as input.
JSON and image outputs are generated only if the corresponding flags are set to True in the configuration file (see more about that later).

## Quick Start

First of all, if you will use InterProScan please change the path to its executable in the config file. 
It is the last entry in the document, scroll down until you see:
---- InterProScan binary path ----

```bash
cd FusFun
python -m run_pipeline --input tests/test_file.csv --arriba tests/pdf_test --gpu 0
```

--gpu 0 means CPU-only mode.
Increase the value (e.g., --gpu 1) to use GPU acceleration if available.

---

### CLI ARGUMENTS TABLE
Here is a table summarizing all the CLI arguments you can input to the pipeline. 
You can also have more in-depth information in the description related to each component below.

COMMAND: python -m run_pipeline
--input <path/to/file> = Input filename under input/ or absolute path
--faiss = Use FAISS for domain retrieval (default)
--ips = Use InterProScan for domain retrieval
--gpu <N>= Number of GPUs to use (Slurm sets CUDA_VISIBLE_DEVICES). 0 means CPU.
--arriba <path/to/folder/>= Folder containing Arriba plot PDFs to analyze
--delinter = Delete all intermediate files under output/interim after the pipeline completes.

COMMAND: python -m qol.build_fasta_lookups

--name <name> = Base name prefix for all generated fasta lookup files

COMMAND: python -m qol.build_gtf_lookups

--name <name> = Base name prefix for all generated gtf lookup files

COMMAND: python -m qol.run_comparison

--ips <path/to/file> = Path to CSV/TSV file with InterPro retrieved domain sets
--faiss <path/to/file> = Path to CSV/TSV file with FAISS retrieved domain sets
--out-rows <path/to/folder/> = Custom output directory for per-row comparison TSV
--out_stats <path/to/folder/> = Custom output directory for aggregated comparison TSV
--id-col <id_col> = Custom join key present in both files (default: id)
--name <name> = Custom basename for outputs (overrides default derived from your file's basename).
 
COMMAND: python -m qol.rebuild_index

--all = Run all the stages, from InterProScan domain retrieval to the construction of the FAISS index.
--embeddings = Run only the embedding stage and index reconstruction stages.
--index = Run only the index reconstruction stage.
--name <name> = The new index name you want.

## Main pipeline runs examples

1. Run complete analysis if you have a dataset coming from Arriba fusion detection tool:

```bash
python -m run_pipeline --input absolute/path/to/your/file --arriba absolute/path/to/folder/containing/arriba_pdfs
```

You can also put your files in the dedicated folders if you prefer, in which case:
```bash
python -m run_pipeline --input input/your_file --arriba input/your_arriba_pdfs_folder/
```

You can set which tool you can use for retrieval by passing --faiss or --ips, by default FAISS is used:
```bash
python -m run_pipeline --input input/your_file --faiss

python -m run_pipeline --input input/your_file --ips
```

2. If you don't have Arriba data, you will only miss the expression analysis (3rd stage) and with it the annotation of the "true fusion" events.

In this case just run:
```bash
python -m run_pipeline --input input/your_file
```

3. By default, the pipeline is run on a single gpu. To change this behaviour you can run
```bash
python -m run_pipeline --input input/your_file --gpu 2
```

```bash
python -m run_pipeline --input input/your_file --gpu 0
```

```bash
python -m run_pipeline --input input/your_file --gpu 4
```

4. If you are not interested about the intermediate files, you can delete them by passing --delinter argument. These files comprehend:
- output/interim/peptide_sequences/file.tsv : tables containing only id + retained and lost peptide sequences of your fusions from stage 1 (preprocessing).
- output/interim/interpro/file.tsv and output/interim/faiss/file.tsv : tables containing only id + retained and lost retrieved protein domains of your fusions from stage 2 (domain retrieval)
- output/interim/file_features.tsv : tables containing only id + columns added by stage 4 (features extraction).
- output/interim/file_annotated.tsv : tables containing only id + columns added by stage 5 (functional annotation).


CARE: ALL THE FILES LIVING UNDER THE FOLDER /interim WILL BE DELETED, PASS THIS COMMAND ONLY AFTER CHECKING NO FILE YOU CARE ABOUT IS THERE.

```bash
python -m run_pipeline --input input/your_file --delinter
```

## 🔹 Stage 01 — Preprocessing (Peptide Reconstruction)

**Purpose**  
Map fusion breakpoints to coding sequences and reconstruct the retained and lost peptide fragments for each fusion partner.

**Inputs**  
| Type | Path | Description |
|------|------|-------------|
| Fusion metadata | `input/<file>.csv` | must include:<br>`gene1, gene2, transcript_id1, transcript_id2, chromosome1, chromosome2, breakpoint1, breakpoint2` |
| Peptide lookup | `resources/lookups/peptide_lookup.tsv` | maps `transcript_id → peptide_sequence` |
| CDS lookup | `resources/lookups/cds_lookup.tsv` | provides `start, end, cumulative_length, length` for CDS segments |

**Process**  
1. Load metadata and lookups with `qol.utilities.load_file`.  
2. Attach peptide sequences (`peptide_sequence1`, `peptide_sequence2`).  
3. Convert breakpoints → CDS coordinates → amino-acid index.  
4. Split each peptide into `retained_peptide{1,2}` and `lost_peptide{1,2}`.  
5. Replace `*` with `X`; fill missing fields with `"."`.  


**Outputs**  
```
output/interim/peptide_sequences/<stem>_peptides.tsv
output/interim/peptide_sequences/<stem>_peptides.fasta
```

**Columns added**  
`retained_peptide1, lost_peptide1, retained_peptide2, lost_peptide2`

**CLI usage**  
```bash
python -m pipeline.S01_preprocessing <input_tsv>
```
or automatically when running  
```bash
python -m run_pipeline --input tests/test_file.csv
```

---

## 🔹 Stage 02 — Domain Retrieval (FAISS / InterProScan)

### Option A — FAISS Domain Retriever (default)

**Purpose**  
Retrieve protein domains from peptide fragments using vector similarity against a prebuilt FAISS index of domain embeddings.

**Inputs**  
| Type | Path | Description |
|------|------|-------------|
| Peptides | `output/interim/peptide_sequences/<stem>_peptides.tsv` | from Stage 01 |
| FAISS index | `resources/FAISS/domain_flat_index.index` | prebuilt domain embedding index |
| Metadata | `resources/FAISS/meta.json` | describes each index vector and InterPro mapping |

**Process**  
1. Load peptides and sanitize missing fragments.  
2. Load FAISS index and metadata.  
3. Embed peptides using ESM2 (t33-650M-UR50D).  
4. Search FAISS for top-`K` nearest neighbors.  
5. Aggregate unique InterPro IDs.  
6. Save both “domains-only” and “domains-full” TSVs.

**Outputs**  
```
output/interim/faiss/<stem>_domains.tsv
output/interim/faiss/<stem>_peptides_domains_full.tsv
```

**Columns added**  
`retained_domains1, lost_domains1, retained_domains2, lost_domains2`

**CLI usage**  
```bash
python -m pipeline.S02_domain_retriever <input_tsv>
```
or via pipeline  
```bash
python -m run_pipeline --input tests/test_file.csv
```

---

### Option B — InterProScan Retriever (S02b)

**Purpose**  
Run InterProScan locally on each peptide fragment and aggregate InterPro accessions per fusion partner.

**Inputs**  
| Type | Path | Description |
|------|------|-------------|
| Peptides | `output/interim/peptide_sequences/<stem>_peptides.tsv` | from Stage 01 |
| InterProScan | `IPS_BIN` in `config/config.py` | path to executable script |

**Process**  
1. Filter out empty sequences.  
2. Write temporary FASTA batches.  
3. Run InterProScan with specified member databases.  
4. Parse column 12 for InterPro accessions.  
5. Aggregate and merge results back to metadata.

**Outputs**  
```
output/interim/interpro/<stem>_ips_domains.tsv
output/interim/interpro/<stem>_ips_full.tsv
```
**Columns added**  
`retained_domains1, lost_domains1, retained_domains2, lost_domains2`

**CLI usage**  
```bash
python -m pipeline.S02b_run_interpro <input_tsv>
```
or via pipeline  
```bash
python -m run_pipeline --input tests/test_file.csv --ips
```

---

## 🔹 Stage 03 — Expression Analysis (Arriba PDFs)

**Purpose**  
Extract retained vs lost expression signal from Arriba-like PDF plots by locating the breakpoint in each histogram and quantifying exon-level bar heights.

**Inputs**  
| Type | Path | Description |
|------|------|-------------|
| Arriba PDFs folder | `<PDF_DIR>` | directory with input PDFs (two stacked histograms per page) |
| HGNC symbols | `resources/ensembl/hgnc_complete.txt` | filters gene tokens parsed from the page text |
| Transcript lookup | `resources/lookups/all_features_lookup.tsv` | provides `transcript_id → strand` |

**Process**  
1. Render each page, detect gap, split histograms.  
2. Detect blue breakpoint column.  
3. Normalize gray bars to `EXP_SIGNAL_GRAY ± EXP_GRAY_TOL`.  
4. Segment exon bars and compute mean/median coverage.  
5. Save CSV, optional JSON + debug images.

**Outputs**  
```
output/arriba_analysis/csv/<pdf_dir_stem>_coverage_metrics_<itr>.tsv
output/arriba_analysis/jsons/<pdf_dir_stem>_in_depth_<itr>.json
output/arriba_analysis/debug_images/<pdf_dir_stem>_page<N>.png
```

**CLI usage**  
```bash
python -m pipeline.S03_expression_analysis --pdf-dir <PDF_DIR> --iteration 1
```
or via pipeline  
```bash
python -m run_pipeline --input tests/test_file.csv --arriba
```

---

## 🔹 Stage 04 — Feature Analysis

**Purpose**  
Derive fusion-level feature flags from lookups and domain outputs (strand coherence, UTR retention, truncation, frame status, and class-specific domain loss).

**Inputs**  
- `output/interim/faiss/<stem>_peptides_domains_full.tsv`  
- `resources/lookups/*` tables (ALL_LOOKUP, MANE_LOOKUP, CDS_LOOKUP)  

**Output**  
```
output/interim/<stem>_features.tsv
```

**Columns produced**  
strand coherence, UTR retention, truncation, MANE/TSS context, domain loss, promoter/read-through helpers.

**CLI usage**  
```bash
python -m pipeline.S04_feature_analysis <input_tsv>
```
or via pipeline  
```bash
python -m run_pipeline --input tests/test_file.csv
```
---

## 🔹 Stage 05 — Functional Annotation

**Purpose**  
Add biological interpretation flags per fusion (same-gene, read-through, promoter hijacking, true-fusion thresholds).

**Inputs**  
- `output/interim/<stem>_features.tsv`  
- lookup tables for adjacency and coordinates  

**Outputs**  
```
output/interim/<stem>_annotated.tsv
output/interim/<stem>_event_counts.tsv
```

**Flags produced**
strand coherence, UTR retention, truncation, MANE/TSS context, domain loss, promoter/read-through helpers.

**CLI usage**  
```bash
python -m pipeline.S05_functional_annotation <input_tsv>
```
or via pipeline  
```bash
python -m run_pipeline --input tests/test_file.csv 
```
---

## 🔹 Index Construction Pipeline (S01–S05)

**Purpose**  
Build a FAISS domain index from InterProScan results and Ensembl proteome.

**Stages**  
S01 Extract Domains → S02 Filter + Merge → S03 Prep Windows → S04 Embed → S05 Build Index.

**Output**  
```
resources/FAISS/domain_flat_index.index
resources/FAISS/domain_windows.tsv
resources/FAISS/embeddings.npy
resources/FAISS/meta.json
```

**CLI**

Run all the stages, from InterProScan domain retrieval to the construction of the FAISS index:
```bash
python -m qol.rebuild_index --all --name new_index.index
```

Run only the embedding stage and index reconstruction stages.
```bash
python -m qol.rebuild_index --embeddings --name new_index.index
```

Run only the index reconstruction stage.
```bash
python -m qol.rebuild_index --index --name new_index.index
```

### CARE
If you want to use the new input files or new generated files you have to change the paths in the config file.
If you don't change the files' paths for 
DOMAIN_WINDOWS = FAISS_DIR / "domain_windows.tsv"  # Windows metadata table
PROTEIN_EMBEDDINGS = FAISS_DIR / "embeddings.npy"  # ESM2 embedding matrix
META_FILE = FAISS_DIR / "meta.json"               # Embedding metadata
you will OVERWRITE those files, so please move them from the folder or replace the names in the config.

---

## Utilities

### run_comparison.py  
Compares FAISS vs InterPro domain retrieval (precision/recall).
How to launch:
```bash
python -m qol.run_comparison --ips /path/to/interproscan/output.tsv --faiss /path/to/faiss/output.tsv
```
Optional flags:
    --out-rows, custom output directory for per-row comparison TSV
    --out-stats, custom output directory for aggregated stats TSV
    --id-col, custom Join key present in both files (default: id)

### build_fasta_lookups.py  
Rebuilds protein lookups from Ensembl FASTA.
How to launch:
```bash
python -m qol.build_fasta_lookups --name new_base_name
```

### build_gtf_lookups.py  
Regenerates all GTF-derived lookups (ALL, CDS, MANE, NEXT-GENE).
How to launch:
```bash
python -m qol.build_gtf_lookups --name new_base_name

### CARE
If you want to use the new input files or new generated files you have to change the paths in the config file.
---

## 🧭 Repository Overview

```
FusFun
├── config/                      # Global configuration and constants
│   └── config.py
│
├── qol/                         # Quality-of-life and domain retrieval utilities
│   ├── domain_retrieval/      # Independent domain-retrieval subpipeline
│   │   ├── S01_extract_domains.py
│   │   ├── S02_filter_and_merge.py
│   │   ├── S03_prep_domain_windows.py
│   │   ├── S04_embed_domain_windows.py
│   │   └── S05_build_index.py
│   │
│   ├── build_fasta_lookups.py   # Generate FASTA-based peptide lookups
│   ├── build_gtf_lookups.py     # Generate GTF-based MANE/CDS lookups
│   ├── run_comparison.py        # Compare InterPro vs FAISS domain outputs
│   ├── rebuild_index.py         # Orchestrator for S01–S05 index construction
│   └── utilities.py             # Common helpers (robust loaders, paths, etc.)
│
├── pipeline/                    # Main multi-stage fusion analysis workflow
│   ├── S01_preprocessing.py
│   ├── S02_domain_retriever.py
│   ├── S02b_run_interpro.py
│   ├── S03_expression_analysis.py
│   ├── S04_feature_analysis.py
│   ├── S05_functional_annotation.py
│
├── input/                       # Example inputs (CSVs, Arriba PDFs)
├── resources/                   # Reference data (Ensembl, lookups, FAISS index)
├── output/                      # Pipeline results and derived data
├── tests/                       # Minimal test inputs for quick validation
├── run_pipeline.py              # Entry point for the full FusFun workflow
└── requirements.txt
```

---

## ⚙️ Configuration Summary

| Category | Key | Description |
|-----------|-----|-------------|
| Paths | `ROOT`, `OUTPUT_DIR`, `RESOURCES` | Root and folder paths |
| Lookups | `PEPTIDE_LOOKUP`, `CDS_LOOKUP`, `MANE_LOOKUP`, `NEXT_GENE_LOOKUP` | Ensembl-derived reference tables |
| InterProScan | `IPS_BIN`, `IPS_APPL`, `IPS_THREADS`, `IPS_BATCH_SIZE` | InterPro executable and parameters |
| Domain Windows | `WINDOW_CTX_LEFT`, `WINDOW_CTX_RIGHT`, `MIN_LEN_AA`, `MAX_LEN_AA` | Domain context length filters |
| ESM-2 Embedding | `ESM_MODEL_NAME`, `ESM_DEVICE`, `ESM_BATCH`, `ESM_MAX_TOKENS` | Transformer and hardware settings |
| FAISS Index | `FAISS_DIR`, `FAISS_INDEX`, `FAISS_METRIC` | Index path and similarity metric |
| Expression Analysis | `EXP_BATCH_SIZE`, `EXP_WRITE_CSV`, `EXP_WRITE_JSON` | Arriba PDF settings |
| Feature / Function | `KINASES_KEY_DOMS`, `TS_KEY_DOMS`, `CANCER_KEY_DOMS` | Domain category keys |


---
# ⚙️ Configuration Reference (`config/config.py`)

<details>
<summary><b>🗂️ Directory Structure</b></summary>

| Variable | Description |
|-----------|-------------|
| `ROOT` | Project root; all paths are derived from it. |
| `CONFIG` | Directory containing configuration files. |
| `QUALITY_OF_LIFE` | Utility helpers and shared functions (`qol/`). |
| `INDEX_CONSTR` | Scripts used for FAISS index construction. |
| `MAIN_PIPE` | Main pipeline stage scripts (`pipeline/S01–S06`). |
| `INPUT_DIR` | Folder for user-provided TSV/CSV inputs. |
| `OUTPUT_DIR` | Root folder for all generated outputs. |
| `RESOURCES` | Static data resources (lookups, indices, references). |
| `TEST_DIR` | Unit and regression test inputs. |

</details>

<details>
<summary><b>📄 Lookup & Reference Files</b></summary>

| Variable | Description |
|-----------|-------------|
| `LOOKUPS` | Folder containing all lookup tables. |
| `ENSEMBL` | Folder containing Ensembl resources. |
| `GTF_ENSEMBL` | Ensembl GTF file with gene and transcript annotations. |
| `FA_ENSEMBL` | Ensembl protein FASTA (canonical proteome). |
| `PEPTIDE_LOOKUP` | Transcript → peptide mapping table. |
| `CDS_LOOKUP` | CDS feature segments per transcript. |
| `MANE_LOOKUP` | MANE Select/Plus transcript list with TSS/TES. |
| `NEXT_GENE_LOOKUP` | Gene adjacency table (for read-through analysis). |
| `ENSP_LOOKUP` | Alternative peptide lookup (ENSP-based). |
| `ALL_LOOKUP` | Unified lookup with all feature types. |

</details>

<details>
<summary><b>📦 Output Folders</b></summary>

| Variable | Description |
|-----------|-------------|
| `INTERIM` | Intermediate outputs between stages. |
| `FINAL` | Final integrated fusion tables. |
| `STATS` | Performance metrics and summary statistics. |
| `VISUALS` | Plots, graphs, and visualizations. |
| `ARRIBA_OUT` | Outputs from Arriba expression analysis. |
| `IPS_OUT` | Intermediate InterProScan results. |
| `PEP_DIR` | Generated peptide TSVs and FASTAs. |
| `FAISS_OUT` | Temporary FAISS search data. |
| `FAISS_DIR` | Stored FAISS index and embeddings. |
| `DOMAIN_WINDOWS` | Domain window metadata table. |
| `PROTEIN_EMBEDDINGS` | Precomputed ESM2 embedding matrix. |
| `META_FILE` | Metadata for the embeddings file. |

</details>

<details>
<summary><b>🔧 I/O Defaults</b></summary>

| Variable | Description |
|-----------|-------------|
| `OUTPUT_EXT` | Default file extension for outputs. |
| `OUTPUT_SEP` | Default separator (`\t` for TSV). |
| `HGNC` | HGNC gene reference file for symbol validation. |
| `EXP_JSON_DIR`, `EXP_IMG_DIR`, `EXP_CSV_DIR` | Subdirectories for Arriba plot JSONs, debug overlays, and CSV summaries. |

</details>

<details>
<summary><b>🧬 Stage-02 — FAISS-Based Domain Retrieval</b></summary>

| Variable | Description |
|-----------|-------------|
| `ID_COL` | Unique identifier column for each peptide. |
| `FAISS_INDEX` | Path to the prebuilt FAISS index file. |
| `TOPK_RAW` | Number of raw nearest neighbors to retrieve. |
| `K_UNIQUE` | Maximum number of unique InterPro IDs per peptide. |
| `FAISS_THREADS` | Number of CPU threads used by FAISS. |
| `GPUS` | Number of GPUs used for ESM2 embedding. |
| `VERY_LONG_AA` | AA length threshold triggering window splitting. |
| `WINDOW_SIZE_AA` | Length of each sliding window in amino acids. |
| `WINDOW_STEP_AA` | Step size between consecutive windows. |
| `BATCH_SHORT` / `BATCH_LONG` | Number of short/long sequences per batch. |
| `LONG_AA_BATCH` | Length cutoff defining long sequences. |
| `BUDGET_SHORT` / `BUDGET_LONG` | Token budget (AA × batch size) for short/long batches. |
| `SEARCH_BATCH` | Number of embeddings per FAISS query batch. |

</details>

<details>
<summary><b>🧩 InterProScan Parameters</b></summary>

| Variable | Description |
|-----------|-------------|
| `IPS_APPL` | Comma-separated list of InterProScan databases. |
| `IPS_THREADS` | CPU threads per InterProScan call. |
| `IPS_BATCH_SIZE` | Number of peptides per batch. |
| `IPS_BIN` | Absolute path to the InterProScan executable. |

</details>

<details>
<summary><b>📊 Arriba Expression Analysis</b></summary>

| Variable | Description |
|-----------|-------------|
| `EXP_BATCH_SIZE` | Number of PDFs processed per iteration. |
| `EXP_WRITE_CSV` | Whether to write CSV summaries. |
| `EXP_WRITE_JSON` | Whether to save detailed JSONs. |
| `EXP_WRITE_DEBUG` | Whether to save debug PNG overlays. |

</details>

<details>
<summary><b>🧠 Functional & Domain Annotation</b></summary>

| Variable | Description |
|-----------|-------------|
| `EXPRESSION_COVERAGE_THRESHOLDS` | Coverage ratio thresholds for expression imbalance detection. |
| `KINASES_KEY_DOMS`, `TS_KEY_DOMS`, `CANCER_KEY_DOMS`, `ACTIONABLE_KEY_DOMS`, `DRUGGABLE_KEY_DOMS` | Key functional domain categories for annotation. |

</details>

<details>
<summary><b>🔬 Stage-05 — Read-Through & Promoter Hijacking</b></summary>

| Variable | Description |
|-----------|-------------|
| `RT_MAX_GAP`, `RT_MAX_GENE_GAP` | Maximum distances (bp) for read-through detection. |
| `RT_BP_WINDOW` | Breakpoint proximity window (bp). |
| `RT_USE_TX_ANCHORS` | 1 = use transcript anchors (TSS/TES); 0 = gene anchors. |
| `RT_PROMOTE_LOOSE` | Upgrade loose read-through to strict (binary flag). |
| `PH_WIN_500`, `PH_WIN_1000`, `PH_PROM_WIN` | Genomic windows for promoter hijacking detection. |

</details>

<details>
<summary><b>🧱 Domain Retrieval & Index Construction (S01–S05)</b></summary>

| Variable | Description |
|-----------|-------------|
| `IPS_TMPDIR` | Temporary folder for InterProScan runs. |
| `SPLIT_PROTEOME` | Whether to split proteome into chunks. |
| `PROTEOME_CHUNKS` | Number of chunks to generate. |
| `IPS_CHUNK_DIR` | Folder for chunked InterPro outputs. |
| `IPS_PROTEOME` | FASTA used for InterProScan indexing. |
| `REQUIRE_IPR` | Keep only hits with InterPro IDs. |
| `MIN_LEN_AA`, `MAX_LEN_AA` | Minimum/maximum accepted domain lengths. |
| `WINDOW_CTX_LEFT`, `WINDOW_CTX_RIGHT` | Context AAs flanking each window. |
| `ESM_MODEL_NAME` | ESM2 model architecture. |
| `ESM_DEVICE` | Compute device (`cuda` or `cpu`). |
| `ESM_BATCH` | Sequences processed per embedding batch. |
| `ESM_MAX_TOKENS` | Max tokens per forward pass. |
| `USE_FP16` | Enable half-precision (mixed precision). |
| `FAISS_METRIC` | FAISS similarity metric (`ip` = inner product, `l2` = Euclidean). |

</details>

## 📜 License

License.
