from pathlib import Path

# ---- roots ----
# Project root; all paths are derived from it.
ROOT = Path(__file__).resolve().parents[1]

### MAIN FOLDERS ###
CONFIG               = ROOT / "config"           # Config files directory
QUALITY_OF_LIFE      = ROOT / "qol"              # Utility helper modules
INDEX_CONSTR         = ROOT / "index_construction"  # FAISS index building scripts
MAIN_PIPE            = ROOT / "pipeline"         # Pipeline stage scripts
INPUT_DIR            = ROOT / "input"            # User-provided inputs
OUTPUT_DIR           = ROOT / "output"           # Main output directory
RESOURCES            = ROOT / "resources"        # Static data (lookups, indices)
TEST_DIR             = ROOT / "tests"            # Unit and regression tests

### FRAMEWORK PARAMETERS ###
# Minimum required columns in any fusion TSV provided by the user.
REQUIRED_COLUMNS = ["gene1", "gene2", "transcript_id1", "transcript_id2", "chromosome1", "chromosome2", "breakpoint1", "breakpoint2"]


LOOKUPS = RESOURCES / "lookups" # Folder for lookup tables
ENSEMBL = RESOURCES / "ensembl" # Folder for Ensembl resources

GTF_ENSEMBL = ENSEMBL / "Homo_sapiens.GRCh38.113.gtf"  # Ensembl GTF file (gene and transcript annotations)
FA_ENSEMBL = ENSEMBL / "Homo_sapiens.GRCh38.pep.fa"
PEPTIDE_LOOKUP = LOOKUPS / "peptide_lookup.tsv"         # Transcript→peptide lookup
CDS_LOOKUP = LOOKUPS / "cds_lookup.tsv"                 # CDS features lookup
MANE_LOOKUP = LOOKUPS / "mane_lookup.tsv"               # MANE transcript + TSS/TES lookup
NEXT_GENE_LOOKUP = LOOKUPS / "next_gene_lookup.tsv"     # Gene adjacency table (for read-through checks)
ENSP_LOOKUP = LOOKUPS / "peptide_lookup_ensp.tsv"       # ENSP peptide lookup (alternative)
ALL_LOOKUP = LOOKUPS / "all_features_lookup.tsv"        # Unified lookup with all feature types

# ---- Output directories ----
INTERIM = OUTPUT_DIR / "interim"               # Intermediate stage outputs
FINAL   = OUTPUT_DIR / "final"                 # Final fused tables
STATS   = OUTPUT_DIR / "statistics"            # Evaluation metrics and stats
VISUALS = OUTPUT_DIR / "visuals"               # Plots and figures
ARRIBA_OUT = OUTPUT_DIR / "arriba_analysis"    # Arriba expression plot outputs

IPS_OUT = OUTPUT_DIR / "interim" / "interpro"  # InterProScan intermediates
PEP_DIR = INTERIM / "peptide_sequences"        # Generated peptide FASTAs
FAISS_OUT = INTERIM / "faiss"                  # FAISS temporary data
FAISS_DIR = RESOURCES / "FAISS"                # Stored FAISS indices
DOMAIN_WINDOWS = FAISS_DIR / "domain_windows.tsv"  # Windows metadata table
PROTEIN_EMBEDDINGS = FAISS_DIR / "embeddings.npy"  # ESM2 embedding matrix
META_FILE = FAISS_DIR / "meta.json"               # Embedding metadata


# ---- I/O defaults ----
OUTPUT_EXT = ".tsv"                             # Default extension for saved files
OUTPUT_SEP = "\t"                               # Default column separator (TSV)

HGNC = ENSEMBL / "hgnc_complete.txt"            # HGNC gene reference for validation
EXP_JSON_DIR = ARRIBA_OUT / "jsons"             # JSONs with detailed Arriba metrics
EXP_IMG_DIR  = ARRIBA_OUT / "debug_images"      # Debug overlay images for Arriba
EXP_CSV_DIR  = ARRIBA_OUT / "csv"               # CSV summaries for Arriba analysis

# ====================== Tunable Parameters ===============================

# ---- Stage 2: FAISS-based Domain Retriever ----
ID_COL = "id"                                   # Unique peptide identifier column
FAISS_INDEX = FAISS_DIR / "domain_flat_index.index"  # Prebuilt FAISS index

TOPK_RAW     = 256                              # Raw neighbors retrieved per query
K_UNIQUE     = 8                                # Max unique InterPro IDs per peptide
FAISS_THREADS = 8                               # FAISS search threads

GPUS = 1                                        # GPUs for ESM2 embedding (1=single GPU)

VERY_LONG_AA   = 1500                           # Threshold for window splitting
WINDOW_SIZE_AA = 1200                           # Window length (AA)
WINDOW_STEP_AA = 800                            # Step between windows (AA)

BATCH_SHORT  = 32                               # Short sequences per batch
BATCH_LONG   = 8                                # Long sequences per batch
LONG_AA_BATCH = 900                             # Length cutoff for long sequences
BUDGET_SHORT = 8000                             # AA budget for short batch
BUDGET_LONG  = 6000                             # AA budget for long batch

SEARCH_BATCH = 2048                             # Embeddings queried per FAISS call

# ---- InterProScan ----
IPS_APPL = "Pfam,ProSiteProfiles,ProSitePatterns,PANTHER"  # Databases to run
IPS_THREADS = 4                               # CPU threads per call
IPS_BATCH_SIZE = 5000                         # Peptides processed per batch

# ---- Arriba expression analysis ----
EXP_BATCH_SIZE = 20                           # PDFs processed per iteration
EXP_WRITE_CSV   = True                        # Write CSV summaries
EXP_WRITE_JSON  = False                       # Write detailed JSONs
EXP_WRITE_DEBUG = False                       # Write debug PNGs

# ---- Domain and function annotation ----
EXPRESSION_COVERAGE_THRESHOLDS = [1.5, 1.6, 1.8, 2.0, 2.5, 3.0]  # Coverage cutoffs
KINASES_KEY_DOMS = [
    # Core catalytic domain
    "Protein kinase domain",                 # generic catalytic domain
    "Serine/threonine-protein kinase domain",
    "Tyrosine-protein kinase domain",

    # ATP-binding and active-site signatures
    "Protein kinase ATP-binding region signature",
    "Protein kinase active site",

    # (Optional) catalytic loop / activation loop
    "Activation loop of protein kinases",    # if present in your InterPro version
]

TS_KEY_DOMS = [
    # TP53 and related DNA-binding tumor suppressors
    "p53 DNA-binding domain",
    "DNA-binding domain of tumor suppressor p53",

    # RB1 tumour-suppressor pocket domain
    "Retinoblastoma-associated protein, pocket domain",

    # PTEN catalytic core (lipid/protein phosphatase)
    "PTEN-like phosphatase domain",
    "Protein tyrosine phosphatase, non-receptor type",

    # BRCA1 C-terminal signalling/repair domains
    "BRCT domain",
]

CANCER_KEY_DOMS = [
    # Kinase catalytic core (overlaps with KINASES_KEY_DOMS)
    "Protein kinase domain",

    # Small GTPases (RAS/RAF pathway, etc.)
    "Small GTP-binding protein domain",
    "Ras-like small GTPase domain",

    # Transcription-factor DNA-binding domains common in drivers
    "Homeobox domain",
    "bZIP transcription factor, basic region",
    "Helix-turn-helix DNA-binding domain",
    "Forkhead (FOX) DNA-binding domain",

    # Nuclear receptor ligand-binding domain (ER, AR, etc.)
    "Nuclear hormone receptor, ligand-binding domain",
]


ACTIONABLE_KEY_DOMS = [
    # Kinases
    "Protein kinase domain",

    # Receptor tyrosine kinase extracellular / transmembrane regions
    "Immunoglobulin-like domain",
    "Fibronectin type III domain",
    "Epidermal growth factor-like domain",
    "Transmembrane receptor protein tyrosine kinase domain",

    # GPCRs
    "G protein-coupled receptor, rhodopsin-like, 7TM domain",

    # Nuclear hormone receptors
    "Nuclear hormone receptor, ligand-binding domain",

    # Ion channels (pore-forming)
    "Ion transport domain",
    "Voltage-gated potassium channel, pore region",
]

DRUGGABLE_KEY_DOMS = [
    # Include all actionable domains
    "Protein kinase domain",
    "G protein-coupled receptor, rhodopsin-like, 7TM domain",
    "Nuclear hormone receptor, ligand-binding domain",
    "Ion transport domain",

    # Enzyme active cores frequently drugged
    "Serine protease domain",
    "Cysteine protease domain",
    "Metalloprotease catalytic domain",
    "Protein tyrosine phosphatase domain",

    # Extracellular receptor domains targeted by antibodies
    "Immunoglobulin-like domain",
    "Fibronectin type III domain",
    "Epidermal growth factor-like domain",
]


# ---- Read-through & promoter hijacking thresholds (S05) ----
RT_MAX_GAP        = 10000                    # Max genomic distance for read-through (bp)
RT_MAX_GENE_GAP   = 20000                    # Max inter-gene gap (bp)
RT_BP_WINDOW      = 1000                     # Breakpoint proximity window (bp)
RT_USE_TX_ANCHORS = 1                         # 1=use transcript TSS/TES; 0=use gene anchors
RT_PROMOTE_LOOSE  = 0                         # 1=upgrade loose read-through to strict

PH_WIN_500   = 500                            # Promoter-hijacking window (±500 bp)
PH_WIN_1000  = 1000                           # Promoter-hijacking window (±1000 bp)
PH_PROM_WIN  = 1500                           # Promoter region window for classification (bp)

# ================= Oncogenicity scoring (S06) =============================
KNOWN_POSITIVES = RESOURCES / "gene_features" / "known_driver_fusions.csv"  

HASH_DIM     = 2**18           # Hashing vectorizer dimensionality
TFIDF_ON     = True            # True: TF-IDF; False: binary hashed
TFIDF_NORM   = "l2"            # "l1", "l2", or "none"

KNN_K        = 20              # Nearest neighbors per node in kNN graph
ALPHA        = 0.9             # Diffusion retention (1−α = teleport rate)
MAX_ITER     = 100             # Max diffusion iterations
TOL          = 1e-6            # Convergence tolerance
METRICS_K    = [10, 50, 100]   # K cutoffs for recall/EF/NDCG metrics
EVAL = False                   # False=production; True=research (split + metrics)
SPLIT_HOLDOUT_FRAC = 0.2       # Fraction of positives reserved for holdout
RANDOM_SEED = 34               # Random seed for reproducibility

# ================= Domain Retrieval (S01–S05) ============================
IPS_TMPDIR         = IPS_OUT / "tmp"          # Temporary InterProScan folder
SPLIT_PROTEOME     = True                     # Split proteome into chunks
PROTEOME_CHUNKS    = 32                       # Number of chunks
IPS_CHUNK_DIR      = IPS_OUT / "chunks"       # Chunk output folder
IPS_PROTEOME       = ENSEMBL / "proteome_for_interpro.fa"  # FASTA for InterProScan

REQUIRE_IPR        = True                     # Keep only hits with InterPro IDs
MIN_LEN_AA         = 30                       # Min accepted domain/window length
MAX_LEN_AA         = 600                      # Max accepted domain/window length

WINDOW_CTX_LEFT    = 32                       # Amino acids of left context per window
WINDOW_CTX_RIGHT   = 32                       # Amino acids of right context per window

ESM_MODEL_NAME     = "esm2_t33_650M_UR50D"    # ESM2 model to use
ESM_DEVICE         = "cuda"                   # Device: "cuda" or "cpu"
ESM_BATCH          = 32                       # Sequences per embedding batch
ESM_MAX_TOKENS     = 4096                     # Max tokens per batch
USE_FP16           = False                    # Use mixed precision if supported

FAISS_METRIC       = "ip"                     # FAISS metric ("ip"=inner product, "l2"=euclidean)

# ---- InterProScan binary path ----
IPS_BIN = "/work/H2020DeciderFicarra/interpro/interproscan-5.76-107.0/interproscan.sh"          # Absolute path to InterProScan executable
