# FusFun/qol/utilities.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import polars as pl
import re
from config.config import OUTPUT_EXT, OUTPUT_SEP


def cleanup_interim(interim_dir: Path) -> None:
    """
    Delete all files under INTERIM while keeping folders.
    Silent if folder missing. Does not remove directories.

    Parameters
    ----------
    interim_dir : Path
        Root 'output/interim' directory.
    """
    if not interim_dir.exists():
        print(f"[CLEANUP] No interim folder found at {interim_dir}")
        return
    print(f"[CLEANUP] Deleting all files inside {interim_dir} (keeping folders).")
    for p in interim_dir.rglob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception as e:
                print(f"[CLEANUP] Could not delete {p}: {e}")
    print("[CLEANUP] All intermediate files deleted.")

def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)

def sniff_separator(path: Path, sample_size: int = 65536) -> str:
    """
    Heuristic CSV/TSV sniffer.
    Priority: trusted suffix -> line-consistency counts -> frequency -> default ','.
    """
    path = _as_path(path)
    try:
        sample = path.read_text(errors="ignore")[:sample_size]
    except Exception:
        return ","

    # 0) empty or 1-line header without delimiters
    if not sample or ("\t" not in sample and "," not in sample and ";" not in sample):
        # suffix hint only if present
        suf = path.suffix.lower()
        if suf == ".tsv": return "\t"
        if suf in (".csv", ".gz"): return ","  # .csv.gz handled elsewhere
        return ","

    # 1) suffix hint
    suf = path.suffix.lower()
    if suf == ".tsv": return "\t"
    if suf == ".csv": return ","

    # 2) line-consistency heuristic
    lines = [ln for ln in sample.splitlines()[:200] if ln]  # cap lines
    def score(delim: str) -> tuple[int, int]:
        counts = [ln.count(delim) for ln in lines]
        return (sum(1 for c in counts if c > 0), len(set(counts)))  # (lines_with_delim, distinct_counts)
    tab_score = score("\t")
    comma_score = score(",")
    semi_score = score(";")  # some "CSV" use semicolons

    # Prefer delimiter with more lines containing it, and more consistent counts (fewer distinct)
    candidates = [
        ("\t", tab_score[0], -tab_score[1]),
        (",", comma_score[0], -comma_score[1]),
        (";", semi_score[0], -semi_score[1]),
    ]
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = candidates[0][0]

    # 3) frequency tie-breaker
    if best == ";":
        tabs = sample.count("\t")
        commas = sample.count(",")
        if tabs >= commas and tabs > 0: return "\t"
        if commas > 0: return ","
        return ";"
    return best if best in ("\t", ",") else ","

def _to_snake(name: str) -> str:
    name = name.strip().replace("-", "_").replace(" ", "_")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()

def _normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: _to_snake(c) for c in df.columns})


def check_columns(df: pl.DataFrame, required: Iterable[str]) -> None:
    """
    Ensure all required columns exist. On failure, exit with a clear message.
    """
    req = list(required)
    missing = [c for c in req if c not in df.columns]
    if missing:
        msg = (
            f"[ERROR] Missing required columns: {', '.join(missing)}\n"
            f"Please provide at least: {', '.join(req)}\n"
            f"Available columns: {', '.join(df.columns)}"
        )
        raise SystemExit(msg)

def load_file(path, required: Optional[Iterable[str]] = None) -> pl.DataFrame:
    """
    Robust CSV/TSV loader.
    - Detects delimiter from first line.
    - Retries with the alternative delimiter if width==1.
    - Normalizes headers to snake_case before validation.
    """
    path = _as_path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # --- detect delimiter from first line
    with open(path, "rb") as fb:
        head = fb.read(4096)
    try:
        first_line = head.decode("utf-8", errors="ignore").splitlines()[0]
    except Exception:
        first_line = ""
    detected = "\t" if ("\t" in first_line and first_line.count("\t") >= first_line.count(",")) else ","

    # parsed raw headers (before normalization)
    raw_headers = [h.strip() for h in first_line.split(detected)] if first_line else []

    # schema overrides to prevent int inference on chr-like columns
    chr_like_raw = {"chr", "chromosome", "chromosome1", "chromosome2", "chr1", "chr2", "chrom", "seqname"}
    schema_overrides = {h: pl.Utf8 for h in raw_headers if h in chr_like_raw}

    nulls = ["", "NA", "na", "null", "Null", "N/A", "None"]

    def _read_with(sep: str) -> pl.DataFrame:
        return pl.read_csv(
            path,
            separator=sep,
            null_values=nulls,
            infer_schema_length=1000,
            ignore_errors=False,
            skip_rows=0,
            quote_char='"',
            try_parse_dates=False,
            schema_overrides=schema_overrides,  # <-- key line
        )

    # first attempt
    df = _read_with(detected)
    # fallback: if only one column, try the other separator
    if df.width == 1 and ("," in first_line or "\t" in first_line):
        alt = "," if detected == "\t" else "\t"
        df = _read_with(alt)

    # normalize headers like 'GeneStrand1' -> 'gene_strand1'
    df = _normalize_headers(df)

    # post-read cast for normalized names (belt-and-suspenders)
    for col in ("chr", "chromosome", "chromosome1", "chromosome2", "chr1", "chr2", "chrom"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8))

    # validate required columns
    if required is not None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {path}")

    return df



def save_file(df: pl.DataFrame, path: Path):
    path = _as_path(path)
    if path.suffix.lower() != ".tsv":
        path = path.with_suffix(OUTPUT_EXT)  # from config
    df.write_csv(path, separator=OUTPUT_SEP)
    

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def add_sequential_id(df: pl.DataFrame, col_name: str = "id") -> pl.DataFrame:
    """
    Add a sequential ID column starting from 1 up to n_rows.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.
    col_name : str
        Name of the new ID column. Defaults to 'id'.

    Returns
    -------
    pl.DataFrame
        DataFrame with the new sequential ID column added.
    """
    return df.with_columns(
        pl.arange(1, df.height + 1).alias(col_name)
    )

def basename(name: str | Path) -> str:
    """
    Return a safe stem for an input path or filename.

    Examples
    --------
    >>> basename("input/my_gene_fusions.csv")
    'my_gene_fusions'
    >>> basename("my.tsv.gz")
    'my'
    """
    p = Path(str(name))
    stem = p.name
    # strip double extensions like .csv.gz, .tsv.gz, etc.
    for suf in (".csv.gz", ".tsv.gz", ".fa.gz", ".fasta.gz"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # then strip single extension if present
    stem = Path(stem).stem
    # normalize to a filesystem-safe token
    return re.sub(r"[^\w.-]+", "_", stem)


# Pipeline tags that should not persist between stages
_PIPELINE_TAGS: set[str] = {
    "peptides",
    "faiss",
    "ips",
    "ips_only",
    "ips_full",
    "domains",
    "domains_full",
    "features",
    "annotated",
    "scores",
    "final",
}

def _strip_pipeline_suffix(stem: str, tags: set[str] = _PIPELINE_TAGS) -> str:
    """
    Remove trailing pipeline tags from a stem.
    Handles single-token tags (e.g., 'peptides') and two-token tags
    (e.g., 'domains_full', 'ips_only'). Repeats until no match.
    """
    parts = stem.split("_")
    while parts:
        # try two-token suffix (e.g., domains_full, ips_only)
        if len(parts) >= 2 and f"{parts[-2]}_{parts[-1]}" in tags:
            parts = parts[:-2]
            continue
        # try single-token suffix
        if parts[-1] in tags:
            parts = parts[:-1]
            continue
        break
    return "_".join(parts)

def stage_path(out_dir: Path, input_name: str | Path, tag: str, ext: str) -> Path:
    """
    Compose a stage-specific output path under out_dir, always based on the
    original input stem (i.e., do not cascade previous stage tags).

    Example
    -------
    >>> stage_path(PEP_DIR, "my_gene_fusions.csv", "peptides", ".csv")
    output/interim/peptide_sequences/my_gene_fusions_peptides.csv

    >>> stage_path(FAISS_OUT, "my_gene_fusions_peptides.tsv", "domains_full", ".tsv")
    output/interim/faiss/my_gene_fusions_domains_full.tsv
    """
    base = basename(input_name)
    base = _strip_pipeline_suffix(base)  # drop any previous pipeline suffix(es)
    if not ext.startswith("."):
        ext = "." + ext
    return Path(out_dir) / f"{base}_{tag}{ext}"

def _assert_unique_id(df: pl.DataFrame, label: str):
    if "id" not in df.columns:
        raise ValueError(f"[{label}] missing 'id' column")
    if df["id"].n_unique() != df.height:
        raise ValueError(f"[{label}] 'id' must be unique per row")

def _join_new_columns(base: pl.DataFrame, inc: pl.DataFrame, label: str) -> pl.DataFrame:
    """Left-join only columns that are not already in base (excluding 'id')."""
    if not base.filter(base["id"].is_duplicated()).is_empty():
        base = base.unique(subset=["id"])
    _assert_unique_id(base, "BASE")
    _assert_unique_id(inc, label)
    new_cols = [c for c in inc.columns if c != "id" and c not in base.columns]
    if not new_cols:
        return base
    inc_sel = inc.select(["id"] + new_cols)
    return base.join(inc_sel, on="id", how="left")
