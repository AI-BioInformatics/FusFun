# qol/domain_retrieval/s03_prep_domain_windows.py
#!/usr/bin/env python3
"""
S03_prep_domain_windows.py
==========================
Stage S03 of the domain-retrieval pipeline.  
Generates amino-acid windows around InterProScan domain hits for downstream
embedding and indexing.

Overview
--------
Input:
  • InterPro merged TSV (`IPS_OUT/merged.tsv`) containing protein_id, start, end,
    signature_accession, ipr_id, etc.
  • Clean proteome FASTA (`IPS_PROTEOME`) with headers = protein IDs, '*'→'X'.

Processing:
  1) Load and filter InterPro hits by coordinate validity and length range.
  2) For each hit, extract the subsequence from the FASTA including left/right
     context (WINDOW_CTX_LEFT, WINDOW_CTX_RIGHT amino acids).
  3) Record coordinates and metadata for each window.

Output:
  • DOMAIN_WINDOWS (TSV): one row per domain window with:
        window_id, protein_id, aa_start, aa_end,
        window_start, window_end, window_len,
        signature_accession, ipr_id, analysis, sequence

Config keys used
----------------
IPS_OUT, IPS_PROTEOME, DOMAIN_WINDOWS,
WINDOW_CTX_LEFT, WINDOW_CTX_RIGHT,
MIN_LEN_AA, MAX_LEN_AA

Notes
-----
• Missing proteins in the FASTA are skipped with a warning.
• Coordinates are treated as 1-based inclusive, consistent with InterProScan.
• Window coordinates are clipped to sequence boundaries.
"""


from __future__ import annotations
from pathlib import Path
import sys
import polars as pl

from config.config import (
    IPS_OUT,
    IPS_PROTEOME,
    DOMAIN_WINDOWS,
    WINDOW_CTX_LEFT,
    WINDOW_CTX_RIGHT,
    MIN_LEN_AA,
    MAX_LEN_AA,
)

# ---------- FASTA ----------
def _fasta_iter(handle):
    """
    Yield (header, sequence) pairs from an open FASTA handle.

    Parameters
    ----------
    handle : io.TextIOBase
        Open text file object.

    Yields
    ------
    tuple[str, str]
        Header without '>' and full sequence concatenated across lines.

    Notes
    -----
    Blank lines are ignored. The function does not validate alphabet contents.
    """

    header, chunks = None, []
    for line in handle:
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)

def _strip_ver(x: str) -> str:
    """
    Remove version suffix from Ensembl-style identifiers.

    Examples
    --------
    'ENSP00000369497.2' → 'ENSP00000369497'
    """

    # keep stable ENSP without version if present
    return x.split(".", 1)[0]

def _load_fasta_dict(fa_path: Path) -> dict[str, str]:
    """
    Load a FASTA file into a dictionary mapping protein_id → sequence.

    Parameters
    ----------
    fa_path : Path
        Path to FASTA file whose headers contain protein IDs.

    Returns
    -------
    dict[str, str]
        Mapping of stable protein identifiers (version removed) to sequences.

    Raises
    ------
    FileNotFoundError
        If the FASTA file does not exist.
    """

    if not Path(fa_path).exists():
        raise FileNotFoundError(f"FASTA not found: {fa_path}")
    d: dict[str, str] = {}
    with open(fa_path, "r", encoding="utf-8") as fin:
        for hdr, seq in _fasta_iter(fin):
            ensp = _strip_ver(hdr.split()[0])
            d[ensp] = seq
    return d

# ---------- IO ----------
def _read_merged(path: Path) -> pl.DataFrame:
    """
    Load and filter the merged InterPro TSV.

    Parameters
    ----------
    path : Path
        Path to IPS_OUT/merged.tsv.

    Returns
    -------
    pl.DataFrame
        Filtered table containing at least: protein_id, start, end, feat_len.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.

    Behavior
    --------
    • Casts start/end to integers and removes invalid coordinates.
    • Adds feat_len = end − start + 1 and keeps hits within
    [MIN_LEN_AA, MAX_LEN_AA].
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"merged.tsv not found: {p}")
    df = pl.read_csv(
        p,
        separator="\t",
        infer_schema_length=0,
        null_values=["", ".", "-"],
        try_parse_dates=False,
        low_memory=True,
    )
    # expected columns (subset): protein_id, start, end, analysis, signature_accession, ipr_id
    req = ["protein_id", "start", "end"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"merged.tsv missing columns: {miss}")
    # numeric casts and bounds
    df = (
        df.with_columns([
            pl.col("start").cast(pl.Int64, strict=False),
            pl.col("end").cast(pl.Int64, strict=False),
        ])
        .filter(pl.col("start").is_not_null() & pl.col("end").is_not_null() & (pl.col("end") >= pl.col("start")))
        .with_columns((pl.col("end") - pl.col("start") + 1).alias("feat_len"))
        .filter((pl.col("feat_len") >= MIN_LEN_AA) & (pl.col("feat_len") <= MAX_LEN_AA))
    )
    return df

# ---------- core ----------
def _slice(seq: str, a1: int, b1: int, ctx_l: int, ctx_r: int) -> tuple[int, int, str]:
    """
    Extract a subsequence window with left/right context.

    Parameters
    ----------
    seq : str
        Full amino-acid sequence.
    a1, b1 : int
        1-based inclusive coordinates of the feature.
    ctx_l, ctx_r : int
        Number of residues to include before and after the feature.

    Returns
    -------
    tuple[int, int, str]
        (window_start_1b, window_end_1b, subsequence).
    """

    n = len(seq)
    ws = max(1, a1 - ctx_l)
    we = min(n, b1 + ctx_r)
    # convert to 0-based slicing
    return ws, we, seq[ws - 1 : we]

def run():
    """
    Main driver for Stage S03.

    Workflow
    --------
    1) Load merged InterPro features and FASTA sequences.
    2) For each valid hit:
    • locate sequence,
    • extract contextual window,
    • record identifiers and metadata.
    3) Write all windows to DOMAIN_WINDOWS as TSV.

    Outputs
    -------
    DOMAIN_WINDOWS : tab-separated file with coordinates and subsequences.

    Warnings
    --------
    Prints the count of proteins missing from the FASTA (if any).
    """

    merged_path = Path(IPS_OUT) / "merged.tsv"
    out_path = Path(DOMAIN_WINDOWS)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[S03] loading merged: {merged_path}")
    df = _read_merged(merged_path)

    print(f"[S03] loading proteome: {IPS_PROTEOME}")
    fa = _load_fasta_dict(Path(IPS_PROTEOME))

    ctx_l, ctx_r = WINDOW_CTX_LEFT, WINDOW_CTX_RIGHT

    rows = []
    missing_seq = 0
    for r in df.iter_rows(named=True):
        pid = _strip_ver(str(r["protein_id"]))
        seq = fa.get(pid)
        if not seq:
            missing_seq += 1
            continue
        a1 = int(r["start"])
        b1 = int(r["end"])
        ws, we, wseq = _slice(seq, a1, b1, ctx_l, ctx_r)
        sig = r.get("signature_accession") or "."
        ipr = r.get("ipr_id") or "."
        analysis = r.get("analysis") or "."
        ipr_name = (r.get("ipr_desc") or r.get("signature_description") or ".")
        window_id = f"{pid}:{a1}-{b1}:{sig}"

        rows.append({
            "window_id": window_id,
            "protein_id": pid,
            "aa_start": a1,
            "aa_end": b1,
            "window_start": ws,
            "window_end": we,
            "window_len": len(wseq),
            "signature_accession": sig,
            "ipr_id": ipr,
            "ipr_name": ipr_name,
            "analysis": analysis,
            "sequence": wseq,
        })

    if missing_seq:
        print(f"[S03][warn] proteins without sequence in FASTA: {missing_seq}", file=sys.stderr)

    out_df = pl.DataFrame(rows, schema={
        "window_id": pl.Utf8,
        "protein_id": pl.Utf8,
        "aa_start": pl.Int64,
        "aa_end": pl.Int64,
        "window_start": pl.Int64,
        "window_end": pl.Int64,
        "window_len": pl.Int64,
        "signature_accession": pl.Utf8,
        "ipr_id": pl.Utf8,
        "ipr_name": pl.Utf8,
        "analysis": pl.Utf8,
        "sequence": pl.Utf8,
    })

    out_df.write_csv(out_path, separator="\t")
    print(f"[S03] windows → {out_path}  rows={out_df.height}")

if __name__ == "__main__":
    run()
