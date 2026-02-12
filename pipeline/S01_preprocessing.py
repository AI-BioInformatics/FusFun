"""
S01_preprocessing.py
====================
Stage 01 — Preprocessing of gene fusion metadata to generate peptide fragments.

Overview
--------
This stage reconstructs the peptide fragments corresponding to the retained and lost
regions of each fusion partner, based on transcript-level breakpoints.

Given:
  • A fusion metadata table containing gene names, transcript IDs, breakpoints, and strands.  
  • A peptide lookup table mapping `transcript_id` → amino acid sequence.  
  • A CDS lookup table providing start, end, and cumulative coding sequence (CDS) lengths.  

It produces:
  • A TSV/CSV file containing retained and lost peptide fragments for each partner.  
  • A FASTA file with all fragments for downstream analyses (e.g., InterProScan, FAISS retrieval).

Pipeline Steps
--------------
1. Load input metadata and lookup tables using the unified I/O utilities.
2. Attach peptide sequences to both fusion partners.
3. For each partner, map breakpoints to CDS coordinates and determine the amino acid index.
4. Split the peptide sequence into the retained (before breakpoint) and lost (after breakpoint) fragments.
5. Save the resulting table and corresponding FASTA file.

Outputs
-------
• `<output>/peptides_<input>.tsv` — retained/lost fragments per partner.
• `<output>/peptides_<input>.fasta` — FASTA-formatted fragment sequences.

Notes
-----
• Stop codons ('*') are replaced with 'X'.
• Missing values are filled with '.' to ensure compatibility with downstream tools.
"""


from __future__ import annotations
from pathlib import Path
import polars as pl

from qol.utilities import load_file, save_file, add_sequential_id, stage_path
from config.config import (
    INPUT_DIR, REQUIRED_COLUMNS,
    PEPTIDE_LOOKUP, CDS_LOOKUP, PEP_DIR
)

# ---------------------------------------------------------------------
def _search_for_transcript(df: pl.DataFrame, lookup_table: pl.DataFrame) -> pl.DataFrame:
    """
    Attach peptide sequences from lookup to both fusion partners.

    Parameters
    ----------
    df : pl.DataFrame
        Input table containing transcript_id1 and transcript_id2.
    lookup_table : pl.DataFrame
        Table mapping transcript_id → peptide_sequence.

    Returns
    -------
    pl.DataFrame
        DataFrame with two new columns:
        - peptide_sequence1
        - peptide_sequence2
        with '*' replaced by 'X' and missing values filled with '.'.
    """
    df = df.join(
        lookup_table.rename({"peptide_sequence": "peptide_sequence1"}),
        left_on="transcript_id1", right_on="transcript_id", how="left"
    ).with_columns(pl.col("peptide_sequence1").fill_null("."))
    df = df.join(
        lookup_table.rename({"peptide_sequence": "peptide_sequence2"}),
        left_on="transcript_id2", right_on="transcript_id", how="left"
    ).with_columns(pl.col("peptide_sequence2").fill_null("."))
    return df.with_columns([
        pl.col("peptide_sequence1").str.replace_all(r"\*", "X"),
        pl.col("peptide_sequence2").str.replace_all(r"\*", "X"),
    ])


def _choose_strand_col(df: pl.DataFrame, pos: int) -> str:
    """
    Select the correct strand column for each fusion partner.

    Parameters
    ----------
    df : pl.DataFrame
        Input fusion table.
    pos : int
        Partner index (1 or 2).

    Returns
    -------
    str
        Column name to use ('gene_strand{pos}' or 'fusion_strand{pos}').

    Raises
    ------
    SystemExit
        If neither column exists.
    """
    cand1, cand2 = f"gene_strand{pos}", f"fusion_strand{pos}"
    if cand1 in df.columns:
        return cand1
    if cand2 in df.columns:
        return cand2
    raise SystemExit(f"Missing strand column for partner {pos} (need {cand1} or {cand2}).")


def _split_pep(df: pl.DataFrame, gtf: pl.DataFrame, pos: int) -> pl.DataFrame:
    """
    Compute amino acid index from breakpoint and split peptide into retained/lost parts.

    Logic
    -----
    1. Join with CDS lookup to map each transcript to its CDS region.
    2. For each breakpoint, compute nucleotide offset within the CDS.
    3. Convert nucleotide offset to amino acid index.
    4. Slice peptide_sequence{pos} accordingly.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion metadata with peptide_sequence{pos}.
    gtf : pl.DataFrame
        CDS lookup table containing start, end, cumulative_length, length.
    pos : int
        Partner index (1 or 2).

    Returns
    -------
    pl.DataFrame
        Same columns as input with added:
        - retained_peptide{pos}
        - lost_peptide{pos}
    """
    strand_col = _choose_strand_col(df, pos)
    bkp_col = f"breakpoint{pos}"
    j = (
        df.join(gtf, left_on=f"transcript_id{pos}", right_on="transcript_id", how="left")
          .filter((pl.col("start") <= pl.col(bkp_col)) & (pl.col("end") >= pl.col(bkp_col)))
          .with_columns(
              pl.when(pl.col(strand_col) == "+")
                .then(pl.col(bkp_col) - pl.col("start"))
                .otherwise(pl.col("end") - pl.col(bkp_col))
                .alias("ret_nc_cds")
          )
          .with_columns(
              ((pl.col("cumulative_length") - pl.col("length") + pl.col("ret_nc_cds")) // 3)
              .alias("aa_index")
          )
          .select(["gene1", "gene2", f"transcript_id{pos}", bkp_col, "aa_index"])
          .unique()
    )
    out = df.join(j, on=["gene1", "gene2", f"transcript_id{pos}", bkp_col], how="left")
    pep_col = f"peptide_sequence{pos}"
    out = out.with_columns([
        pl.when(pl.col("aa_index").is_null() | (pl.col("aa_index") < 0))
          .then(pl.lit("."))
          .otherwise(pl.col(pep_col).str.slice(0, pl.col("aa_index") + 1))
          .alias(f"retained_peptide{pos}"),
        pl.when(pl.col("aa_index").is_null() | (pl.col("aa_index") < 0))
          .then(pl.lit("."))
          .otherwise(pl.col(pep_col).str.slice(pl.col("aa_index") + 1))
          .alias(f"lost_peptide{pos}")
    ]).with_columns([
        pl.col(f"retained_peptide{pos}").fill_null("."),
        pl.col(f"lost_peptide{pos}").fill_null("."),
    ]).drop(["aa_index", pep_col])
    return out


def _write_fasta(peps_csv: Path, fasta_out: Path) -> Path:
    """
    Export retained and lost peptide fragments into a FASTA file.

    Parameters
    ----------
    peps_csv : Path
        CSV produced by build_fusion_peptides().
    fasta_out : Path
        Destination FASTA file.

    Returns
    -------
    Path
        Path to generated FASTA file.
    """
    df = load_file(peps_csv, required=[
        "id","retained_peptide1","lost_peptide1","retained_peptide2","lost_peptide2"
    ])
    fasta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_out, "w", encoding="utf-8") as f:
        for row in df.iter_rows(named=True):
            rid = row["id"]
            for k in ("retained_peptide1","lost_peptide1","retained_peptide2","lost_peptide2"):
                seq = row[k]
                if seq and seq != ".":
                    f.write(f">{rid}|side={k}|len={len(seq)}\n{seq}\n")
    return fasta_out


def build_fusion_peptides(
    input_file: Path,
    pep_lookup_path: Path,
    cds_lookup_path: Path,
    out_path: Path,
) -> Path:
    """
    Main builder combining metadata and lookup data into peptide fragments.

    Parameters
    ----------
    input_file : Path
        Fusion metadata table (CSV/TSV).
    pep_lookup_path : Path
        Path to peptide lookup with transcript_id → sequence.
    cds_lookup_path : Path
        Path to CDS lookup with positional and cumulative length data.
    out_path : Path
        Destination CSV file.

    Returns
    -------
    Path
        Path to output CSV with peptide fragments.
    """
    fasta_lookup = load_file(pep_lookup_path, required=["transcript_id", "peptide_sequence"])
    cds_lookup   = load_file(cds_lookup_path, required=["transcript_id","start","end","cumulative_length","length"])
    df           = load_file(input_file, required=REQUIRED_COLUMNS)

    df = add_sequential_id(df, "id")
    df = _search_for_transcript(df, fasta_lookup)
    df = _split_pep(df, cds_lookup, 1)
    df = _split_pep(df, cds_lookup, 2)

    # sanitize peptide sequences but keep all metadata columns
    for c in ("retained_peptide1","lost_peptide1","retained_peptide2","lost_peptide2"):
        if c in df.columns:
            df = df.with_columns(
                pl.col(c)
                  .cast(pl.Utf8)
                  .str.replace_all('"', ".")
                  .str.replace_all(r"\*", "X")
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(df, out_path)
    return df



def run(input_name: str) -> tuple[Path, Path]:
    """
    Stage entry point orchestrating preprocessing and output generation.

    Parameters
    ----------
    input_name : str
        Input filename under `input/` or absolute path.

    Returns
    -------
    tuple[Path, Path]
        (peptides_csv_path, peptides_fasta_path)
    """
    in_path = Path(input_name)
    if not in_path.exists():
        in_path = Path(INPUT_DIR) / input_name
    if not in_path.exists():
        raise SystemExit(f"Input not found: {input_name}")

    peps_csv = stage_path(PEP_DIR, in_path, "peptides", ".tsv")
    peps_fas = stage_path(PEP_DIR, in_path, "peptides", ".fasta")

    peps_csv.parent.mkdir(parents=True, exist_ok=True)

    build_fusion_peptides(
        input_file=in_path,
        pep_lookup_path=Path(PEPTIDE_LOOKUP),
        cds_lookup_path=Path(CDS_LOOKUP),
        out_path=peps_csv,
    )
    _write_fasta(peps_csv, peps_fas)
    return peps_csv, peps_fas


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: S01_preprocessing.py <input_csv>")
    run(sys.argv[1])
