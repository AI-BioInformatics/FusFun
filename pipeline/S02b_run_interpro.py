#!/usr/bin/env python3
"""
S02_run_interpro.py
===================
Stage 02b — Run InterProScan on peptide fragments from Stage 01.

Overview
--------
This stage calls InterProScan (IPS) on four peptide columns produced in S01
(retained_peptide1, lost_peptide1, retained_peptide2, lost_peptide2). It writes
per-column InterPro accession lists back to disk as both an "IPS-only" table and
a "full" table joined to all S01 metadata.

For each peptide column it now produces:
  • <side>_domains  (InterPro IDs; renamed to retained/lost_domains1/2)
  • <side>_domain_names (InterPro descriptions; renamed accordingly)

Inputs
------
• Stage-01 peptide table with columns:
  id, retained_peptide1, lost_peptide1, retained_peptide2, lost_peptide2
• Configuration from config.config:
  IPS_BIN, IPS_APPL, IPS_THREADS, IPS_BATCH_SIZE, IPS_OUT

Method
------
1) Filter out empty/placeholder sequences.
2) Batch sequences (size = IPS_BATCH_SIZE) and write a temporary FASTA.
   FASTA headers encode the source row and column as: >{id}|{col_name}
3) Invoke InterProScan with TSV output and parse:
   - column 12 (InterPro accession)
   - column 13 (InterPro description)
4) For each peptide cell, aggregate unique InterPro accessions and aligned
   descriptions in input order.
5) Emit two files:
   • ips_only:  id + four *_domains and four *_domain_names columns
   • ips_full:  S01 metadata joined with those eight columns

Outputs
-------
• <IPS_OUT>/<input_stem>_ips_only.tsv
• <IPS_OUT>/<input_stem>_ips_full.tsv

Notes
-----
• Empty or '.' sequences are skipped. Missing results are recorded as '.'.
• Temporary working directory is created under IPS_OUT/tmp per batch.
• All tunables and paths are read exclusively from config.config.
"""

from __future__ import annotations
from pathlib import Path
import os, csv, subprocess, tempfile
from typing import Iterable, List, Dict, Tuple
import polars as pl

# --- config (single source of truth) ---
from config.config import (
    IPS_BIN,
    IPS_APPL,
    IPS_THREADS,
    IPS_BATCH_SIZE,
    IPS_OUT
)

# --- utilities (framework helpers) ---
from qol.utilities import load_file, save_file, stage_path

# --- schema ---
PEP_COLS = [
    "retained_peptide1", "lost_peptide1",
    "retained_peptide2", "lost_peptide2",
]
REQUIRED = ["id"] + PEP_COLS


# --- internals ---
def _non_empty_mask(s: pl.Series) -> pl.Series:
    """
    Return a boolean mask selecting non-empty peptide strings.

    A value is considered empty if it is null, equal to ".", or has zero length.

    Parameters
    ----------
    s : pl.Series
        UTF-8 string series.

    Returns
    -------
    pl.Series
        Boolean mask with True for valid, non-empty sequences.
    """

    return (~s.is_null()) & (s != ".") & (s.str.len_chars() > 0)


def _write_fasta(rows: Iterable[Tuple[int, str, str]], out_fa: Path) -> None:
    """
    Write peptide records to FASTA with headers encoding row id and column name.

    FASTA header format: >{row_id}|{col_name}
    Sequence is written verbatim on the next line.

    Parameters
    ----------
    rows : Iterable[Tuple[int, str, str]]
        Triples of (row_id, column_name, sequence).
    out_fa : Path
        Destination FASTA path.

    Returns
    -------
    None
    """

    # rows: (row_id, col_name, sequence)
    with open(out_fa, "w") as f:
        for rid, col, seq in rows:
            f.write(f">{rid}|{col}\n{seq}\n")


def _run_interproscan(
    interpro_bin: Path,
    fasta_path: Path,
    out_tsv: Path,
    appl: str,
    n_threads: int,
) -> None:
    """
    Invoke InterProScan on a FASTA file and write TSV results.

    Parameters
    ----------
    interpro_bin : Path
        Executable path to InterProScan (IPS_BIN).
    fasta_path : Path
        Input FASTA file.
    out_tsv : Path
        Output TSV path produced by InterProScan.
    appl : str
        Comma-separated application list (INTERPRO_APPL), e.g. "Pfam,SMART".
    n_threads : int
        CPU threads for InterProScan (INTERPRO_THREADS).

    Returns
    -------
    None

    Notes
    -----
    • Uses '-f TSV -dp -cpu {n_threads} -appl {appl}'.
    • A temporary working directory under IPS_OUT/tmp is created per call.
    • subprocess.run(..., check=True) raises on non-zero exit.
    """

    tmp_dir = out_tsv.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(interpro_bin),
        "-i", str(fasta_path),
        "-f", "TSV",
        "-appl", appl,
        "-dp",
        "-T", str(tmp_dir),
        "-cpu", str(n_threads),
        "-o", str(out_tsv),
    ]
    subprocess.run(cmd, check=True)


def _parse_interpro_tsv(tsv_path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse an InterProScan TSV and extract
    (protein_accession, interpro_accession, interpro_description).

    Parameters
    ----------
    tsv_path : Path
        Path to InterProScan TSV output.

    Returns
    -------
    List[Tuple[str, str, str]]
        Triples of:
        • protein_accession   — FASTA header as echoed by IPS
                                (e.g., "123|retained_peptide1")
        • interpro_accession  — InterPro identifier (column 12)
        • interpro_description — InterPro description (column 13)

    Notes
    -----
    • Lines starting with '#' are ignored.
    • If the file is missing or empty, an empty list is returned.
    • Column indices follow the standard IPS TSV layout where:
        - InterPro accession is at index 11 (1-based column 12)
        - InterPro description is at index 12 (1-based column 13)
    """

    out: List[Tuple[str, str, str]] = []
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return out
    with open(tsv_path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        for cols in r:
            if not cols or cols[0].startswith("#"):
                continue
            protein = cols[0] if len(cols) > 0 else ""
            ipr = cols[11] if len(cols) > 11 else ""
            desc = cols[12] if len(cols) > 12 else ""
            if protein and ipr:
                out.append((protein, ipr, desc))
    return out


def _run_ips_on_columns(
    df: pl.DataFrame,
    pep_cols: List[str],
    interpro_bin: Path,
    appl: str,
    threads: int,
    batch_size: int,
) -> pl.DataFrame:
    """
    Run InterProScan over selected peptide columns and append
    *_iprs and *_names results.

    Parameters
    ----------
    df : pl.DataFrame
        Input table containing 'id' and the peptide columns.
    pep_cols : List[str]
        Columns to analyze (e.g., PEP_COLS).
    interpro_bin : Path
        Executable path to InterProScan.
    appl : str
        Comma-separated application list (INTERPRO_APPL).
    threads : int
        CPU threads for InterProScan.
    batch_size : int
        Number of peptide entries per IPS batch.

    Returns
    -------
    pl.DataFrame
        A copy of the input with two added columns per peptide column:
        '{col}_iprs'  — ' | '-separated InterPro accessions,
        '{col}_names' — ' | '-separated InterPro descriptions,
        each set to '.' when no hits are found.

    Behavior
    --------
    • Skips null/'.'/empty sequences.
    • Writes a temporary FASTA per batch using headers '>id|col'.
    • Aggregates accessions and descriptions per row id while preserving
      first-seen order and uniqueness by InterPro ID.
    • Fills missing *_iprs and *_names with '.' to keep schema stable.
    """

    out_df = df.clone()

    for col in pep_cols:
        sub = df.select(["id", col]).with_columns(pl.col(col).cast(pl.Utf8))
        sub = sub.filter(_non_empty_mask(pl.col(col)))

        if sub.is_empty():
            out_df = out_df.with_columns([
                pl.lit(".").alias(f"{col}_iprs"),
                pl.lit(".").alias(f"{col}_names"),
            ])
            continue

        ids = sub["id"].to_list()
        seqs = sub[col].to_list()
        ipr_map: Dict[int, str] = {}
        name_map: Dict[int, str] = {}

        for start in range(0, len(ids), batch_size):
            end = min(len(ids), start + batch_size)
            batch_ids = ids[start:end]
            batch_seqs = seqs[start:end]

            with tempfile.TemporaryDirectory() as td:
                fa = Path(td) / "q.fa"
                tsv = Path(td) / "out.tsv"
                pairs = [(int(rid), col, s) for rid, s in zip(batch_ids, batch_seqs)]
                _write_fasta(pairs, fa)
                _run_interproscan(interpro_bin, fa, tsv, appl, threads)
                rows = _parse_interpro_tsv(tsv)

            # bucket by row id using header "rid|col"
            bucket: Dict[int, List[Tuple[str, str]]] = {}
            for protein, ipr, desc in rows:
                try:
                    rid_str, c_name = protein.split("|", 1)
                    if c_name != col:
                        continue
                    rid = int(rid_str)
                except Exception:
                    continue
                bucket.setdefault(rid, []).append((ipr, desc))

            for rid in batch_ids:
                pairs_rd = bucket.get(int(rid), [])
                seen = set()
                iprs: List[str] = []
                names: List[str] = []
                for ipr, desc in pairs_rd:
                    if ipr in seen:
                        continue
                    seen.add(ipr)
                    iprs.append(ipr)
                    names.append(desc or "")
                ipr_map[int(rid)] = " | ".join(iprs) if iprs else "."
                name_map[int(rid)] = " | ".join(names) if names else "."

        out_df = (
            out_df.join(
                pl.DataFrame({
                    "id": list(ipr_map.keys()),
                    f"{col}_iprs": [ipr_map[k] for k in ipr_map.keys()],
                    f"{col}_names": [name_map[k] for k in name_map.keys()],
                }),
                on="id",
                how="left",
            )
            .with_columns([
                pl.col(f"{col}_iprs").fill_null("."),
                pl.col(f"{col}_names").fill_null("."),
            ])
        )

    return out_df


# --- entry point ---
def run(input_file: Path | str) -> Path:
    """
    Execute Stage 02b end-to-end on a Stage-01 peptide table.

    Parameters
    ----------
    input_file : Path | str
        Path to the S01 peptide table.

    Returns
    -------
    Path
        Path to the generated ips_full TSV.

    Side Effects
    ------------
    Writes two files under IPS_OUT:
    • '<input_stem>_ips_only.tsv' with:
        id +
        retained_domains1, lost_domains1, retained_domains2, lost_domains2 +
        retained_domain_names1, lost_domain_names1,
        retained_domain_names2, lost_domain_names2
    • '<input_stem>_ips_full.tsv' with all S01 metadata + the same 8 columns.
    """

    # Load full Stage-01 table to preserve metadata
    full_df = load_file(input_file)

    # Slice to peptides only for IPS
    seq_df = full_df.select([
        "id",
        "retained_peptide1", "lost_peptide1",
        "retained_peptide2", "lost_peptide2",
    ]).with_columns([
        pl.col("retained_peptide1").fill_null("."),
        pl.col("lost_peptide1").fill_null("."),
        pl.col("retained_peptide2").fill_null("."),
        pl.col("lost_peptide2").fill_null("."),
    ])

    out_df = _run_ips_on_columns(
        df=seq_df,
        pep_cols=PEP_COLS,
        interpro_bin=Path(IPS_BIN),
        appl=str(IPS_APPL),
        threads=int(IPS_THREADS),
        batch_size=int(IPS_BATCH_SIZE),
    )

    # 1) ips-only file: id + four domain ID columns + four domain name columns
    ips_only = (
        out_df
        .select([
            pl.col("id"),

            # IDs
            pl.col("retained_peptide1_iprs").alias("retained_domains1"),
            pl.col("lost_peptide1_iprs").alias("lost_domains1"),
            pl.col("retained_peptide2_iprs").alias("retained_domains2"),
            pl.col("lost_peptide2_iprs").alias("lost_domains2"),

            # Names
            pl.col("retained_peptide1_names").alias("retained_domain_names1"),
            pl.col("lost_peptide1_names").alias("lost_domain_names1"),
            pl.col("retained_peptide2_names").alias("retained_domain_names2"),
            pl.col("lost_peptide2_names").alias("lost_domain_names2"),
        ])
    )

    # 2) full file = all Stage-01 columns + the eight domain columns
    ips_full = full_df.join(ips_only, on="id", how="left")

    Path(IPS_OUT).mkdir(parents=True, exist_ok=True)
    ips_only_path = stage_path(IPS_OUT, input_file, "ips_only", ".tsv")
    ips_full_path = stage_path(IPS_OUT, input_file, "ips_full", ".tsv")

    save_file(ips_only, ips_only_path)
    save_file(ips_full, ips_full_path)

    print(f"[S02] Saved extracted domains: {ips_only_path}")
    return ips_full_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: S02_run_interpro.py <input_tsv>")
    run(Path(sys.argv[1]))
