# qol/domain_retrieval/S01_extract_domains.py
#!/usr/bin/env python3
"""
S01_extract_domains.py
======================
Stage S01 of the domain-retrieval pipeline: prepare an InterPro-ready proteome,
optionally split it into chunks, and run InterProScan.

Overview
--------
This stage performs three sequential tasks driven entirely by `config.config`:
1) Create InterPro-ready inputs
   • Write a cleaned proteome FASTA (`IPS_PROTEOME`) with headers set to protein IDs
     (version suffix removed) and '*' replaced by 'X'.
   • Write a protein lookup TSV (`ENSP_LOOKUP`) mapping protein_id → peptide_sequence.
2) Optionally chunk the cleaned FASTA into `PROTEOME_CHUNKS` balanced parts
   under `IPS_CHUNK_DIR` (controlled by `SPLIT_PROTEOME`).
3) Run InterProScan either on the whole cleaned FASTA or on each chunk, producing
   TSV outputs in `IPS_OUT` (single FASTA) or `IPS_CHUNK_DIR` (chunked mode).

Config keys used
----------------
FA_ENSEMBL, IPS_PROTEOME, ENSP_LOOKUP,
SPLIT_PROTEOME, PROTEOME_CHUNKS,
IPS_BIN, IPS_APPL, IPS_THREADS, IPS_TMPDIR,
IPS_OUT, IPS_CHUNK_DIR

Outputs
-------
• Clean FASTA  : IPS_PROTEOME
• Protein TSV  : ENSP_LOOKUP  (protein_id, peptide_sequence)
• InterPro TSV : one TSV per input FASTA (either in IPS_OUT or IPS_CHUNK_DIR)

Notes
-----
• This module performs no deduplication of protein IDs; it assumes Ensembl-style
  FASTA headers and uses the first token (version-stripped) as the protein_id.
• InterProScan is invoked via `subprocess.run()` and will raise on non-zero exit.
"""

from __future__ import annotations
from pathlib import Path
import os, sys, subprocess

from config.config import (
    FA_ENSEMBL,        # input proteome FASTA
    ENSP_LOOKUP,       # output TSV: protein_id -> peptide
    IPS_PROTEOME,      # output cleaned FASTA for InterPro
    SPLIT_PROTEOME,    # bool
    PROTEOME_CHUNKS,   # int
    IPS_BIN,           # path to interproscan.sh
    IPS_APPL,          # comma list of member DBs
    IPS_THREADS,       # CPU threads
    IPS_TMPDIR,        # temp dir for InterProScan
    IPS_OUT,           # base output dir for InterPro TSVs
    IPS_CHUNK_DIR,     # chunk subdir for TSVs when split
)

# ---------- FASTA utils ----------
def _fasta_iter(handle):
    """
    Stream FASTA records from an open text handle.

    Parameters
    ----------
    handle : io.TextIOBase
        Open text file object positioned at the beginning of a FASTA file.

    Yields
    ------
    tuple[str, str]
        (header_without_gt, sequence) for each record. The leading '>' is removed
        and sequence lines are concatenated without newlines.

    Notes
    -----
    Blank lines are ignored. The function does not validate FASTA content beyond
    the presence of '>' record headers.
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

def _extract_ensp_id(raw_header: str) -> str:
    """
    Extract the protein identifier from a FASTA header and strip version suffix.

    Parameters
    ----------
    raw_header : str
        Full FASTA header text (without the leading '>').

    Returns
    -------
    str
        The first whitespace-separated token with any trailing '.<version>' removed.
    """

    first_tok = raw_header.split()[0]
    return first_tok.split(".")[0]

# ---------- S01: prepare InterPro-ready files ----------
def create_ready_files():
    """
    Create InterPro-ready inputs: cleaned proteome FASTA and protein lookup TSV.

    Side Effects
    ------------
    • Writes `IPS_PROTEOME` with headers set to protein IDs and '*' → 'X'.
    • Writes `ENSP_LOOKUP` with columns: protein_id, peptide_sequence.
    • Prints a short summary with output paths and record count.

    Raises
    ------
    OSError / IOError
        If input `FA_ENSEMBL` cannot be read or outputs cannot be written.

    Notes
    -----
    Relies on `_fasta_iter` and `_extract_ensp_id`. Output directories are created
    as needed.
    """

    IPS_PROTEOME.parent.mkdir(parents=True, exist_ok=True)
    ENSP_LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    n_seq = 0

    with open(FA_ENSEMBL, "r", encoding="utf-8") as fin, \
         open(IPS_PROTEOME, "w", encoding="utf-8") as fo_fa, \
         open(ENSP_LOOKUP, "w", encoding="utf-8") as fo_tsv:
        fo_tsv.write("protein_id\tpeptide_sequence\n")
        for hdr, seq in _fasta_iter(fin):
            ensp = _extract_ensp_id(hdr)
            pep = seq.replace("*", "X")
            fo_fa.write(f">{ensp}\n{pep}\n")
            fo_tsv.write(f"{ensp}\t{pep}\n")
            n_seq += 1

    print(f"[S01] FASTA ready: {IPS_PROTEOME}")
    print(f"[S01] ENSP lookup: {ENSP_LOOKUP}")
    print(f"[S01] Sequences  : {n_seq}")

# ---------- S02: optional chunking ----------
def _stream_records(path: Path):
    """
    Iterate over a FASTA file on disk, yielding (record_id, sequence).

    Parameters
    ----------
    path : Path
        Path to a FASTA file whose headers encode the protein ID as the first token.

    Yields
    ------
    tuple[str, str]
        (protein_id_without_version, sequence) for each record.

    Notes
    -----
    This is similar to `_fasta_iter` but opens the file internally and normalizes
    the record ID to the first token of the header with version removed.
    """

    with open(path, "r", encoding="utf-8") as f:
        rec_id, seq = None, []
        for line in f:
            if line.startswith(">"):
                if rec_id is not None:
                    yield rec_id, "".join(seq)
                rec_id = line.strip().split()[0][1:].split(".")[0]
                seq = []
            else:
                seq.append(line.strip())
        if rec_id is not None:
            yield rec_id, "".join(seq)

def chunk_fasta_proteome():
    """
    Split the cleaned proteome FASTA into `PROTEOME_CHUNKS` balanced files.

    Behavior
    --------
    • Counts sequences in `IPS_PROTEOME` and computes a per-chunk target size.
    • Writes chunk files as `proteome_chunk_###.fa` into `IPS_CHUNK_DIR`,
    wrapping sequences at 60 characters per line for readability.
    • Prints a summary with total sequences and ~per-chunk count.

    Side Effects
    ------------
    Creates `IPS_CHUNK_DIR` and the chunk FASTA files.

    Raises
    ------
    OSError / IOError
        If the source FASTA cannot be read or chunk files cannot be written.
    """

    IPS_CHUNK_DIR.parent.mkdir(parents=True, exist_ok=True)  # ensure base exists
    IPS_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(1 for _ in _stream_records(IPS_PROTEOME))
    per_chunk = max(1, (total + PROTEOME_CHUNKS - 1) // PROTEOME_CHUNKS)

    writers = [open(IPS_CHUNK_DIR / f"proteome_chunk_{i:03d}.fa", "w", encoding="utf-8")
               for i in range(PROTEOME_CHUNKS)]
    try:
        for idx, (rid, seq) in enumerate(_stream_records(IPS_PROTEOME)):
            b = min(idx // per_chunk, PROTEOME_CHUNKS - 1)
            w = writers[b]
            w.write(f">{rid}\n")
            for s in range(0, len(seq), 60):
                w.write(seq[s:s+60] + "\n")
    finally:
        for w in writers:
            w.close()

    print(f"[S02] Wrote {PROTEOME_CHUNKS} chunks → {IPS_CHUNK_DIR}")
    print(f"[S02] Total sequences: {total} (~{per_chunk} per chunk)")

# ---------- S03: InterProScan ----------
def _run_interpro_on_file(in_fasta: Path, out_dir: Path,
                          apps: str, threads: int, tempdir: Path | None):
    """
    Run InterProScan on a single FASTA and write a TSV result.

    Parameters
    ----------
    in_fasta : Path
        Input FASTA file to analyze.
    out_dir : Path
        Destination directory for the InterPro TSV (created if needed).
    apps : str
        Comma-separated list of member databases to run (e.g., 'Pfam,SMART').
    threads : int
        Number of CPU threads to pass to InterProScan.
    tempdir : Path | None
        Optional scratch directory for InterProScan ('-T' flag). Created if provided.

    Side Effects
    ------------
    Invokes the `IPS_BIN` command via `subprocess.run()` with `check=True`.
    Writes `<in_fasta.stem>.tsv` into `out_dir`.

    Raises
    ------
    SystemExit
        If `IPS_BIN` does not exist (exit code 2).
    subprocess.CalledProcessError
        If InterProScan exits with non-zero status.
    """

    if not Path(IPS_BIN).exists():
        print(f"[S03][error] InterProScan not found: {IPS_BIN}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_fasta.stem  # proteome_for_interpro or proteome_chunk_###
    outfile = out_dir / f"{stem}.tsv"

    cmd = [
        str(IPS_BIN),
        "-i", str(in_fasta),
        "-f", "tsv",
        "-o", str(outfile),
        "-cpu", str(threads),
        "-appl", apps,
        "-dp",
    ]
    if tempdir:
        tempdir.mkdir(parents=True, exist_ok=True)
        cmd += ["-T", str(tempdir)]

    print(f"[S03] Running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    print(f"[S03] TSV: {outfile}")

def run_interpro_all():
    """
    Run InterProScan either on all chunks (if enabled) or on the single cleaned FASTA.

    Behavior
    --------
    • If `SPLIT_PROTEOME` is True:
        - Look for chunk files in `IPS_CHUNK_DIR`. If none exist, fall back to single FASTA.
        - Otherwise, run `_run_interpro_on_file` on each chunk and write TSVs into `IPS_CHUNK_DIR`.
    • If `SPLIT_PROTEOME` is False:
        - Run `_run_interpro_on_file` once on `IPS_PROTEOME` and write TSV into `IPS_OUT`.

    Notes
    -----
    This function does not create chunks; call `chunk_fasta_proteome()` beforehand
    when `SPLIT_PROTEOME` is True.
    """

    if SPLIT_PROTEOME:
        IPS_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
        chunk_files = sorted(IPS_CHUNK_DIR.glob("proteome_chunk_*.fa"))
        if not chunk_files:
            print(f"[S03][warn] No chunks found in {IPS_CHUNK_DIR}. Falling back to single FASTA.")
            _run_interpro_on_file(IPS_PROTEOME, IPS_OUT, IPS_APPL, IPS_THREADS, IPS_TMPDIR)
            return
        for f in chunk_files:
            _run_interpro_on_file(f, IPS_CHUNK_DIR, IPS_APPL, IPS_THREADS, IPS_TMPDIR)
    else:
        _run_interpro_on_file(IPS_PROTEOME, IPS_OUT, IPS_APPL, IPS_THREADS, IPS_TMPDIR)

# ---------- driver ----------
def main():
    create_ready_files()
    if SPLIT_PROTEOME:
        chunk_fasta_proteome()
    run_interpro_all()
