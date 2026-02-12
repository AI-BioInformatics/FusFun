# qol/rebuild_fasta_lookup.py
""" build_fasta_lookup.py 
===================== 
Rebuild FASTA-derived lookup tables and a cleaned proteome FASTA. 

Overview 
-------- 
Parses an Ensembl proteome FASTA and produces: 
1) Transcript lookup (PEPTIDE_LOOKUP): transcript_id → peptide_sequence (+ protein_id) 
2) Protein lookup (ENSP_LOOKUP): protein_id → peptide_sequence 
3) Clean FASTA for InterProScan: headers are protein IDs, optionally replacing stop codons '*' with 'X' (default behavior) 
Header conventions
------------------ 
The script expects transcript identifiers to appear in the FASTA header as 'transcript:<ENST...>' 
(e.g., Ensembl conventions). Protein IDs are taken as the first whitespace-separated token of the header line. 
Version suffixes ('.1', '.2') are stripped from both. 

Inputs 
------
• FA_ENSEMBL: path to the proteome FASTA (.fa or .fa.gz) 
• config paths for outputs (PEPTIDE_LOOKUP, ENSP_LOOKUP, cleaned FASTA path) 

Outputs 
------- 
• TSV: transcript_id, peptide_sequence, protein_id 
• TSV: protein_id, peptide_sequence 
• FASTA: protein_id as header, peptide sequence (with or without '*') 
Notes 
----- 
• If a transcript tag is missing in a header, the protein record is still written, but the transcript lookup entry is skipped and reported at the end. 
• Reading supports both plain text and gzip-compressed FASTA files. 

"""

from __future__ import annotations
from pathlib import Path
import argparse, io, gzip, re, sys

from config.config import (
    ENSEMBL,      # directory for Ensembl-related outputs
    FA_ENSEMBL,   # input proteome FASTA (.fa or .fa.gz)
    LOOKUPS
)

HDR_TR_RE = re.compile(r"\btranscript:([A-Za-z0-9_.-]+)")

def _strip_ver(x: str) -> str:
    return x.split(".", 1)[0] if x else x

def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def _fasta_iter(handle):
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

def _fail_if_exists(*paths: Path):
    clashes = [p for p in paths if p.exists()]
    if clashes:
        msg = "Refusing to overwrite existing files:\n" + "\n".join(str(p) for p in clashes)
        raise SystemExit(msg)

def build_from_fasta(in_fa: Path, out_tr: Path, out_prot: Path, out_clean_fa: Path):
    out_tr.parent.mkdir(parents=True, exist_ok=True)
    out_prot.parent.mkdir(parents=True, exist_ok=True)
    out_clean_fa.parent.mkdir(parents=True, exist_ok=True)
    keep_stop = False

    with open(out_tr, "w", encoding="utf-8") as tr_w, \
         open(out_prot, "w", encoding="utf-8") as pr_w, \
         open(out_clean_fa, "w", encoding="utf-8") as fa_w, \
         _open_text(in_fa) as fin:

        tr_w.write("transcript_id\tpeptide_sequence\tprotein_id\n")
        pr_w.write("protein_id\tpeptide_sequence\n")

        n_total = n_tr = n_pr = n_fa = n_tr_miss = 0
        for hdr, seq in _fasta_iter(fin):
            n_total += 1
            ensp = _strip_ver(hdr.split()[0])  # first token
            pep = seq if keep_stop else seq.replace("*", "X")
            m = HDR_TR_RE.search(hdr)
            tr_id = _strip_ver(m.group(1)) if m else None

            fa_w.write(f">{ensp}\n{pep}\n"); n_fa += 1
            pr_w.write(f"{ensp}\t{pep}\n");   n_pr += 1
            if tr_id:
                tr_w.write(f"{tr_id}\t{pep}\t{ensp}\n"); n_tr += 1
            else:
                n_tr_miss += 1

    print(f"Input records:           {n_total:,}")
    print(f"Clean FASTA written:     {out_clean_fa}")
    print(f"Protein lookup written:  {out_prot}")
    print(f"Transcript lookup writ.: {out_tr} (missing transcript tag in {n_tr_miss:,})")

def parse_args():
    ap = argparse.ArgumentParser(
        prog="qol.rebuild_fasta_lookup",
        description="Build FASTA-derived lookups and a cleaned proteome FASTA with a single name prefix."
    )
    ap.add_argument("--name", required=True, help="Base name prefix for all generated files")
    return ap.parse_args()

def main():
    a = parse_args()
    name = a.name.strip()
    if not name:
        raise SystemExit("Empty --name is not allowed.")
    in_fa = Path(FA_ENSEMBL)

    out_dir = Path(LOOKUPS)
    out_tr       = out_dir / f"{name}_peptide_lookup.tsv"
    out_prot     = out_dir / f"{name}_protein_lookup.tsv"
    out_clean_fa = out_dir / f"{name}_proteome_for_interpro.fa"

    _fail_if_exists(out_tr, out_prot, out_clean_fa)

    if not in_fa.exists():
        raise SystemExit(f"Input FASTA not found: {in_fa}")

    build_from_fasta(in_fa, out_tr, out_prot, out_clean_fa)

if __name__ == "__main__":
    sys.exit(main())
