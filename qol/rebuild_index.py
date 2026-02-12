"""
rebuild_index.py
================
Unified entry point to rebuild the FAISS domain index from any stage.

Purpose
-------
Wraps the domain-retrieval pipeline stages S01–S05 and allows selective
execution depending on which intermediate data already exist.

Execution modes
---------------
--ips          Run stages S01→S05  (InterPro prep → index build)
--embeddings   Run stages S04→S05  (embedding + index build only)
--index        Run stage  S05      (index build only)

Required
--------
--name <filename.index>
    Name of the FAISS index file to create inside FAISS_DIR.

Config dependencies
-------------------
Uses paths from config.config:
  IPS_OUT, DOMAIN_WINDOWS, PROTEIN_EMBEDDINGS, FAISS_DIR

Outputs
-------
• FAISS_DIR/<name>  – serialized FAISS index
• Prints to stdout the path of each completed stage output.

Notes
-----
All stage functions (s01–s05) are imported from the domain_retrieval package
and executed in process. This script performs no data validation; it assumes
all previous stage dependencies are in place.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from config.config import IPS_OUT, DOMAIN_WINDOWS, PROTEIN_EMBEDDINGS, FAISS_DIR
from qol.domain_retrieval.S01_extract_domains import main as s01_run
from qol.domain_retrieval.S02_filter_and_merge import main as s02_run
from qol.domain_retrieval.S03_prep_domain_windows import run as s03_run
from qol.domain_retrieval.S04_embed_domain_windows import embed as s04_run
from qol.domain_retrieval.S05_build_index import build_index as s05_run

def run_ips_to_end(index_name: str):
    print("[S01] InterPro prep + scan"); s01_run()
    print("[S02] filter + merge");       s02_run(); print(f"merged → {Path(IPS_OUT) / 'merged.tsv'}")
    print("[S03] windows");              s03_run(); print(f"windows → {DOMAIN_WINDOWS}")
    print("[S04] embeddings");           s04_run(); print(f"embeddings → {PROTEIN_EMBEDDINGS}")
    print("[S05] build index");          s05_run(index_name); print(f"index → {FAISS_DIR / index_name}")

def run_embeddings_to_end(index_name: str):
    print("[S04] embeddings"); s04_run(); print(f"embeddings → {PROTEIN_EMBEDDINGS}")
    print("[S05] build index"); s05_run(index_name); print(f"index → {FAISS_DIR / index_name}")

def run_index_only(index_name: str):
    print("[S05] build index"); s05_run(index_name); print(f"index → {FAISS_DIR / index_name}")

def main():
    ap = argparse.ArgumentParser(description="Rebuild FAISS index with selectable start step.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Run S01→S05.")
    g.add_argument("--embeddings", action="store_true", help="Run S04→S05.")
    g.add_argument("--index", action="store_true", help="Run S05 only.")
    ap.add_argument("--name", required=True, help="Output index filename (stored under FAISS_DIR).")
    args = ap.parse_args()

    idx_name = args.name
    if args.embeddings:
        run_embeddings_to_end(idx_name)
    elif args.index:
        run_index_only(idx_name)
    else:
        run_ips_to_end(idx_name)

if __name__ == "__main__":
    main()
