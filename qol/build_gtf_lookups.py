# qol/rebuild_gtf_lookups.py
from __future__ import annotations
from pathlib import Path
import argparse, gzip, sys
import polars as pl

from config.config import (
    GTF_ENSEMBL,
    LOOKUPS,            # <-- base output directory
)

def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")

def _ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True)
def _strip_ver(x: str) -> str: return x.split(".", 1)[0] if x else x

def _parse_attributes(attr_str: str):
    attrs, tags = {}, []
    for raw in (x.strip() for x in attr_str.split(";")):
        if not raw: continue
        if raw.startswith("tag "):
            tags.append(raw.split(" ", 1)[1].strip().strip('"'))
        elif " " in raw:
            k, v = raw.split(" ", 1)
            v = v.strip().strip('"')
            if k in ("transcript_id", "gene_id") and v:
                v = _strip_ver(v)
            attrs[k] = v
    return attrs, tags

def _build_nextgene(gene_rows: list[dict]) -> pl.DataFrame:
    if not gene_rows:
        return pl.DataFrame({"gene_id": [], "chr": [], "strand": [], "tss": [], "tes": [],
                             "next_gene_id": [], "next_tss": [], "next_tes": []})
    base = (pl.DataFrame(gene_rows)
            .select(["gene_id","chr","strand","start","end"])
            .with_columns([
                pl.when(pl.col("strand")=="+").then(pl.col("start")).otherwise(pl.col("end")).alias("tss"),
                pl.when(pl.col("strand")=="+").then(pl.col("end")).otherwise(pl.col("start")).alias("tes"),
            ]))
    plus = (base.filter(pl.col("strand")=="+")
                 .sort(["chr","tss"])
                 .with_columns([
                     pl.col("gene_id").shift(-1).over("chr").alias("next_gene_id"),
                     pl.col("tss").shift(-1).over("chr").alias("next_tss"),
                     pl.col("tes").shift(-1).over("chr").alias("next_tes"),
                 ]))
    minus = (base.filter(pl.col("strand")=="-")
                  .sort(["chr","tss"], descending=[False, True])
                  .with_columns([
                      pl.col("gene_id").shift(-1).over("chr").alias("next_gene_id"),
                      pl.col("tss").shift(-1).over("chr").alias("next_tss"),
                      pl.col("tes").shift(-1).over("chr").alias("next_tes"),
                  ]))
    return (pl.concat([plus, minus], how="diagonal")
              .drop_nulls(subset=["next_gene_id"])
              .select(["gene_id","chr","strand","tss","tes","next_gene_id","next_tss","next_tes"]))

def _fail_if_exists(*paths: Path):
    clashes = [p for p in paths if p.exists()]
    if clashes:
        msg = "Refusing to overwrite existing files:\n" + "\n".join(str(p) for p in clashes)
        raise SystemExit(msg)

def build_lookups(gtf_path: Path, out_all: Path, out_cds: Path, out_mane: Path, out_next: Path):
    rows, mane_rows, gene_rows = [], [], []
    with _open_text(gtf_path) as gtf:
        for line in gtf:
            if not line or line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9: continue
            chr_, _, feature, start, end, _, strand, _, attributes = parts
            try:
                start_i, end_i = int(start), int(end)
            except ValueError:
                continue
            attrs, tags = _parse_attributes(attributes)
            tr_id  = attrs.get("transcript_id", "")
            gene_id = attrs.get("gene_id", "")
            rows.append({
                "transcript_id": tr_id,
                "gene_id": gene_id,
                "feature_type": feature,
                "strand": strand,
                "chr": chr_,
                "start": start_i,
                "end": end_i
            })
            if feature == "gene" and gene_id:
                gene_rows.append({"gene_id": gene_id, "chr": chr_, "strand": strand,
                                  "start": start_i, "end": end_i})
            if feature == "transcript" and tr_id and gene_id and (
                "MANE_Select" in tags or "MANE_Plus_Clinical" in tags
            ):
                tss = start_i if strand == "+" else end_i
                mane_rows.append({
                    "gene_id": gene_id,
                    "transcript_id": tr_id,
                    "mane_transcript_id": tr_id,
                    "mane_tss": tss,
                    "mane_tag": "MANE_Select" if "MANE_Select" in tags else "MANE_Plus_Clinical"
                })

    gtf_df = pl.DataFrame(rows)

    all_df = gtf_df.select(["transcript_id","gene_id","feature_type","strand","chr","start","end"])
    _ensure_parent(out_all); all_df.write_csv(out_all, separator="\t")

    cds_df = (gtf_df.filter(pl.col("feature_type")=="CDS")
              .with_columns([
                  (pl.col("end") - pl.col("start") + 1).alias("length"),
                  pl.when(pl.col("strand")=="+").then(pl.col("start")).otherwise(-pl.col("end")).alias("order_key")
              ])
              .sort(["transcript_id","order_key"])
              .with_columns(pl.col("length").cum_sum().over("transcript_id").alias("cumulative_length"))
              .with_columns((pl.col("cumulative_length") // 3).alias("cumulative_length_aa"))
              .select(["transcript_id","strand","start","end","length","cumulative_length","cumulative_length_aa"]))
    _ensure_parent(out_cds); cds_df.write_csv(out_cds, separator="\t")

    mane_df = (pl.DataFrame(mane_rows)
               .sort([pl.when(pl.col("mane_tag")=="MANE_Select").then(0).otherwise(1), "gene_id"])
               .unique("gene_id", keep="first")
               .select(["gene_id","transcript_id","mane_transcript_id","mane_tss"]))
    _ensure_parent(out_mane); mane_df.write_csv(out_mane, separator="\t")

    next_df = _build_nextgene(gene_rows)
    _ensure_parent(out_next); next_df.write_csv(out_next, separator="\t")

def parse_args():
    ap = argparse.ArgumentParser(
        prog="qol.rebuild_gtf_lookups",
        description="Build GTF-derived lookups with a single name prefix, no overwrites."
    )
    ap.add_argument("--name", required=True, help="Base name prefix for all generated files")
    return ap.parse_args()

def main():
    a = parse_args()
    name = a.name.strip()
    if not name:
        raise SystemExit("Empty --name is not allowed.")

    gtf_path = Path(GTF_ENSEMBL)
    base_dir = Path(LOOKUPS)  

    out_all  = base_dir / f"{name}_all_lookup.tsv"
    out_cds  = base_dir / f"{name}_cds_lookup.tsv"
    out_mane = base_dir / f"{name}_mane_lookup.tsv"
    out_next = base_dir / f"{name}_next_gene_lookup.tsv"

    _fail_if_exists(out_all, out_cds, out_mane, out_next)

    if not gtf_path.exists():
        raise SystemExit(f"GTF not found: {gtf_path}")

    build_lookups(gtf_path, out_all, out_cds, out_mane, out_next)
    print("Written:\n", out_all, "\n", out_cds, "\n", out_mane, "\n", out_next)

if __name__ == "__main__":
    sys.exit(main())
