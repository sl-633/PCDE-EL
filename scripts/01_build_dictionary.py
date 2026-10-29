"""Build D0 and the context-free test set from the raw sources.

Writes data/<dataset>/processed/{d0,test}.csv. Both BC5CDR datasets read
from data/bc5cdr/raw/.

    python scripts/01_build_dictionary.py --dataset ncbi_disease
    python scripts/01_build_dictionary.py --dataset cdr_disease cdr_chemical
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcde_el import paths
from pcde_el.data import bc5cdr, ncbi, polyinfo
from pcde_el.data.schema import write_rows

DICT_HEADER = ["name", "pid", "category"]


def build_polyinfo(raw: Path) -> None:
    ds = "polyinfo"
    d0 = polyinfo.build_d0(raw / "polyinfo_polymer_names.csv")
    write_rows(paths.d0_csv(ds), DICT_HEADER, d0)
    test = polyinfo.build_test(raw / "common.xlsx", raw / "uncommon.xlsx")
    write_rows(paths.test_csv(ds), ["mention", "gold_pid", "category", "list"], test)
    print(f"[{ds}] D0: {polyinfo.summarize(d0)}\n[{ds}] test: {len(test):,} queries")


def build_ncbi(raw: Path) -> None:
    ds = "ncbi_disease"
    d0, equiv = ncbi.build_d0(raw / "CTD_diseases.csv")
    write_rows(paths.d0_csv(ds), DICT_HEADER, d0)
    # Alternate ids (mostly OMIM) only decide whether a gold id is covered;
    # they are never used as names.
    equiv_csv = paths.processed_dir(ds) / "d0_equivalent_ids.csv"
    write_rows(equiv_csv, ["pid", "equivalent_id"], equiv)
    splits = {
        "train": raw / "NCBItrainset_corpus.txt",
        "dev": raw / "NCBIdevelopset_corpus.txt",
        "test": raw / "NCBItestset_corpus.txt",
    }
    test = ncbi.build_test(paths.d0_csv(ds), equiv_csv, splits)
    write_rows(paths.test_csv(ds), ["mention", "gold_pid", "list"], test)
    print(f"[{ds}] D0: {len(d0):,} names; test: {len(test):,} queries")


def build_bc5cdr(datasets: list[str], raw: Path) -> None:
    # MeSH is parsed once and split into disease and chemical records.
    d0 = bc5cdr.build_d0(raw / "desc2026.xml", raw / "supp2026.xml")
    splits = {
        "train": raw / "CDR_TrainingSet.PubTator.txt",
        "dev": raw / "CDR_DevelopmentSet.PubTator.txt",
        "test": raw / "CDR_TestSet.PubTator.txt",
    }
    for ds in datasets:
        entity = ds.split("_")[1]  # cdr_disease -> disease
        write_rows(paths.d0_csv(ds), DICT_HEADER, d0[entity])
        test = bc5cdr.build_test(entity, paths.d0_csv(ds), splits)
        write_rows(paths.test_csv(ds), ["mention", "gold_pid", "list"], test)
        print(f"[{ds}] D0: {len(d0[entity]):,} names; test: {len(test):,} queries")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        choices=["polyinfo", "ncbi_disease", "cdr_disease", "cdr_chemical"],
    )
    ap.add_argument(
        "--raw-dir", type=Path, default=None, help="override the raw-source directory"
    )
    args = ap.parse_args()

    cdr = [d for d in args.dataset if d.startswith("cdr_")]
    if "polyinfo" in args.dataset:
        build_polyinfo(args.raw_dir or paths.raw_dir("polyinfo"))
    if "ncbi_disease" in args.dataset:
        build_ncbi(args.raw_dir or paths.raw_dir("ncbi_disease"))
    if cdr:
        build_bc5cdr(cdr, args.raw_dir or paths.data_root() / "bc5cdr" / "raw")


if __name__ == "__main__":
    main()
