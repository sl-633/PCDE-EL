"""Per-dataset settings. To add a dataset, register it here and write its
processed/{d0,test}.csv."""

from __future__ import annotations

from dataclasses import dataclass

SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# MatBERT is distributed from github.com/lbnlp/MatBERT, not the HF Hub;
# download it and point this path (or --base-model) at the local copy.
MATBERT = "models/matbert-base-uncased"


@dataclass(frozen=True)
class DatasetConfig:
    name: str  # folder under data/
    short: str  # short code used in tables
    display_name: str  # name used in the paper
    database: str  # source database of D0
    categories: tuple[str, ...]  # name categories the LLM generates
    base_model: str  # bi-encoder initialisation
    train_steps: int  # fixed optimisation budget
    protocol: str  # for lambda selection, "split" (official train/dev/test) or "cv" (5-fold cross validation)


DATASETS: dict[str, DatasetConfig] = {
    "polyinfo": DatasetConfig(
        name="polyinfo",
        short="polymer",
        display_name="PoLyMer",
        database="PoLyInfo",
        categories=("abbreviation", "alias", "trivial_name", "brand_name"),
        base_model=MATBERT,
        train_steps=20_000,
        protocol="cv",
    ),
    "ncbi_disease": DatasetConfig(
        name="ncbi_disease",
        short="ncbi",
        display_name="NCBI-d",
        database="MEDIC (MeSH+OMIM)",
        categories=("abbreviation", "alias", "eponym", "common_name"),
        base_model=SAPBERT,
        train_steps=20_000,
        protocol="split",
    ),
    "cdr_disease": DatasetConfig(
        name="cdr_disease",
        short="cdr-d",
        display_name="CDR-d",
        database="MeSH (Disease)",
        categories=("abbreviation", "alias", "eponym", "common_name"),
        base_model=SAPBERT,
        train_steps=20_000,
        protocol="split",
    ),
    "cdr_chemical": DatasetConfig(
        name="cdr_chemical",
        short="cdr-c",
        display_name="CDR-c",
        database="MeSH (Chemical)",
        categories=(
            "systematic_name",
            "common_name",
            "abbreviation",
            "code_name",
            "trade_name",
        ),
        # The chemical dictionary is ~8x larger than the others.
        base_model=SAPBERT,
        train_steps=160_000,
        protocol="split",
    ),
}

PAPER_ORDER = ("polyinfo", "ncbi_disease", "cdr_disease", "cdr_chemical")


def get_dataset(name: str) -> DatasetConfig:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}; Registered: {', '.join(DATASETS)}"
        ) from None
