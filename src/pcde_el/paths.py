"""
Three roots, each overridable by an environment variable:

    PCDE_DATA_ROOT     inputs:  raw sources, D0/D1/test CSVs, LLM generations
    PCDE_RESULTS_ROOT  outputs: retrieval dumps, calibration, tables, figures
    PCDE_MODELS_ROOT   fine-tuned bi-encoder checkpoints

Layout::

    data/<dataset>/raw/                       source files (downloaded by you)
    data/<dataset>/processed/d0.csv           name,pid,category
    data/<dataset>/processed/<condition>.csv  D0 + generated_* rows
    data/<dataset>/processed/test.csv         mention,gold_pid[,list,category]
    data/<dataset>/generation/                resumable LLM outputs
    results/<dataset>/<condition>/{bm25,dense,calibration,...}/
    results/tables/, results/figures/
    models/<dataset>/{d0,<condition>}/
"""

from __future__ import annotations

import os
from pathlib import Path

from pcde_el.config import get_dataset


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _root(env: str, default: str) -> Path:
    return Path(os.environ.get(env, repo_root() / default))


def data_root() -> Path:
    return _root("PCDE_DATA_ROOT", "data")


def results_root() -> Path:
    return _root("PCDE_RESULTS_ROOT", "results")


def models_root() -> Path:
    return _root("PCDE_MODELS_ROOT", "models")


def model_slug(model: str) -> str:
    """'gpt-oss:120b' -> 'gpt-oss-120b', safe for filenames."""
    return model.replace(":", "-").replace("/", "-")


# -------- inputs
def dataset_dir(dataset: str) -> Path:
    get_dataset(dataset)
    return data_root() / dataset


def raw_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "raw"


def processed_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "processed"


def d0_csv(dataset: str) -> Path:
    return processed_dir(dataset) / "d0.csv"


def d1_csv(dataset: str, condition: str) -> Path:
    return processed_dir(dataset) / f"{condition}.csv"


def test_csv(dataset: str) -> Path:
    return processed_dir(dataset) / "test.csv"


def generation_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "generation"


def dictionary_generation_json(dataset: str, model: str) -> Path:
    return generation_dir(dataset) / f"dictionary_{model_slug(model)}.json"


def query_generation_json(dataset: str, model: str) -> Path:
    return generation_dir(dataset) / f"query_{model_slug(model)}.json"


def formal_names_csv(dataset: str, model: str) -> Path:
    return generation_dir(dataset) / f"formal_names_{model_slug(model)}.csv"


def prompt_file(kind: str, dataset: str) -> Path:
    """kind is 'dictionary' or 'query'."""
    return repo_root() / "prompts" / kind / f"{dataset}.txt"


# -------- outputs


def run_dir(dataset: str, condition: str) -> Path:
    get_dataset(dataset)
    return results_root() / dataset / condition


def retrieval_dir(dataset: str, condition: str, source: str) -> Path:
    """source is 'bm25' or 'dense'. Holds hits/names/queries/outcome."""
    return run_dir(dataset, condition) / source


def calibration_dir(dataset: str, condition: str) -> Path:
    return run_dir(dataset, condition) / "calibration"


def data_efficiency_dir(dataset: str, condition: str) -> Path:
    return run_dir(dataset, condition) / "data_efficiency"


def query_expansion_dir(dataset: str, condition: str) -> Path:
    return run_dir(dataset, condition) / "query_expansion"


def tables_dir() -> Path:
    return results_root() / "tables"


def figures_dir() -> Path:
    return results_root() / "figures"


# -------- models


def model_dir(dataset: str, tag: str) -> Path:
    """tag is 'd0' or a condition such as 'gpt-d1'."""
    get_dataset(dataset)
    return models_root() / dataset / tag


def resolve_model(dataset: str, condition: str, model_tag: str) -> str:
    """Map a model tag used throughout the paper to a checkpoint.

    vanilla  untrained base encoder            -> Dense(vanilla)
    ft_d0    trained on pairs from D0          -> Dense(D0)
    ft_gen   trained on pairs from D1          -> Dense(D1)
    """
    if model_tag == "vanilla":
        return get_dataset(dataset).base_model
    if model_tag == "ft_d0":
        return str(model_dir(dataset, "d0"))
    if model_tag == "ft_gen":
        return str(model_dir(dataset, condition))
    raise ValueError(f"unknown model tag {model_tag!r} (vanilla/ft_d0/ft_gen)")
