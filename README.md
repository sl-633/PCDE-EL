# PCDE-EL

Code for **"Generated Names as Weak Evidence: Provenance-Calibrated Dictionary Expansion for Scientific Entity Linking"**.

PCDE-EL investigates dictionary expansion for context-free entity linking, where a name is mapped to a database entry without using its surrounding context.

LLM-generated names can improve dictionary coverage, but adding them as retrieval targets can also introduce incorrect matches. Instead of removing generated names, PCDE-EL retains them in the dictionary and adjusts their contribution to entry-level retrieval scores according to their provenance.

For each entry $e$, let:

- $s_o(e)$: maximum retrieval score among its original names
- $s_g(e)$: maximum retrieval score among its generated names

Two calibration schemes are implemented:

**Subtractive calibration**

$$
s(e) = \max(s_o(e), s_g(e) - \lambda), \quad \lambda \geq 0
$$

**Multiplicative calibration**

$$
s(e) = \max(s_o(e), \lambda s_g(e)), \quad 0 \leq \lambda \leq 1
$$

The calibration parameter $\lambda$ is selected using labelled queries, with out-of-fold evaluation based on the official train/dev splits or 5-fold cross-validation over entries.

## Installation

Requires Python 3.10 or later.

```bash
pip install -e .             # Dense retrieval, training, and calibration
pip install -e ".[bm25]"     # BM25 retrieval (Pyserini)
pip install -e ".[llm]"      # Dictionary expansion via Ollama
```

BM25 retrieval requires Java 21 on `PATH`.

## Data Preparation

The original datasets are not redistributed. Download the source files and organize them as follows:

```text
data/
├── ncbi_disease/
│   └── raw/
│       ├── CTD_diseases.csv
│       ├── NCBItrainset_corpus.txt
│       ├── NCBIdevelopset_corpus.txt
│       └── NCBItestset_corpus.txt
└── bc5cdr/
    └── raw/
        ├── desc2026.xml
        ├── supp2026.xml
        ├── CDR_TrainingSet.PubTator.txt
        ├── CDR_DevelopmentSet.PubTator.txt
        └── CDR_TestSet.PubTator.txt
```

Source files:

| File | Source |
|---|---|
| `CTD_diseases.csv` | [CTD MEDIC vocabulary](http://ctdbase.org/downloads/) (decompress `CTD_diseases.csv.gz`) |
| `NCBI*set_corpus.txt` | [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) |
| `desc2026.xml`, `supp2026.xml` | [MeSH 2026](https://www.nlm.nih.gov/mesh/) (descriptor and supplementary records) |
| `CDR_*.PubTator.txt` | BioCreative V CDR corpus |

Notes:

- CDR-d (disease) and CDR-c (chemical) are both constructed from `bc5cdr/raw/`.
- **MeSH 2026 is required.** The legacy-to-current ID mappings in `src/pcde_el/data/bc5cdr.py` are defined for this release. Using a different release changes the resulting dictionary and test set.
- The PoLyMer dataset, based on PoLyInfo, is not publicly available. Its parser is provided in `src/pcde_el/data/polyinfo.py` for reference.

### Build the datasets

Run:

```bash
python scripts/01_build_dictionary.py --dataset ncbi_disease cdr_disease cdr_chemical
```

For each dataset, this generates two files under `data/<dataset>/processed/`:

- `d0.csv`: original dictionary, with columns `name,pid,category`
- `test.csv`: context-free evaluation queries, with columns `mention,gold_pid,list`

### Using a custom dataset

To use another dataset, prepare `d0.csv` and `test.csv` in the same formats and register the dataset in `src/pcde_el/config.py`.

The remaining pipeline operates on these processed files and does not require the original source format.

## Running the Experiments

The pipeline consists of dictionary expansion, retrieval, bi-encoder training, and calibration.

### 1. Prepare the expanded dictionary

The expanded dictionaries (D1) used in the paper are included in the repository as compressed CSV files:

```text
data/
├── ncbi_disease/processed/gpt-d1.csv.gz
├── cdr_disease/processed/gpt-d1.csv.gz
└── cdr_chemical/processed/gpt-d1.csv.gz
```

Each D1 contains the original D0 names followed by LLM-generated names, identified by categories beginning with `generated_`.

After building D0 from the source datasets, decompress the corresponding D1:

```bash
D=ncbi_disease
C=gpt-d1

gunzip -k data/$D/processed/$C.csv.gz
```

Alternatively, generate D1 using a locally running Ollama server:

```bash
python scripts/02_expand_dictionary.py \
    --dataset $D \
    --model gpt-oss:120b \
    --condition $C
```

Dictionary expansion is cached and can be resumed.

The released D1 files should be used to reproduce the results in the paper, since LLM generation may produce different names across runs.

### 2. Run retrieval, training, and calibration

```bash
python scripts/run_pipeline.py --dataset $D --condition $C
```

The pipeline executes the following stages:

| Stage | Script | Description |
|---|---|---|
| `bm25` | `05_retrieve_bm25.py` | BM25 retrieval using D0 and D1 |
| `train_d0` | `04_train_biencoder.py` | Fine-tune a bi-encoder using D0 |
| `train_gen` | `04_train_biencoder.py` | Fine-tune a bi-encoder using D1 |
| `dense` | `06_retrieve_dense.py` | Dense retrieval using the original, D0-tuned, and D1-tuned encoders |
| `calibrate` | `07_calibrate.py` | Select calibration parameters and evaluate the calibrated rankings |

Individual stages can be selected with `--stages`:

```bash
python scripts/run_pipeline.py \
    --dataset $D \
    --condition $C \
    --stages bm25,calibrate
```

Other options:

- `--dry-run`: display the commands without executing them
- `--smoke`: limit training to 100 steps for a quick pipeline check

Each stage can also be executed independently. See the corresponding script's `--help` for available arguments.

Training and dense retrieval use CUDA when available, with CPU fallback otherwise.

## Experimental Settings

The default settings correspond to those used in the paper.

Dataset-specific configurations are defined in `src/pcde_el/config.py`. Other settings can be changed through command-line arguments.

### Dictionary expansion

Implementation: `src/pcde_el/expansion/llm_client.py`

Prompts: `prompts/dictionary/`

| Parameter | Value |
|---|---|
| Model | `gpt-oss:120b` (Ollama) |
| Reasoning effort | `low` |
| Temperature | 0.2 |
| Top-p | 0.9 |
| Top-k | 20 |
| Repeat penalty | 1.1 |
| Maximum output tokens | 4096 |
| Prompting | Zero-shot, one prompt per D0 entry |

### Retrieval

| Parameter | BM25 | Dense |
|---|---|---|
| Implementation | Pyserini | Exact cosine similarity |
| Normalization level | 1 | 0 |
| Retrieval depth | Top 1000 names | Top 1000 names |
| Entry aggregation | Top 50 entries | Top 50 entries |

Additional settings:

- BM25: `k1 = 0.9`, `b = 0.4`, without stemming or stopword removal
- Dense retrieval: mean pooling, maximum sequence length of 32 tokens

### Bi-encoder training

Implementation: `src/pcde_el/training/`

| Parameter | Value |
|---|---|
| Base model | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` |
| PoLyMer base model | MatBERT |
| Training pairs | All ordered name pairs within an entry, at most 20 names per entry |
| Objective | Multi-positive in-batch InfoNCE |
| Temperature (τ) | 0.07 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup | 10%, linear |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| Training steps | 20,000 (160,000 for CDR-c) |
| Sampling | With replacement |
| Mixed precision | FP16 on CUDA |
| Seed | 42 |

### Calibration

Two schemes are evaluated:

- **Multiply:** 21 equally spaced values of $\lambda$ between 0 and 1.
- **Subtract:** $\lambda = 0$, 21 quantiles of the score margins where generated names outperform original names, and one value above the maximum margin.

The parameter is selected to maximize Pass@1 on the labelled training/development queries, using an out-of-fold evaluation protocol.

For PoLyMer, 5-fold cross-validation over entries is used (seed 0).

### Evaluation

The evaluation includes:

- **Pass@1/5:** retrieval accuracy at rank 1 or 5
- **Coverage@50:** proportion of queries for which the correct entry appears among the top 50 candidates
- **Resolution@1/5:** ranking accuracy conditional on candidate coverage
- **MRR:** mean reciprocal rank
- **IN-D0 / OUT-D0:** evaluation by whether a query mention already exists in the original dictionary

Statistical comparisons use:

- Entry-clustered bootstrap with 5,000 resamples and 95% percentile confidence intervals
- Exact McNemar tests

For data-efficiency analysis, calibration is evaluated with labelled-query budgets from 8 to 8192, using 30 random draws per budget.

## Outputs

Results are written under:

```text
results/<dataset>/<condition>/
├── bm25/
│   ├── hits.parquet
│   └── outcome.parquet
├── dense/
│   ├── hits.parquet
│   └── outcome.parquet
└── calibration/
    ├── <retriever>_<scheme>.json
    └── <retriever>_<scheme>.csv
```

Retrieval results contain:

- `hits.parquet`: top-50 entry candidates per query, including original-name and generated-name scores (`s_o` and `s_g`)
- `outcome.parquet`: per-query gold rank, coverage, and Pass@k results

Calibration results contain:

- Selected $\lambda$
- Pass@1/5, Coverage@50, Resolution@1/5, and MRR for the calibrated rankings and baselines
- Comparison against uncalibrated rankings, including entry-clustered bootstrap confidence intervals and exact McNemar p-values

## Reproducibility

Random seeds are fixed throughout the experiments. BM25 retrieval and calibration are deterministic given the expanded dictionary.

Two sources of variation remain:

1. **Dictionary expansion:** LLM sampling may produce different generated names. The released D1 files correspond to those used in the paper.
2. **Bi-encoder training:** GPU training is not guaranteed to be bit-exact, so fine-tuned model checkpoints and retrieval scores may vary slightly between runs.

For reproducing the reported results, use the released D1 files rather than regenerating the dictionaries.

## Paths

By default, data, results, and model checkpoints are stored in `data/`, `results/`, and `models/` under the repository root.

These locations can be overridden using environment variables:

```bash
export PCDE_DATA_ROOT=/path/to/data
export PCDE_RESULTS_ROOT=/path/to/results
export PCDE_MODELS_ROOT=/path/to/models
```

## Repository Structure

```text
src/pcde_el/
├── config.py          Dataset configurations
├── paths.py           Input and output paths
├── normalization.py   String normalization
├── data/              Dataset preprocessing
├── expansion/         Dictionary and query expansion
├── retrieval/         BM25, dense retrieval, score aggregation
├── training/          Training pairs, InfoNCE, bi-encoder training
├── calibration/       Calibration and parameter selection
└── evaluation/        Metrics and statistical evaluation

scripts/                Pipeline stages (01–07) and run_pipeline.py
prompts/                LLM prompts
```

## License

MIT
