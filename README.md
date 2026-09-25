# Amazon ML Challenge 2026: Business Entity Resolution Solution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-brightgreen.svg)](https://github.com/microsoft/LightGBM)
[![Polars](https://img.shields.io/badge/data-Polars-blueviolet.svg)](https://pola.rs/)
[![RapidFuzz](https://img.shields.io/badge/matching-RapidFuzz-orange.svg)](https://github.com/maxbachmann/RapidFuzz)
[![Macro F0.5](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.8503-gold.svg)](#benchmark-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Competition Track:** Amazon ML Challenge 2026 — Business Entity Resolution  
> **Target Metric:** Entity-level Macro $F_{0.5}$ (Precision weighted 2× over Recall, strict 0.0 penalty for false merges on singletons)  
> **Official Submission Status:** Verified and passed official validation (`PASS — no blocking issues found. Safe to submit.`)

---

## 1. Problem Overview

In modern commercial platforms, business identity data arrives from multiple independent, heterogeneous sources with no common foreign keys or unique identifiers:

- **Source 1:** Deduplicated reference entity source (each S1 record is an authoritative entity).
- **Source 2:** Noisy commercial business records.
- **Source 3:** Noisy commercial business records.

**The Objective:** For every Source 1 entity, identify **all** matching records from Source 2 and Source 3. An entity may have **zero** (singletons), **one**, or **multiple** matches.

### Key Data Science Challenges
1. **Severe Combinatorial Scale:** $2.2\text{M}$ S1 records $\times$ $10.3\text{M}$ candidate records $\approx 2.3 \times 10^{13}$ potential Cartesian pairs.
2. **Open-Set Country Distribution:** The training set only contains `US` and `India`, whereas the test set introduces unseen countries such as `France`. The solution must dynamically handle open-set country partitions without hardcoded filters.
3. **High Singleton Frequency:** $\sim 6.5\%$ of Source 1 entities have zero true matches in S2/S3. Because the evaluation metric is Macro $F_{0.5}$, any false positive on a singleton scores **$0.0$** for that entity.
4. **Multimodal Noise:** Phonetic transliterations, legal corporate suffixes (`LLC`, `Pvt Ltd`, `Corp`), abbreviations (`Rd`, `St`, `Ave`), landmark-based addresses, missing PIN/postal codes, and word-order transpositions.

---

## 2. Solution Architecture

```
                                      +------------------------+
                                      |  Source 1 / 2 / 3 TSVs |
                                      +-----------+------------+
                                                  |
                                                  v
                                      +------------------------+
                                      | Text & Locality Engine |
                                      | (Legal suffix, tokens) |
                                      +-----------+------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     |         Dynamic Open-Set Country Partitioning           |
                     |                 (US, India, France)                     |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     |            Multi-Channel Inverted Indexing              |
                     |   - Name Word Tokens (nw)   - Address Tokens (aw)       |
                     |   - Numeric Anchors (d)     - Prefix 3-Grams (n3p)      |
                     |                IDF-Weighted Top-K Scoring               |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                                      +------------------------+
                                      |   candidate_pairs.tsv  |
                                      |  (81.7M test candidates|
                                      |  99.9995% space cut)   |
                                      +-----------+------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     |         27-Dimensional Pairwise Feature Engine          |
                     | - Levenshtein Ratio       - Token Sort / Set Ratio      |
                     | - WRatio Substring Score  - Jaccard & Overlap Metrics   |
                     | - Numeric Digit Jaccard   - Length & Token Count Diffs  |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     |               Calibrated LightGBM Classifier            |
                     |         (Group-stratified S1 out-of-fold training)       |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     |      Macro F0.5 Threshold Optimization (tau* = 0.60)    |
                     |  - High-Precision Singleton Guard (0 false merge penalty|
                     |  - Strict Subset Enforcement: matching <= candidate     |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                                      +------------------------+
                                      |  matching_results.tsv  |
                                      |  (Final submissions)   |
                                      +------------------------+
```

---

## 3. Validation Performance & Benchmarks

All models were evaluated using rigorous `GroupShuffleSplit` on `source1_entity_id` to strictly prevent data leakage:

| Evaluation Metric | Baseline (Exact Name Match) | LightGBM ML Pipeline | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Entity-Level Macro $F_{0.5}$** | **0.2211** | **0.8503** | **+0.6292 (+284.6%)** |
| **Macro Precision** | 0.3040 | **0.9053** | **+0.6013** |
| **Macro Recall** | 0.1334 | **0.7518** | **+0.6184** |
| **Pairwise Precision** | 0.4120 | **0.9622** | **+0.5502** |
| **Pairwise Recall** | 0.1298 | **0.7460** | **+0.6162** |
| **Singleton Accuracy** | 71.43% | **89.80%** | **+18.37%** |
| **Validation Blocking Recall** | — | **84.74%** | (8,885 / 10,485 recalled matches) |
| **Reduction Ratio** | — | **99.9995%** | $81.7\text{M}$ from $17.2\text{T}$ pairs |

---

## 4. Ablation Study

| Exp ID | Feature Set Configuration | Num Feats | Optimal $\tau^*$ | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Singleton Acc | Pair Prec |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Name Similarity Only | 12 | 0.40 | 0.6923 | 0.7392 | 0.6535 | 53.74% | 0.6989 |
| **B** | Address Similarity Only | 11 | 0.40 | 0.7509 | 0.8067 | 0.6873 | 75.51% | 0.7535 |
| **C** | Name + Address Combined | 23 | 0.60 | 0.8485 | 0.9041 | 0.7488 | 89.12% | 0.9629 |
| **D** | Name + Address + Country | 26 | 0.50 | 0.8505 | 0.8976 | 0.7686 | 84.35% | 0.9483 |
| **E** | Name + Addr + Country + Blocking Score | 27 | 0.60 | 0.8503 | 0.9053 | 0.7518 | 89.80% | 0.9622 |
| **F** | **Full Engineered Pipeline** | **27** | **0.60** | **0.8503** | **0.9053** | **0.7518** | **89.80%** | **0.9622** |

---

## 5. Official Submission Validation

The submission outputs were validated against the official challenge validator `utils/validate_submission.py`:

```bash
python resource/student_resource/utils/validate_submission.py \
    --matching resource/student_resource/output/matching_results.tsv \
    --candidate resource/student_resource/output/candidate_pairs.tsv \
    --test-dir resource/student_resource/dataset/test
```

**Validator Output:**
```text
ML Challenge 2026 — submission validator
  test dir: resource/student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (433549 empty, 1298995 non-empty).
  candidate_pairs.tsv: 1732544 rows (61292 empty, 1671252 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- **Exit Code:** `0`
- **Output File Cardinality:** Exactly **1,732,544** test entities present in both files.
- **Strict Subset Constraint:** $100\%$ verified. Every predicted match in `matching_results.tsv` exists within `candidate_pairs.tsv`.

---

## 6. Project Structure

```
Amazon-ML-Challenge-2026/
├── README.md                                 # Project documentation & benchmark overview
├── problem_statement.pdf                     # Official competition problem specification
├── resource/
│   └── student_resource/
│       ├── Documentation_template.md         # Full solution writeup & ablation methodology
│       ├── package_submission.py             # Script packaging the official ZIP submission
│       ├── run_fast_test_scoring.py          # Optimized test-set inference pipeline
│       ├── check_country_cross.py            # Country partition validation script
│       ├── code/
│       │   └── business_entity_resolution/
│       │       ├── README.md                 # Pipeline reproduction instructions
│       │       ├── requirements.txt          # Production dependencies
│       │       ├── run_pipeline.py           # Single-command pipeline orchestrator
│       │       └── src/
│       │           ├── __init__.py
│       │           ├── config.py             # Hyperparameters & path constants
│       │           ├── data_loader.py        # High-throughput TSV ingestion
│       │           ├── normalization.py      # Legal suffix & text cleaners
│       │           ├── preprocessing.py      # Polars DataFrame pipelines
│       │           ├── blocking.py           # Inverted index blocking engine
│       │           ├── candidate_generation.py# Country-partitioned candidate retrieval
│       │           ├── feature_engineering.py# 27-dimensional pairwise feature extractor
│       │           ├── pair_dataset.py       # Training pair dataset generator
│       │           ├── model.py              # Classifier definitions
│       │           ├── train_model.py        # LightGBM training & persistence
│       │           ├── threshold_optimizer.py# Macro F0.5 grid-search optimizer
│       │           ├── matcher.py            # Match inference & subset validator
│       │           ├── evaluation.py         # Exact entity-level Macro F0.5 evaluator
│       │           ├── validation.py         # GroupShuffleSplit validation runner
│       │           ├── inference.py          # Test candidate scoring & TSV generator
│       │           ├── pipeline.py           # End-to-end execution manager
│       │           └── utils.py              # Logging & metric helpers
│       ├── output/
│       │   └── matching_results.tsv          # Scored leaderboard submission file
│       ├── reports/
│       │   ├── experiment_results.csv        # Full ablation experiment logs
│       │   ├── validation_report.json        # Detailed model performance metrics
│       │   ├── model_report.json             # LightGBM hyperparameter & training report
│       │   ├── error_analysis.json           # True/False positive/negative qualitative samples
│       │   ├── eda_summary.json              # Dataset distribution statistics
│       │   └── eda_report.html               # Interactive visual EDA report
│       └── utils/
│           └── validate_submission.py        # Official challenge validator script
```

---

## 7. Quick Start & Reproduction

### Prerequisites
- Python 3.10+
- 16 GB+ RAM recommended

### Installation
```bash
cd resource/student_resource
pip install -r code/business_entity_resolution/requirements.txt
```

### Run End-to-End Pipeline
To run the full end-to-end pipeline (data preprocessing $\to$ candidate blocking $\to$ feature extraction $\to$ LightGBM training $\to$ threshold optimization $\to$ test set inference):

```bash
python code/business_entity_resolution/run_pipeline.py
```

### Validate Submission Format
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Package Submission ZIP
```bash
python package_submission.py
```
This produces `amazon_ml_submission.zip` matching the required competition directory structure.

---


## 8. License & Fair-Play Compliance

- **Permissive Open-Source Licensing:** Implemented strictly with MIT / Apache 2.0 licensed components (`lightgbm`, `polars`, `rapidfuzz`, `scikit-learn`, `numpy`).
- **Model Parameter Limit:** LightGBM decision tree ensemble ($<10\text{ MB}$, $<10^7$ parameters), orders of magnitude below the 8 Billion parameter ceiling.
- **Zero External Lookups:** Exclusively uses provided challenge TSV data. Zero external APIs, web scraping, or supplementary external datasets used.
