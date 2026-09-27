# Amazon ML Challenge 2026: Business Entity Resolution Solution

## Overview
This package implements a production-grade, highly-scalable Business Entity Resolution pipeline to match Source 1 reference entities against noisy records in Source 2 and Source 3 across open-set countries (US, India, France).

## Key Architecture
1. **Multi-Channel Country-Partitioned Inverted Index Blocking**:
   - Dynamic per-country partitioning (supporting unseen countries like France).
   - Multi-channel inverted index generation: Name word tokens (`nw`), Address locality tokens (`aw`), Numeric street/PIN anchors (`d`), and Name 3-gram prefixes (`n3p`).
   - Inverse Document Frequency (IDF) candidate scoring with multi-threaded Polars aggregation.
2. **Comprehensive Pairwise Feature Engineering**:
   - 27-dimensional feature vector combining RapidFuzz token set/sort/partial metrics, Jaccard token overlap, exact numeric anchors, length differences, and structural indicators.
3. **LightGBM Gradient Boosted Matcher**:
   - Trained on hard negative pairs from blocking with binary log-loss.
4. **Precision-Oriented F0.5 Threshold Optimization & Singleton Detection**:
   - Systematic grid search over decision thresholds specifically targeting Macro-Averaged F0.5.
   - Robust singleton detection and one-to-many match consistency enforcement.

## Reproduction Instructions
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete end-to-end pipeline
python ../../run_pipeline.py
```

## Generated Outputs
- `output/matching_results.tsv`: Final predicted matches per Source 1 entity.
- `output/candidate_pairs.tsv`: Final candidate set evaluated by the ML matcher.
- `reports/`: EDA reports, validation metrics, ablation studies, and error analysis.
