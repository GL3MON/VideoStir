# MedVidQA Retrieval Evaluation Framework

Comprehensive system to evaluate the retrieval performance of VideoStir on the MedVidQA dataset.

## Overview

This framework allows you to:

1. **Run VideoStir retrieval** on MedVidQA samples
2. **Evaluate retrieval quality** using standard IR metrics
3. **Generate detailed reports** with statistics and visualizations
4. **Identify bottlenecks** and improvement opportunities

## File Structure

```
VideoStir/
├── evaluate_retrieval.py          # Main evaluation script
├── run_retrieval_medvidqa.py      # Run VideoStir pipeline on MedVidQA
├── generate_eval_report.py        # Generate detailed reports
├── demo_eval.py                   # Quick demo and explanation
├── EVAL_README.md                 # This file
├── MedVidQA/
│   ├── train.json                 # Training set annotations
│   ├── val.json                   # Validation set annotations
│   └── test.json                  # Test set annotations
├── eval_results/                  # Evaluation results (created)
├── eval_reports/                  # Generated reports (created)
└── medvidqa_retrieval/            # Cached retrieval results (created)
```

## Quick Start

### 0. Prerequisites

```bash
# Ensure VideoStir is installed
cd VideoStir/inference
pip install -r requirements_inference.txt
cd ..

# Install evaluation dependencies
pip install numpy
```

### 1. View Demo

```bash
python demo_eval.py
```

This shows the data structures and expected workflow.

### 2. Run Evaluation on Validation Set

```bash
# Quick evaluation on 5 samples
python evaluate_retrieval.py --dataset val --num-samples 5

# Full evaluation set
python evaluate_retrieval.py --dataset val
```

Output: `eval_results/eval_results.json` with detailed metrics.

### 3. Generate Reports

```bash
python generate_eval_report.py --results eval_results/eval_results.json
```

Generates:
- `eval_reports/evaluation_report.txt` - Detailed text report
- `eval_reports/evaluation_report.md` - Markdown for documentation
- `eval_reports/sample_results.csv` - Per-sample metrics in CSV

## Complete Workflow

### Step 1: Run Retrieval (Optional)

If you have MedVidQA videos downloaded:

```bash
# Download videos first:
# Videos available from YouTube links in MedVidQA
# Use pytube or youtube-dl to download

# Then run retrieval:
python run_retrieval_medvidqa.py --dataset val --num-samples 10
```

This:
- Downloads or uses cached retrieval results
- Runs full VideoStir pipeline on each sample
- Saves retrieved frames for evaluation
- Reports success/failure statistics

### Step 2: Evaluate Retrieval

```bash
python evaluate_retrieval.py --dataset val
```

This:
- Loads cached retrieval results
- Compares with ground truth answer timestamps
- Computes metrics for each sample
- Aggregates statistics
- Saves results to JSON

### Step 3: Generate Reports

```bash
python generate_eval_report.py --results eval_results/eval_results.json
```

Creates human-readable reports with:
- Summary statistics
- Per-sample detailed results
- Top/bottom performing samples
- Analysis and recommendations
- CSV export for further analysis

## Evaluation Metrics

### Mean Reciprocal Rank (MRR)

**Definition:** Position of the first relevant frame

$$\text{MRR} = \frac{1}{\text{rank of first relevant item}}$$

**Interpretation:**
- 1.0 = First frame is relevant (perfect)
- 0.5 = First relevant frame at rank 2
- 0.2 = First relevant frame at rank 5
- 0.0 = No relevant frames retrieved

**Target:** > 0.5 (first relevant typically in top-2)

### Recall@K

**Definition:** Fraction of ground truth content covered in top-K results

$$\text{Recall@K} = \frac{\text{# relevant frames in top-K}}{\text{# frames in ground truth range}}$$

**Interpretation:**
- Recall@10 = Coverage in first 10 frames
- Recall@25 = Coverage in first 25 frames
- Recall@128 = Coverage in all retrieved frames

**Target:**
- Recall@25 > 0.6 (covers 60% of answer)
- Recall@128 > 0.8 (covers 80% of answer)

### MAP@K (Mean Average Precision)

**Definition:** Average precision at each relevant position

$$\text{MAP@K} = \frac{1}{m} \sum_{k=1}^{K} P(k) \cdot \text{rel}(k)$$

Where:
- P(k) = precision at rank k
- rel(k) = relevance of item at rank k
- m = number of relevant items

**Interpretation:** Balances precision (few false positives) and recall (many true positives)

**Target:** Similar to recall, > 0.6 at K=25

### Temporal Coverage@128

**Definition:** Percentage of answer time range covered by retrieved frames

**Algorithm:**
1. Divide ground truth time range into 10 equal buckets
2. Count how many buckets have at least one retrieved frame
3. Coverage = (# covered buckets) / 10

**Interpretation:**
- Ensures retrieved frames span across the answer, not clustered
- A system can have high recall by getting 10 frames next to each other
- Coverage penalizes this clustering

**Target:** > 0.7 (covers 7+ out of 10 time buckets)

## Interpreting Results

### Good Retrieval System

```
MRR:           0.65 ± 0.25   ✓ First relevant frame typically high
Recall@25:     0.72 ± 0.18   ✓ Most content retrieved in top-25
Recall@128:    0.89 ± 0.10   ✓ Comprehensive coverage overall
MAP@25:        0.68 ± 0.20   ✓ High quality ranking
Coverage@128:  0.78 ± 0.15   ✓ Frames well-distributed in time
```

**Analysis:** System effectively identifies answer timestamps and ranks them highly.

### Moderate System

```
MRR:           0.35 ± 0.25   ⚠ First relevant often at rank 3-4
Recall@25:     0.45 ± 0.20   ⚠ Missing ~50% of answer content
Recall@128:    0.72 ± 0.15   ⚠ Incomplete even with all frames
MAP@25:        0.40 ± 0.20   ⚠ Relevance ranking needs work
Coverage@128:  0.55 ± 0.20   ⚠ Sparse temporal distribution
```

**Analysis:** System finds relevant frames but not perfectly ranked. Needs improvements in:
- Intent analysis (understanding query)
- Visual embeddings (semantic matching)
- Reranking (scoring relevance)

### Poor Retrieval System

```
MRR:           0.10 ± 0.12   ✗ First relevant rarely in top-10
Recall@25:     0.20 ± 0.15   ✗ Poor early coverage
Recall@128:    0.35 ± 0.20   ✗ Very incomplete
MAP@25:        0.15 ± 0.12   ✗ Ranking completely off
Coverage@128:  0.30 ± 0.20   ✗ Isolated frames
```

**Analysis:** Major issues in retrieval pipeline. Consider:
- Retraining models
- Checking pipeline configuration
- Debugging intent analysis
- Validating video preprocessing

## Debugging Guide

### Low MRR

**Issue:** First relevant frame appears late in ranking

**Possible Causes:**
1. Intent analysis failing to understand query
   - Check intent analyzer prompt
   - Verify subtitle extraction works
   - Test on sample queries

2. Visual embeddings not semantically aligned
   - Check embedding model (PE-Core-G14)
   - Verify tokenizer works correctly
   - Inspect query-segment similarity scores

3. Query encoding/preprocessing issues
   - Check for typos/normalization
   - Verify CLIP text tokenizer handles query well

**Debugging Steps:**
```python
# Load cache and inspect
import json
cache = json.load(open("pipeline_output/sample_1/cache/retrieval.json"))
print("Top-k selected segments:", cache["top_k_segments"][:3])
print("Similarity scores:", [s["similarity"] for s in cache["top_k_segments"]])
```

### Low Recall@25

**Issue:** Fewer than expected frames from answer range retrieved

**Possible Causes:**
1. Top-k too small in retrieval stage (increase --top-k)
2. Spatial expansion not working (check graph connections)
3. Answer spans multiple unrelated segments
4. Visual embeddings fragmented within answer range

**Debugging Steps:**
```bash
# Try with higher top-k
python run_retrieval_medvidqa.py --top-k 5 --spatial-k 5
```

### Low Coverage

**Issue:** Retrieved frames clustered in time, not spanning answer range

**Possible Causes:**
1. Temporal edges not connecting segments properly
2. System finds one good segment but misses others
3. Answer contains multiple distinct visual moments

**Debugging Steps:**
- Check segment boundary detection
- Verify temporal graph connections
- Inspect which segments are selected

## Output Files

### Evaluation Results

`eval_results/eval_results.json`:
```json
{
  "timestamp": "eval_results",
  "aggregate_metrics": {
    "num_samples": 145,
    "mrr_avg": 0.65,
    "mrr_std": 0.25,
    "recall@25_avg": 0.72,
    "recall@25_std": 0.18,
    // ... more metrics
  },
  "sample_results": [
    {
      "sample_id": 1,
      "question": "How to perform epley maneuver?",
      "answer_start": 12,
      "answer_end": 108,
      "retrieved_count": 128,
      "mrr": 0.8,
      "recall@25": 0.75,
      "recall@128": 0.92,
      // ... more per-sample metrics
    }
    // ... more samples
  ]
}
```

### Reports

**Text Report** (`evaluation_report.txt`):
- Overview and summary statistics
- Per-K metrics breakdown
- Detailed per-sample analysis
- Statistical distributions
- Recommendations

**Markdown Report** (`evaluation_report.md`):
- Summary table format
- Top/bottom performing samples
- GitHub-friendly formatting

**CSV Export** (`sample_results.csv`):
- All samples and metrics
- Easy import to Excel/analysis tools

## Advanced Usage

### Evaluate Multiple Datasets

```bash
# Validate on train, val, and test
for DATASET in train val test; do
    python evaluate_retrieval.py --dataset $DATASET --output-file eval_${DATASET}.json
    python generate_eval_report.py --results eval_results/eval_${DATASET}.json
done
```

### Compare Different Configurations

```bash
# Test with different retrieval parameters
for K in 3 5 10; do
    python run_retrieval_medvidqa.py --top-k $K --num-samples 20
    python evaluate_retrieval.py --output-file eval_k${K}.json
done
```

### Statistical Analysis

```python
import json
import numpy as np

# Load results
with open("eval_results/eval_results.json") as f:
    results = json.load(f)

samples = results["sample_results"]
mrrs = [s["mrr"] for s in samples]

# Quartile analysis
print(f"Q1 (25%): {np.percentile(mrrs, 25):.4f}")
print(f"Median:   {np.percentile(mrrs, 50):.4f}")
print(f"Q3 (75%): {np.percentile(mrrs, 75):.4f}")

# Failure analysis
failures = [s for s in samples if s["mrr"] == 0]
print(f"Completely failed: {len(failures)/len(samples)*100:.1f}%")
```

## Troubleshooting

### No cached results found

```
Error: No cached results for video_xxx, skipping...
```

**Solution:** Run retrieval first:
```bash
python run_retrieval_medvidqa.py --dataset val --num-samples 10
```

### MedVidQA data not found

```
Error: File not found: ./MedVidQA/val.json
```

**Solution:** Extract the MedVidQA archive:
```bash
# Archive should already be extracted
ls -la MedVidQA/
```

### VideoStir pipeline errors

```
Error running pipeline: [specific error]
```

**Solution:**
1. Check VideoStir dependencies installed
2. Verify video files exist
3. Check disk space for temporary files
4. Review pipeline logs in output directory

## Citation

If you use this evaluation framework, cite:

```bibtex
@inproceedings{videostir2026,
  title={VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG},
  author={...},
  booktitle={ACL 2026},
  year={2026}
}
```

And cite MedVidQA:

```bibtex
@article{medvidqa2022,
  title={MedVidQA: Efficient Video Understanding with Dynamic Vision Language Modeling},
  author={...},
  journal={...},
  year={2022}
}
```

## Support

For issues or questions:
1. Check demo_eval.py for examples
2. Review PIPELINE.md for system architecture
3. Check VideoStir inference logs
4. Inspect cached pipeline outputs

---

**Last Updated:** 2026-08-23
