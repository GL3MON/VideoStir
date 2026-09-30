# MedVidQA Retrieval Evaluation Framework - Summary

## What Was Created

A comprehensive system to evaluate the **retrieval performance** of VideoStir on the MedVidQA medical video QA dataset. The system measures how well retrieved frames match the ground truth answer timestamps.

## Key Files Created

### 1. **evaluate_retrieval.py** - Main evaluation script
- Loads MedVidQA dataset (train/val/test splits)
- Compares retrieved frames with ground truth timestamps
- Computes standard IR metrics (MRR, Recall@K, MAP@K, Coverage)
- Saves results to JSON format
- Generates summary statistics

**Key Features:**
- Flexible -K cutoff evaluation
- Temporal tolerance (±2 seconds) for ground truth matching
- Coverage metric for time-range span distribution
- Sample-level and aggregate metrics

**Usage:**
```bash
python evaluate_retrieval.py --dataset val --num-samples 5
```

### 2. **run_retrieval_medvidqa.py** - Run retrieval pipeline
- Executes VideoStir pipeline on MedVidQA samples
- Caches retrieval results for evaluation
- Handles missing videos gracefully
- Reports processing statistics

**Usage:**
```bash
python run_retrieval_medvidqa.py --dataset val --num-samples 10
```

### 3. **generate_eval_report.py** - Report generation
- Creates detailed evaluation reports
- Generates multiple output formats:
  - Text report (detailed analysis)
  - Markdown report (GitHub-friendly)
  - CSV export (Excel/analysis tools)
- Includes recommendations and analysis

**Usage:**
```bash
python generate_eval_report.py --results eval_results/eval_results.json
```

### 4. **demo_eval.py** - Interactive demo
- Demonstrates the evaluation system
- Shows data structures
- Explains metrics
- Provides example commands
- Displays expected behavior

**Usage:**
```bash
python demo_eval.py
```

### 5. **test_evaluation.py** - Functional test
- Tests evaluation on existing test data
- Uses real MedVidQA samples + VideoStir output
- Demonstrates metric computation
- Provides interpretation

**Usage:**
```bash
python test_evaluation.py
```

### 6. **EVAL_README.md** - Comprehensive documentation
- Complete guide to the evaluation framework
- Workflow instructions
- Metric explanations
- Debugging guide
- Advanced usage examples

## Evaluation Metrics

### MRR (Mean Reciprocal Rank)
**What:** Position of first relevant frame in ranking

$$\text{MRR} = \frac{1}{\text{rank of first relevant item}}$$

**Interpretation:**
- 1.0 = Perfect (first frame is relevant)
- 0.5 = First relevant at rank 2
- 0.2 = First relevant at rank 5
- **Target:** > 0.5

### Recall@K
**What:** Fraction of ground truth content in top-K results

$$\text{Recall@K} = \frac{\text{# relevant in top-K}}{\text{# in ground truth}}$$

**Interpretation:**
- Measures earliness and completeness
- Recall@25 = Coverage in first 25 frames
- Recall@128 = Complete coverage

**Targets:**
- Recall@25 > 0.6 (60% coverage in top-25)
- Recall@128 > 0.8 (80% overall coverage)

### MAP@K (Mean Average Precision)
**What:** Average precision at relevant positions

**Interpretation:**
- Balances finding relevant + ranking them highly
- Penalizes non-relevant items between relevant ones

**Target:** > 0.6 at K=25

### Temporal Coverage@128
**What:** % of answer time range covered by retrieved frames

**Algorithm:**
1. Divide ground truth time range into 10 buckets
2. Count covered buckets
3. Coverage = (# covered) / 10

**Interpretation:**
- Ensures frames span across answer (not clustered)
- Complements recall by measuring distribution

**Target:** > 0.7 (covers 7+ buckets)

## System Architecture

```
MedVidQA Dataset (train/val/test JSON)
                    ↓
        run_retrieval_medvidqa.py
        (Run VideoStir on each sample)
                    ↓
        Cached retrieval results (JSON)
                    ↓
        evaluate_retrieval.py
        (Compute metrics against ground truth)
                    ↓
        eval_results.json (Metrics)
                    ↓
        generate_eval_report.py
        (Create reports & analysis)
                    ↓
        evaluation_report.{txt,md,csv}
```

## Quick Start (5 minutes)

### 1. View Demo
```bash
python demo_eval.py
```

### 2. Run Quick Test  
```bash
python test_evaluation.py
```

### 3. Evaluate on Real Data
```bash
# (Requires cached retrieval results or run them)
python evaluate_retrieval.py --dataset val
```

### 4. View Results
```bash
cat eval_results/eval_results.json | python -m json.tool
```

## Complete Workflow (Production)

### Step 1: Run Retrieval (30 min - 2 hours)
If you have MedVidQA videos:
```bash
python run_retrieval_medvidqa.py --dataset val
```

### Step 2: Evaluate (5 minutes)
```bash
python evaluate_retrieval.py --dataset val
```

### Step 3: Generate Reports (2 minutes)
```bash
python generate_eval_report.py --results eval_results/eval_results.json
```

### Step 4: View Reports
```bash
# Text report
cat eval_reports/evaluation_report.txt

# Markdown (GitHub)
cat eval_reports/evaluation_report.md

# CSV for Excel
cat eval_reports/sample_results.csv
```

## Output Files

### Evaluation Results
- `eval_results/eval_results.json` - Complete metrics
  - Aggregate statistics
  - Per-sample results
  - All metric values

### Reports
- `eval_reports/evaluation_report.txt` - Detailed text analysis
- `eval_reports/evaluation_report.md` - GitHub markdown format
- `eval_reports/sample_results.csv` - Spreadsheet format

## Interpreting Results

### Excellent System
```
MRR: 0.65 ± 0.25
Recall@25: 0.72 ± 0.18
Recall@128: 0.89 ± 0.10
Coverage@128: 0.78 ± 0.15
```
→ Consistently finds relevant frames early with good coverage

### Good System
```
MRR: 0.50 ± 0.25
Recall@25: 0.60 ± 0.20
Recall@128: 0.78 ± 0.12
Coverage@128: 0.70 ± 0.18
```
→ Generally effective but some room for improvement

### Needs Work
```
MRR: 0.25 ± 0.25
Recall@25: 0.35 ± 0.20
Recall@128: 0.50 ± 0.20
Coverage@128: 0.45 ± 0.25
```
→ Significant issues in retrieval stage

## Debugging Guide

### Low MRR
- Problem: First relevant frame appears late
- Causes:
  - Intent analysis not understanding query
  - Visual embeddings misaligned
  - Query preprocessing issues
- Fix: Check intent analyzer, verify embeddings

### Low Recall@25
- Problem: Missing relevant content in top-25
- Causes:
  - top_k too small
  - Spatial expansion weak
  - Semantic search failing
- Fix: Increase top-k, verify graph connections

### Low Coverage
- Problem: Retrieved frames clustered in time
- Causes:
  - Temporal edges not working
  - Multiple answer segments not connected
  - Graph creation issues
- Fix: Check temporal edge weights, verify segmentation

### Low Reranking Quality
- Problem: Frames not properly ranked by relevance
- Causes:
  - Reranker LoRA adapter issues
  - Prompt formatting problems
  - Model not loaded correctly
- Fix: Verify adapter path, test reranker independently

## Key Design Decisions

1. **Temporal Tolerance (±2 seconds)**
   - Ground truth is segment boundaries, not pixel-perfect
   - Adjacent frames near boundaries considered relevant
   - Prevents penalizing near-misses

2. **Coverage Metric**
   - Prevents gaming recall by clustering frames
   - Ensures temporal diversity in results
   - Complements recall measurement

3. **Recall Normalization**
   - Scales recall by expected ground truth size
   - Handles questions with different answer lengths fairly
   - Prevents bias toward shorter/longer answers

4. **Multi-Level Evaluation**
   - Sample-level: Understand per-question performance
   - Aggregate: Understand overall system quality
   - Both needed for comprehensive analysis

## Extending the Framework

### Adding New Metrics
Edit `evaluate_retrieval.py` to add:
```python
def _compute_ndcg(self, ...):
    # Implement NDCG metric
    pass
```

### Analyzing Specific Questions
```python
results = json.load(open("eval_results/eval_results.json"))
medical = [r for r in results["sample_results"] 
          if "medical" in r["question"]]
```

### Comparing Configurations
Run evaluation multiple times with different settings:
```bash
for k in 3 5 10; do
    python run_retrieval_medvidqa.py --top-k $k
    python evaluate_retrieval.py --output-file eval_k${k}.json
done
```

## Expected Performance

Based on VideoStir paper and MedVidQA characteristics:

**Typical Medical QA (Answer 10-30 seconds):**
- MRR: 0.45-0.65
- Recall@25: 0.55-0.75
- Recall@128: 0.75-0.90

**Short Queries (< 30 seconds):**
- Better performance (less ambiguity)

**Long Queries (> 60 seconds):**
- Lower performance (temporal coverage harder)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No cached results found" | Run `run_retrieval_medvidqa.py` first |
| "MedVidQA data not found" | Verify MedVidQA/ folder exists with JSON files |
| Pipeline errors | Check VideoStir logs in pipeline output directory |
| Import errors | Install dependencies: `pip install numpy` |
| Out of memory | Reduce num-samples or batch-size |

## Citation

```bibtex
@inproceedings{videostir2026,
  title={VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG},
  author={...},
  booktitle={ACL 2026},
  year={2026}
}
```

## Support

1. Read EVAL_README.md for detailed documentation
2. Run demo_eval.py to understand system
3. Check pipeline logs in artifacts/output/*/logs/
4. Review cached results in eval_results/

---

**Created:** August 23, 2026  
**Framework Version:** 1.0  
**Status:** Fully functional ✓
