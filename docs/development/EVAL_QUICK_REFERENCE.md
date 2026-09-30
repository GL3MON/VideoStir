# MedVidQA Evaluation - Quick Reference Card

## One-Liners

```bash
# View demo
python demo_eval.py

# Test system
python test_evaluation.py

# Evaluate 5 validation samples
python evaluate_retrieval.py --dataset val --num-samples 5

# Generate reports
python generate_eval_report.py --results eval_results/eval_results.json

# Full pipeline (if videos available)
python run_retrieval_medvidqa.py --dataset val && \
python evaluate_retrieval.py --dataset val && \
python generate_eval_report.py --results eval_results/eval_results.json
```

## Key Metrics

| Metric | Good | Fair | Poor | What It Means |
|--------|------|------|------|---------------|
| MRR | >0.5 | 0.2-0.5 | <0.2 | Where is first relevant frame? |
| Recall@25 | >0.6 | 0.3-0.6 | <0.3 | Coverage in top 25 frames |
| Recall@128 | >0.8 | 0.5-0.8 | <0.5 | Coverage overall |
| Coverage@128 | >0.7 | 0.4-0.7 | <0.4 | Time range span |
| MAP@25 | >0.6 | 0.3-0.6 | <0.3 | Quality of ranking |

## Files Created

| File | Purpose | Command |
|------|---------|---------|
| `evaluate_retrieval.py` | Main evaluation | `python evaluate_retrieval.py --help` |
| `run_retrieval_medvidqa.py` | Run pipeline | `python run_retrieval_medvidqa.py --help` |
| `generate_eval_report.py` | Create reports | `python generate_eval_report.py --help` |
| `demo_eval.py` | Learn system | `python demo_eval.py` |
| `test_evaluation.py` | Test metrics | `python test_evaluation.py` |
| `EVAL_README.md` | Full docs | Read it! |

## Output Files

```
eval_results/
├── eval_results.json          # Complete metrics
└── processing_stats_*.json    # Run statistics

eval_reports/
├── evaluation_report.txt      # Detailed analysis
├── evaluation_report.md       # GitHub format
└── sample_results.csv         # Spreadsheet
```

## Pipeline Architecture

```
MedVidQA (Question + Ground Truth Timestamps)
                    ↓
        VideoStir Retrieval (128 frames)
                    ↓
        Compare with Ground Truth
                    ↓
        Compute Metrics (MRR, Recall, etc.)
                    ↓
        Generate Report + Analysis
```

## Common Issues

| Problem | Solution |
|---------|----------|
| No results | Run `run_retrieval_medvidqa.py` first |
| Can't import | Install: `pip install numpy` |
| Missing MedVidQA | Extract: `unzip pc594-*.zip` |
| Video not found | Download from YouTube links in JSON |
| Out of memory | Reduce `--num-samples` |

## Metric Explanations

### MRR (Mean Reciprocal Rank)
- First relevant frame at rank 1 → MRR = 1.0 (perfect)
- First relevant frame at rank 3 → MRR = 0.33
- No relevant frames → MRR = 0.0
- **Higher is better**

### Recall@K
- What % of answer is covered in top-K?
- Recall@25 = 0.8 means 80% of answer covered in top 25
- **Higher is better**

### Coverage@128
- Do frames span across the answer time?
- 1.0 = covers all 10 time buckets
- 0.5 = covers only 5 time buckets
- Prevents clustering all frames together
- **Higher is better**

### MAP@K
- Combines precision (few false positives) + recall (many true positives)
- Considers ranking order
- **Higher is better**

## Interpreting Results

```
Good Results:
✓ MRR > 0.5 (first hit typically rank 1-2)
✓ Recall@25 > 0.6 (most answer frames in top-25)
✓ Coverage > 0.7 (well-distributed across time)
→ System is working well!

Moderate Results:
⚠ MRR 0.2-0.5 (first hit typically rank 3-5)
⚠ Recall@25 0.3-0.6 (might miss some content)
⚠ Coverage 0.4-0.7 (somewhat clustered)
→ Room for improvement!

Poor Results:
✗ MRR < 0.2 (first hit rarely found)
✗ Recall@25 < 0.3 (missing most content)
✗ Coverage < 0.4 (very clustered)
→ Major issues need fixing!
```

## What Each Component Does

**Intent Analysis:**
- Understands if query is about time, subtitles, or visuals
- Extracts key phrases
- Poor intent → Low MRR

**Visual Embeddings:**
- Converts video segments to feature vectors
- Uses CLIP ViT-G/14 model
- Bad embeddings → Low Recall@25

**Spatiotemporal Graph:**
- Connects related segments
- Uses temporal + spatial edges
- Weak graph → Lower retrieval quality

**Reranker (Qwen2.5-VL):**
- Scores each frame for relevance
- Uses LoRA adapter
- Poor reranking → Good recall but wrong ranking

## Debugging by Metric

| Metric | Issue | Check | Fix |
|--------|-------|-------|-----|
| Low MRR | Can't find relevant | Intent analysis | Verify query understanding |
| Low Recall@25 | Missing content | Visual embeddings | Increase top-k per-hop |
| Low Recall@128 | Incomplete coverage | Graph structure | Check temporal edges |
| Low Coverage | Frames clustered | Temporal distribution | Add more segments |
| Low MAP | Bad ranking | Reranker quality | Check LoRA adapter |

## Environment

```bash
# Requirements
Python 3.10+
PyTorch 2.0+
CUDA GPU (recommended)

# Install
cd VideoStir/inference
pip install -r requirements_inference.txt
cd ..

pip install numpy
```

## Example Results

### Test on Medical Video
```
Sample: "How to apply pressure on wound?"
Ground Truth: 29-48 seconds
Retrieved: 128 frames

MRR: 0.0769 (first relevant at rank 13)
Recall@25: 1.0000 (100% coverage in top 25)
Recall@128: 0.4688 (47% coverage overall)
Coverage@128: 0.5000 (covers 5 of 10 time buckets)

Analysis: Moderate performance
- MRR weak (rank 13 is late)
- Good recall in first 25 but drops off
- Coverage could be better (frames not well-distributed)
```

## Next Steps

1. **Quick Learning**
   - Run: `python demo_eval.py`
   - Read: `EVAL_README.md`

2. **Test on Real Data**
   - Run: `python test_evaluation.py`
   - View: `cat eval_results/eval_results.json`

3. **Full Evaluation**
   - Run: `python evaluate_retrieval.py --dataset val`
   - Report: `python generate_eval_report.py --results eval_results/eval_results.json`
   - Analyze: `cat eval_reports/evaluation_report.txt`

4. **Improve System**
   - Identify bottleneck metrics
   - Check debugging guide
   - Adjust VideoStir parameters
   - Re-evaluate

## Contact

For detailed information:
- See `EVAL_README.md` - Comprehensive guide
- See `EVALUATION_SYSTEM_SUMMARY.md` - System overview
- Check pipeline logs in `artifacts/output/*/logs/`

---

**Last Updated:** August 23, 2026
