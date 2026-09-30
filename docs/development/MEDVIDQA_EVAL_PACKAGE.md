# MedVidQA Evaluation - Modularized Package

Clean, organized evaluation framework for VideoStir retrieval on MedVidQA.

## Installation

```bash
# No special installation needed - it's a local package
cd VideoStir
python -m medvidqa_eval --help
```

## Quick Start

### 1. View Dataset Info
```bash
python -m medvidqa_eval info
# or
python medvidqa_eval.py info
```

### 2. Evaluate Retrieval (5 minutes)
```bash
python -m medvidqa_eval evaluate --dataset val --num-samples 5
```

### 3. Generate Reports (2 minutes)
```bash
python -m medvidqa_eval report --results eval_results/eval_results.json
```

## Package Structure

```
medvidqa_eval/
├── __init__.py           # Package initialization, public API
├── __main__.py           # Entry point for: python -m medvidqa_eval
├── cli.py                # Command-line interface
│
├── metrics/
│   ├── __init__.py
│   └── evaluator.py      # MedVidQAEvaluator class
│
├── reports/
│   ├── __init__.py
│   └── generator.py      # EvaluationReportGenerator class
│
├── retrieval/
│   ├── __init__.py
│   └── runner.py         # MedVidQARetriever class
│
└── utils/
    ├── __init__.py
    └── helpers.py        # Utility functions
```

## Usage Patterns

### Command Line

```bash
# Evaluate
python -m medvidqa_eval evaluate --dataset val --num-samples 10

# Generate reports
python -m medvidqa_eval report --results eval_results/eval_results.json

# Run retrieval
python -m medvidqa_eval retrieve --dataset val --num-samples 5

# View dataset info
python -m medvidqa_eval info
```

### Python API

```python
from medvidqa_eval import MedVidQAEvaluator, EvaluationReportGenerator

# Evaluate
evaluator = MedVidQAEvaluator()
results, aggregate = evaluator.evaluate_dataset("val", num_samples=10)
evaluator.save_results(results, aggregate)

# Generate reports
generator = EvaluationReportGenerator()
generator.generate_text_report(aggregate, results)
generator.generate_markdown_report(aggregate, results)
generator.generate_csv_export(results)
```

### Run Retrieval

```python
from medvidqa_eval import MedVidQARetriever

retriever = MedVidQARetriever(
    medvidqa_root="./MedVidQA",
    video_root="./videos",
    output_dir="./medvidqa_retrieval",
)

stats = retriever.run_on_dataset("val", num_samples=10)
```

## Commands

### evaluate

Evaluate retrieval performance on MedVidQA.

```bash
python -m medvidqa_eval evaluate [OPTIONS]
```

**Options:**
- `--dataset {train,val,test}` - Dataset to evaluate (default: val)
- `--num-samples N` - Limit to N samples (default: all)
- `--medvidqa-root PATH` - MedVidQA root directory (default: ./MedVidQA)
- `--output-dir PATH` - Output directory (default: ./eval_results)
- `--output-file NAME` - Output JSON filename (default: eval_results.json)

**Output:**
- `eval_results/eval_results.json` - Complete metrics

### report

Generate evaluation reports in multiple formats.

```bash
python -m medvidqa_eval report [OPTIONS]
```

**Options:**
- `--results FILE` - Path to eval_results.json (required)
- `--output-dir PATH` - Output directory (default: ./eval_reports)

**Output:**
- `eval_reports/evaluation_report.txt` - Detailed text analysis
- `eval_reports/evaluation_report.md` - GitHub markdown format
- `eval_reports/sample_results.csv` - Spreadsheet export

### retrieve

Run VideoStir retrieval pipeline on MedVidQA samples.

```bash
python -m medvidqa_eval retrieve [OPTIONS]
```

**Options:**
- `--dataset {train,val,test}` - Dataset to process (default: val)
- `--num-samples N` - Limit to N samples (default: all)
- `--medvidqa-root PATH` - MedVidQA root directory (default: ./MedVidQA)
- `--video-root PATH` - Video directory (default: ./videos)
- `--output-dir PATH` - Cache directory (default: ./medvidqa_retrieval)
- `--skip-existing` - Skip cached results (default: true)

**Output:**
- `medvidqa_retrieval/{video_id}_{sample_id}_frames.json` - Cached retrieval results

### info

Show MedVidQA dataset statistics.

```bash
python -m medvidqa_eval info [OPTIONS]
```

**Options:**
- `--medvidqa-root PATH` - MedVidQA root directory (default: ./MedVidQA)

**Output:**
```
MedVidQA Dataset Statistics
========================================
train     :  2710 samples
val       :   145 samples
test      :   155 samples
Total     :  3010 samples
```

## Complete Workflow

```bash
# Step 1: View dataset
python -m medvidqa_eval info

# Step 2: Evaluate (if you have cached retrieval results)
python -m medvidqa_eval evaluate --dataset val

# Step 3: Generate reports
python -m medvidqa_eval report --results eval_results/eval_results.json

# Step 4: View results
cat eval_reports/evaluation_report.txt
cat eval_reports/evaluation_report.md
```

## Python API Examples

### Simple Evaluation

```python
from medvidqa_eval import MedVidQAEvaluator

evaluator = MedVidQAEvaluator()
results, aggregate = evaluator.evaluate_dataset("val", num_samples=5)
print(f"MRR: {aggregate['mrr_avg']:.4f}")
```

### Generate All Reports

```python
from medvidqa_eval import EvaluationReportGenerator

generator = EvaluationReportGenerator()
aggregate, samples = generator.load_results("eval_results/eval_results.json")

generator.generate_text_report(aggregate, samples)
generator.generate_markdown_report(aggregate, samples)
generator.generate_csv_export(samples)

print("Reports generated in eval_reports/")
```

### Custom Evaluation Loop

```python
from medvidqa_eval import MedVidQAEvaluator
import json

evaluator = MedVidQAEvaluator(output_dir="./my_eval")
results, aggregate = evaluator.evaluate_dataset("val", num_samples=20)

# Custom analysis
for sample in results:
    if sample["mrr"] < 0.2:
        print(f"Poor performance: {sample['question']}")

# Save custom results
evaluator.save_results(results, aggregate, "custom_eval.json")
```

## Metrics Explained

| Metric | Definition | Good Value |
|--------|-----------|-----------|
| **MRR** | Position of first relevant frame | > 0.5 |
| **Recall@K** | % of answer content in top-K frames | Recall@25 > 0.6 |
| **MAP@K** | Average precision at K | > 0.6 |
| **Coverage@128** | % of answer time range covered | > 0.7 |

See [EVAL_README.md](./EVAL_README.md) for detailed metric explanations.

## File Organization

### Generated Directories

```
eval_results/
├── eval_results.json              # Metrics
└── processing_stats_*.json        # Statistics

eval_reports/
├── evaluation_report.txt          # Text analysis
├── evaluation_report.md           # GitHub format
└── sample_results.csv             # Spreadsheet

medvidqa_retrieval/
└── {video_id}_{sample_id}_frames.json  # Cached frames
```

## Import Examples

```python
# Single imports
from medvidqa_eval import MedVidQAEvaluator
from medvidqa_eval.metrics import MedVidQAEvaluator
from medvidqa_eval.reports import EvaluationReportGenerator
from medvidqa_eval.retrieval import MedVidQARetriever

# Utility imports
from medvidqa_eval import setup_logging, get_medvidqa_stats
from medvidqa_eval.utils import ensure_directory

# Direct class imports
from medvidqa_eval.metrics.evaluator import MedVidQAEvaluator
from medvidqa_eval.reports.generator import EvaluationReportGenerator
from medvidqa_eval.retrieval.runner import MedVidQARetriever
```

## Logging

```python
from medvidqa_eval import setup_logging
import logging

setup_logging(level=logging.DEBUG)  # Enable debug logging
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named medvidqa_eval" | Run from VideoStir root directory |
| "MedVidQA not found" | Ensure MedVidQA/ folder exists with JSON files |
| "No cached results" | Run `python -m medvidqa_eval retrieve` first |
| Import errors | Check package structure: `ls medvidqa_eval/` |

## Benefits of Modularization

✅ **Organized** - Clear separation of concerns
✅ **Reusable** - Import individual components
✅ **Testable** - Each module can be tested independently
✅ **Maintainable** - Easy to update/fix specific components
✅ **Extensible** - Simple to add new features
✅ **Professional** - Standard Python package structure

## Next Steps

1. Read [EVAL_README.md](./EVAL_README.md) for detailed guide
2. Try `python -m medvidqa_eval --help`
3. Run `python -m medvidqa_eval info` to verify setup
4. Start with `python -m medvidqa_eval evaluate --num-samples 5`

---

**Version:** 1.0.0  
**Status:** Fully Functional ✓
