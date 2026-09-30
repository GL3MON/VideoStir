"""MedVidQA Retrieval Evaluation Package

Comprehensive evaluation framework for VideoStir retrieval on MedVidQA dataset.

Example:
    >>> from videostir.eval import MedVidQAEvaluator
    >>> evaluator = MedVidQAEvaluator()
    >>> results, aggregate = evaluator.evaluate_dataset("val", num_samples=10)

    >>> from videostir.eval import EvaluationReportGenerator
    >>> generator = EvaluationReportGenerator()
    >>> generator.generate_text_report(aggregate, results)

Modules:
    metrics: Core evaluation metrics and evaluation logic
    reports: Report generation in multiple formats  
    retrieval: VideoStir pipeline execution
    utils: Utility functions and helpers
    cli: Command-line interface
"""

__version__ = "1.0.0"
__author__ = "VideoStir Team"

from .metrics import MedVidQAEvaluator
from .reports import EvaluationReportGenerator
from .retrieval import MedVidQARetriever
from .utils import (
    setup_logging,
    ensure_directory,
    get_medvidqa_stats,
    print_metrics_summary,
)

__all__ = [
    "MedVidQAEvaluator",
    "EvaluationReportGenerator",
    "MedVidQARetriever",
    "setup_logging",
    "ensure_directory",
    "get_medvidqa_stats",
    "print_metrics_summary",
]


def evaluate_dataset(dataset="val", num_samples=None, medvidqa_root="./MedVidQA",
                     output_dir="./eval_results", retrieval_dir="./medvidqa_retrieval"):
    """Quick evaluation on a dataset.
    
    Args:
        dataset: "train", "val", or "test"
        num_samples: Limit number of samples (None = all)
        medvidqa_root: Path to MedVidQA data
        output_dir: Output directory
        retrieval_dir: Directory with cached retrieval frames
        
    Returns:
        Tuple of (results, aggregate_metrics)
    """
    evaluator = MedVidQAEvaluator(
        medvidqa_root=medvidqa_root,
        output_dir=output_dir,
        retrieval_dir=retrieval_dir,
    )
    return evaluator.evaluate_dataset(dataset=dataset, num_samples=num_samples)


def generate_report(results_file, output_dir="./eval_reports"):
    """Generate reports from evaluation results.
    
    Args:
        results_file: Path to eval_results.json
        output_dir: Where to save reports
    """
    generator = EvaluationReportGenerator(output_dir=output_dir)
    aggregate, samples = generator.load_results(results_file)
    generator.generate_text_report(aggregate, samples)
    generator.generate_markdown_report(aggregate, samples)
    generator.generate_csv_export(samples)


def run_retrieval(dataset="val", num_samples=None, medvidqa_root="./MedVidQA",
                  video_root="./videos", output_dir="./medvidqa_retrieval"):
    """Run VideoStir retrieval on MedVidQA samples.
    
    Args:
        dataset: "train", "val", or "test"
        num_samples: Limit number of samples (None = all)
        medvidqa_root: Path to MedVidQA data
        video_root: Path to video directory
        output_dir: Cache directory for results
        
    Returns:
        Statistics dictionary
    """
    retriever = MedVidQARetriever(
        medvidqa_root=medvidqa_root,
        video_root=video_root,
        output_dir=output_dir,
    )
    return retriever.run_on_dataset(dataset=dataset, num_samples=num_samples)
