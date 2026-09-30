"""Report generation for evaluation results."""

import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """Generate detailed evaluation reports from metrics."""

    def __init__(self, output_dir: str = "./eval_reports"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, results_file: str) -> tuple:
        """Load evaluation results from JSON file."""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        return data.get("aggregate_metrics", {}), data.get("sample_results", [])

    def generate_text_report(
        self,
        aggregate: Dict,
        samples: List[Dict],
        output_file: str = "evaluation_report.txt",
    ) -> None:
        """Generate comprehensive text report."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VIDEO RETRIEVAL EVALUATION REPORT (MedVidQA)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples evaluated: {aggregate.get('num_samples', 0)}\n")
            f.write(f"Evaluation method: Temporal overlap with ground truth answer range\n\n")
            
            f.write("SUMMARY METRICS\n")
            f.write("-" * 80 + "\n")
            
            mrr_avg = aggregate.get("mrr_avg", 0)
            mrr_std = aggregate.get("mrr_std", 0)
            f.write(f"Mean Reciprocal Rank (MRR):\n")
            f.write(f"  Mean: {mrr_avg:.4f} ± {mrr_std:.4f}\n")
            f.write(f"  Interpretation: On average, first relevant frame is at rank {1/max(mrr_avg, 0.001):.1f}\n\n")
            
            f.write("RETRIEVAL METRICS BY CUTOFF\n")
            f.write("-" * 80 + "\n")
            
            for k in [10, 25, 50, 128]:
                recall_key = f"recall@{k}_avg"
                map_key = f"map@{k}_avg"
                
                if recall_key in aggregate:
                    recall_avg = aggregate[recall_key]
                    recall_std = aggregate.get(f"recall@{k}_std", 0)
                    f.write(f"\n@{k}:\n")
                    f.write(f"  Recall@{k}:  {recall_avg:.4f} ± {recall_std:.4f} ({recall_avg*100:.1f}%)\n")
                
                if map_key in aggregate:
                    map_avg = aggregate[map_key]
                    map_std = aggregate.get(f"map@{k}_std", 0)
                    f.write(f"  MAP@{k}:     {map_avg:.4f} ± {map_std:.4f} ({map_avg*100:.1f}%)\n")
            
            if "coverage_avg" in aggregate:
                f.write(f"\nTemporal Coverage @128:\n")
                f.write(f"  Mean: {aggregate['coverage_avg']:.4f} ± {aggregate.get('coverage_std', 0):.4f}\n")
                f.write(f"  Interpretation: Retrieved frames cover {aggregate['coverage_avg']*100:.1f}% of answer buckets\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS (sorted by MRR)\n")
            f.write("=" * 80 + "\n\n")
            
            sorted_samples = sorted(samples, key=lambda x: x.get("mrr", 0), reverse=True)
            
            for i, sample in enumerate(sorted_samples[:20], 1):
                f.write(f"[{i}] Sample {sample.get('sample_id', 'N/A')}\n")
                f.write(f"    Question: {sample.get('question', 'N/A')[:70]}\n")
                f.write(f"    Ground truth: {sample.get('answer_start', 0):.1f}s - {sample.get('answer_end', 0):.1f}s\n")
                f.write(f"    Retrieved: {sample.get('retrieved_count', 0)} frames\n")
                f.write(f"    MRR: {sample.get('mrr', 0):.4f}\n")
                f.write(f"    Recall@25: {sample.get('recall@25', 0):.4f}\n")
                f.write(f"    Recall@128: {sample.get('recall@128', 0):.4f}\n")
                f.write(f"    MAP@128: {sample.get('map@128', 0):.4f}\n\n")
            
            f.write("=" * 80 + "\n" + "STATISTICAL ANALYSIS\n" + "=" * 80 + "\n\n")

            if samples:
                mrr_values = [s.get("mrr", 0) for s in samples]
                f.write("MRR Distribution:\n")
                f.write(f"  Min: {min(mrr_values):.4f}\n")
                f.write(f"  Max: {max(mrr_values):.4f}\n")
                f.write(f"  Median: {statistics.median(mrr_values):.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Report saved to {output_path}")

    def generate_csv_export(
        self,
        samples: List[Dict],
        output_file: str = "sample_results.csv",
    ) -> None:
        """Export detailed sample results to CSV."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("sample_id,question,gt_start,gt_end,retrieved_count,mrr,")
            f.write("recall@10,recall@25,recall@50,recall@128,")
            f.write("map@10,map@25,map@50,map@128,coverage@128\n")
            
            for sample in samples:
                f.write(f"{sample.get('sample_id', '')},")
                f.write(f"\"{sample.get('question', '').replace(chr(34), chr(34)*2)}\",")
                f.write(f"{sample.get('answer_start', '')},")
                f.write(f"{sample.get('answer_end', '')},")
                f.write(f"{sample.get('retrieved_count', '')},")
                f.write(f"{sample.get('mrr', '')},")
                f.write(f"{sample.get('recall@10', '')},")
                f.write(f"{sample.get('recall@25', '')},")
                f.write(f"{sample.get('recall@50', '')},")
                f.write(f"{sample.get('recall@128', '')},")
                f.write(f"{sample.get('map@10', '')},")
                f.write(f"{sample.get('map@25', '')},")
                f.write(f"{sample.get('map@50', '')},")
                f.write(f"{sample.get('map@128', '')},")
                f.write(f"{sample.get('coverage@128', '')}\n")
        
        logger.info(f"CSV export saved to {output_path}")

    def generate_markdown_report(
        self,
        aggregate: Dict,
        samples: List[Dict],
        output_file: str = "evaluation_report.md",
    ) -> None:
        """Generate markdown report."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("# VideoStir Retrieval Evaluation - MedVidQA\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Samples Evaluated:** {aggregate.get('num_samples', 0)}\n")
            f.write(f"- **Mean Reciprocal Rank (MRR):** {aggregate.get('mrr_avg', 0):.4f} ± {aggregate.get('mrr_std', 0):.4f}\n\n")
            
            f.write("## Metrics by Cutoff\n\n")
            f.write("| Cutoff | Recall | Recall Std | MAP | MAP Std |\n")
            f.write("|--------|--------|-----------|-----|----------|\n")
            
            for k in [10, 25, 50, 128]:
                recall_key = f"recall@{k}_avg"
                map_key = f"map@{k}_avg"
                
                if recall_key in aggregate:
                    recall = aggregate[recall_key]
                    recall_std = aggregate.get(f"recall@{k}_std", 0)
                    map_val = aggregate.get(map_key, 0)
                    map_std = aggregate.get(f"map@{k}_std", 0)
                    
                    f.write(f"| @{k:3d} | {recall:.4f} | {recall_std:.4f} | {map_val:.4f} | {map_std:.4f} |\n")
        
        logger.info(f"Markdown report saved to {output_path}")
