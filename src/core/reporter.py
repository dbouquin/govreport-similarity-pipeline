"""
Statistical reporting and analysis for similarity results.
Generates data-focused reports without subjective interpretations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..utils.config import Config


class SimilarityReporter:
    """
    Generates data-focused reports from similarity analysis results.
    Provides statistical analysis and metrics without subjective interpretations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the reporter.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger("reporter")
    
    def generate_report(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        include_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive data-focused report from analysis results.
        
        Args:
            input_path: Path to analysis results (CSV or JSON)
            output_path: Where to save the report
            include_plots: Whether to generate visualization plots (unused - simplified)
            
        Returns:
            Report dictionary
        """
        try:
            self.logger.info("Generating similarity report")
            
            # Load analysis results (prioritizing JSON for metadata)
            results = self._load_analysis_results(input_path)
            
            # Generate report sections - focused on data only
            report = {
                "report_metadata": {
                    "input_path": str(input_path),
                    "report_timestamp": datetime.now().isoformat(),
                    "analysis_metadata": results.get("analysis_metadata", {}),
                    "generator": "Government Report Similarity Pipeline"
                },
                "statistical_analysis": self._generate_statistical_analysis(results),
                "quality_metrics": self._generate_quality_metrics(results),
                "distribution_analysis": self._generate_distribution_analysis(results),
                "tranches_analysis": results.get("distance_tranches", {})
            }
            
            # Save report if output path provided
            if output_path:
                self._save_report(report, output_path)
            
            self.logger.info("Report generation completed successfully")
            return report
                
        except Exception as e:
            error_msg = f"Failed to generate report: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _load_analysis_results(self, input_path: Path) -> Dict[str, Any]:
        """
        Load analysis results from file, prioritizing JSON to preserve metadata.
        
        Args:
            input_path: Path to results file
            
        Returns:
            Analysis results dictionary
        """
        try:
            # First, try to load the JSON version to get full metadata
            json_path = input_path.with_suffix(".json")
            if json_path.exists():
                self.logger.info(f"Loading results from JSON: {json_path}")
                with open(json_path, 'r') as f:
                    return json.load(f)
            
            # If input is specifically JSON
            elif input_path.suffix.lower() == ".json":
                self.logger.info(f"Loading results from JSON: {input_path}")
                with open(input_path, 'r') as f:
                    return json.load(f)
            
            # Fallback to CSV (with limited metadata)
            elif input_path.suffix.lower() == ".csv":
                self.logger.warning(f"Loading from CSV - metadata will be limited: {input_path}")
                df = pd.read_csv(input_path)
                return {
                    "similarity_scores": df["similarity_score"].dropna().tolist(),
                    "sample_metadata": df.to_dict('records'),
                    "analysis_metadata": {
                        "note": "Limited metadata available from CSV file",
                        "csv_source": str(input_path)
                    },
                    "processing_stats": {
                        "samples_processed": len(df.dropna(subset=["similarity_score"])),
                        "samples_failed": len(df) - len(df.dropna(subset=["similarity_score"]))
                    }
                }
            
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load results from {input_path}: {e}") from e
    
    def _generate_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed statistical analysis.
        
        Args:
            results: Analysis results
            
        Returns:
            Statistical analysis dictionary
        """
        try:
            similarity_scores = results.get("similarity_scores", [])
            sample_metadata = results.get("sample_metadata", [])
            
            if not similarity_scores:
                return {"error": "No similarity scores available for analysis"}
            
            scores = np.array(similarity_scores)
            
            # Descriptive statistics
            stats = {
                "descriptive_statistics": {
                    "count": len(scores),
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores)),
                    "standard_deviation": float(np.std(scores)),
                    "variance": float(np.var(scores)),
                    "minimum": float(np.min(scores)),
                    "maximum": float(np.max(scores)),
                    "range": float(np.max(scores) - np.min(scores)),
                    "skewness": float(self._calculate_skewness(scores)),
                    "kurtosis": float(self._calculate_kurtosis(scores))
                },
                "percentiles": {
                    "5th": float(np.percentile(scores, 5)),
                    "10th": float(np.percentile(scores, 10)),
                    "25th": float(np.percentile(scores, 25)),
                    "50th": float(np.percentile(scores, 50)),
                    "75th": float(np.percentile(scores, 75)),
                    "90th": float(np.percentile(scores, 90)),
                    "95th": float(np.percentile(scores, 95))
                },
                "confidence_intervals": self._calculate_confidence_intervals(scores)
            }
            
            # Length analysis if available
            if sample_metadata:
                stats["length_analysis"] = self._analyze_content_lengths(
                    sample_metadata, similarity_scores
                )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to generate statistical analysis: {e}")
            return {"error": str(e)}
    
    def _generate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate objective quality metrics without subjective assessments.
        
        Args:
            results: Analysis results
            
        Returns:
            Quality metrics dictionary
        """
        try:
            similarity_scores = results.get("similarity_scores", [])
            
            if not similarity_scores:
                return {"error": "No similarity scores available for metrics"}
            
            scores = np.array(similarity_scores)
            
            # Define thresholds
            thresholds = {
                "very_high": 0.9,
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }
            
            # Calculate counts and percentages for each threshold
            metrics = {
                "threshold_analysis": {},
                "processing_stats": results.get("processing_stats", {}),
                "sample_distribution": {}
            }
            
            total = len(scores)
            
            # Threshold-based analysis
            for label, threshold in thresholds.items():
                count = int(np.sum(scores >= threshold))
                percentage = float(count / total * 100)
                metrics["threshold_analysis"][f"{label}_similarity"] = {
                    "threshold": threshold,
                    "count": count,
                    "percentage": percentage
                }
            
            # Range-based distribution
            ranges = [
                ("very_high", 0.9, 1.0),
                ("high", 0.8, 0.9),
                ("medium", 0.6, 0.8),
                ("low", 0.4, 0.6),
                ("very_low", 0.0, 0.4)
            ]
            
            for label, lower, upper in ranges:
                if lower == 0.0:
                    count = int(np.sum(scores < upper))
                else:
                    count = int(np.sum((scores >= lower) & (scores < upper)))
                percentage = float(count / total * 100)
                
                metrics["sample_distribution"][label] = {
                    "range": f"{lower} - {upper}",
                    "count": count,
                    "percentage": percentage
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to generate quality metrics: {e}")
            return {"error": str(e)}
    
    def _generate_distribution_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate distribution analysis of similarity scores.
        
        Args:
            results: Analysis results
            
        Returns:
            Distribution analysis dictionary
        """
        try:
            similarity_scores = results.get("similarity_scores", [])
            
            if not similarity_scores:
                return {"error": "No similarity scores available for distribution analysis"}
            
            scores = np.array(similarity_scores)
            
            # Create histogram
            bins = 20
            hist, bin_edges = np.histogram(scores, bins=bins, range=(0, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            distribution = {
                "histogram": {
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": bin_centers.tolist(),
                    "counts": hist.tolist(),
                    "frequencies": (hist / len(scores)).tolist(),
                    "bin_width": float(bin_edges[1] - bin_edges[0])
                },
                "distribution_properties": {
                    "is_normal_approximate": bool(self._test_normality(scores)),
                    "is_symmetric_approximate": bool(abs(self._calculate_skewness(scores)) < 0.5),
                    "has_outliers": bool(self._detect_outliers(scores)),
                    "peak_location": float(bin_centers[np.argmax(hist)]),
                    "concentration_coefficient": float(self._calculate_concentration(scores, hist, bin_centers))
                }
            }
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Failed to generate distribution analysis: {e}")
            return {"error": str(e)}
    
    def _save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """
        Save report to files.
        
        Args:
            report: Report dictionary
            output_path: Where to save report
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save full report as JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=self._json_serializer)
            
            # Save summary statistics as CSV
            csv_path = output_path.parent / f"{output_path.stem}_summary.csv"
            self._save_summary_csv(report, csv_path)
            
            self.logger.info(f"Report saved to {json_path} and {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for NumPy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _save_summary_csv(self, report: Dict[str, Any], csv_path: Path) -> None:
        """
        Save summary statistics as CSV.
        
        Args:
            report: Report dictionary
            csv_path: Path to save CSV
        """
        try:
            summary_data = []
            
            # Extract statistical metrics
            stats = report.get("statistical_analysis", {})
            quality_metrics = report.get("quality_metrics", {})
            
            if "error" not in stats:
                desc_stats = stats.get("descriptive_statistics", {})
                for metric, value in desc_stats.items():
                    summary_data.append({"Category": "Descriptive Statistics", "Metric": metric, "Value": value})
                
                percentiles = stats.get("percentiles", {})
                for metric, value in percentiles.items():
                    summary_data.append({"Category": "Percentiles", "Metric": f"{metric}_percentile", "Value": value})
            
            if "error" not in quality_metrics:
                threshold_analysis = quality_metrics.get("threshold_analysis", {})
                for metric, data in threshold_analysis.items():
                    summary_data.append({"Category": "Threshold Analysis", "Metric": f"{metric}_count", "Value": data.get("count", 0)})
                    summary_data.append({"Category": "Threshold Analysis", "Metric": f"{metric}_percentage", "Value": data.get("percentage", 0)})
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(csv_path, index=False)
            
        except Exception as e:
            self.logger.warning(f"Failed to save summary CSV: {e}")
    
    # Helper methods for statistical calculations
    def _calculate_skewness(self, scores: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        mean = np.mean(scores)
        std = np.std(scores)
        return np.mean(((scores - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, scores: np.ndarray) -> float:
        """Calculate kurtosis of distribution."""
        mean = np.mean(scores)
        std = np.std(scores)
        return np.mean(((scores - mean) / std) ** 4) - 3
    
    def _calculate_confidence_intervals(self, scores: np.ndarray) -> Dict[str, float]:
        """Calculate confidence intervals for the mean."""
        try:
            from scipy import stats
            mean = np.mean(scores)
            sem = stats.sem(scores)
            ci_95 = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=sem)
            ci_99 = stats.t.interval(0.99, len(scores)-1, loc=mean, scale=sem)
            
            return {
                "95_percent_lower": float(ci_95[0]),
                "95_percent_upper": float(ci_95[1]),
                "99_percent_lower": float(ci_99[0]),
                "99_percent_upper": float(ci_99[1])
            }
        except ImportError:
            # Fallback without scipy
            mean = np.mean(scores)
            std_err = np.std(scores) / np.sqrt(len(scores))
            return {
                "95_percent_lower": float(mean - 1.96 * std_err),
                "95_percent_upper": float(mean + 1.96 * std_err),
                "99_percent_lower": float(mean - 2.58 * std_err),
                "99_percent_upper": float(mean + 2.58 * std_err)
            }
    
    def _analyze_content_lengths(
        self,
        sample_metadata: List[Dict[str, Any]],
        similarity_scores: List[float]
    ) -> Dict[str, Any]:
        """Analyze relationship between content lengths and similarity."""
        try:
            report_lengths = []
            summary_lengths = []
            scores = []
            
            for meta, score in zip(sample_metadata, similarity_scores):
                if (meta.get("report_length") and 
                    meta.get("summary_length") and 
                    score is not None):
                    report_lengths.append(meta["report_length"])
                    summary_lengths.append(meta["summary_length"])
                    scores.append(score)
            
            if not report_lengths:
                return {"error": "No length data available"}
            
            # Calculate correlations
            report_corr = np.corrcoef(report_lengths, scores)[0, 1]
            summary_corr = np.corrcoef(summary_lengths, scores)[0, 1]
            
            return {
                "report_length_correlation": float(report_corr) if not np.isnan(report_corr) else 0.0,
                "summary_length_correlation": float(summary_corr) if not np.isnan(summary_corr) else 0.0,
                "average_report_length": float(np.mean(report_lengths)),
                "average_summary_length": float(np.mean(summary_lengths)),
                "length_ratio_statistics": {
                    "mean_ratio": float(np.mean([s/r for r, s in zip(report_lengths, summary_lengths) if r > 0])),
                    "median_ratio": float(np.median([s/r for r, s in zip(report_lengths, summary_lengths) if r > 0]))
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _test_normality(self, scores: np.ndarray) -> bool:
        """Test if distribution is approximately normal."""
        skew = abs(self._calculate_skewness(scores))
        kurt = abs(self._calculate_kurtosis(scores))
        return skew < 1.0 and kurt < 3.0
    
    def _detect_outliers(self, scores: np.ndarray) -> bool:
        """Detect if there are significant outliers."""
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return np.any((scores < lower_bound) | (scores > upper_bound))
    
    def _calculate_concentration(
        self,
        scores: np.ndarray,
        hist: np.ndarray,
        bin_centers: np.ndarray
    ) -> float:
        """Calculate concentration around the peak."""
        peak_idx = np.argmax(hist)
        peak_value = bin_centers[peak_idx]
        
        # Calculate what percentage of scores are within 0.1 of the peak
        within_peak = np.sum(np.abs(scores - peak_value) <= 0.1)
        return float(within_peak / len(scores))