"""
Simplified data loading utilities for government report dataset.
Handles efficient loading of HuggingFace datasets with memory optimization.
Works seamlessly with Model2Vec's internal batching.
"""

from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path

from datasets import load_dataset, Dataset
import pandas as pd

from ..utils.logging import get_logger, DataLoadError
from ..utils.config import Config


class DatasetLoader:
    """
    Handles loading and streaming of the government report dataset.
    Simplified for memory efficiency and compatibility with Model2Vec.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the dataset loader.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger("data_loader")
        
        # Column names for the government report dataset
        self.report_column = "report"
        self.summary_column = "summary"
        self.required_columns = [self.report_column, self.summary_column]
    
    def load_dataset_streaming(
        self,
        dataset_name: Optional[str] = None,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Load dataset in streaming mode for memory efficiency.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            num_samples: Maximum number of samples to load (None for all)
            
        Yields:
            Dictionary containing report and summary pairs
            
        Raises:
            DataLoadError: If dataset loading fails
        """
        if dataset_name is None:
            dataset_name = self.config.dataset_name
        
        try:
            self.logger.info(f"Loading dataset {dataset_name} (split: {split}) in streaming mode")
            
            # Load dataset with streaming enabled
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True
            )
            
            self.logger.info("Dataset loaded successfully")
            
            # Validate dataset structure with first sample
            sample_iter = iter(dataset)
            try:
                first_sample = next(sample_iter)
                self._validate_dataset_structure(first_sample)
            except StopIteration:
                raise DataLoadError("Dataset is empty")
            
            # Process first sample
            cleaned_sample = self._clean_sample(first_sample)
            if cleaned_sample is not None:
                yield cleaned_sample
                count = 1
            else:
                count = 0
            
            # Process remaining samples
            for sample in sample_iter:
                cleaned_sample = self._clean_sample(sample)
                if cleaned_sample is not None:
                    yield cleaned_sample
                    count += 1
                    
                    if num_samples and count >= num_samples:
                        self.logger.info(f"Reached sample limit: {num_samples}")
                        break
                    
                    if count % 1000 == 0:
                        self.logger.debug(f"Processed {count} samples")
            
            self.logger.info(f"Finished processing {count} samples from {dataset_name}")
            
        except Exception as e:
            error_msg = f"Failed to load dataset {dataset_name}: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg) from e
    
    def load_dataset_batch(
        self,
        dataset_name: Optional[str] = None,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load dataset in batch mode (loads all data into memory).
        Use only for smaller datasets or when full dataset access is needed.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            num_samples: Maximum number of samples to load
            
        Returns:
            List of cleaned samples
            
        Raises:
            DataLoadError: If dataset loading fails
        """
        try:
            # Convert streaming iterator to list
            samples = list(self.load_dataset_streaming(
                dataset_name=dataset_name,
                split=split,
                num_samples=num_samples
            ))
            
            self.logger.info(f"Loaded {len(samples)} samples in batch mode")
            return samples
            
        except Exception as e:
            error_msg = f"Failed to load dataset in batch mode: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg) from e
    
    def _validate_dataset_structure(self, sample: Dict[str, Any]) -> None:
        """
        Validate that a dataset sample has the required structure.
        
        Args:
            sample: Sample from the dataset
            
        Raises:
            DataLoadError: If sample structure is invalid
        """
        missing_columns = []
        for col in self.required_columns:
            if col not in sample:
                missing_columns.append(col)
        
        if missing_columns:
            raise DataLoadError(
                f"Dataset missing required columns: {missing_columns}. "
                f"Available columns: {list(sample.keys())}"
            )
        
        # Check for reasonable content
        report_text = sample.get(self.report_column, "")
        summary_text = sample.get(self.summary_column, "")
        
        if not isinstance(report_text, str) or not isinstance(summary_text, str):
            raise DataLoadError("Report and summary must be strings")
        
        self.logger.debug("Dataset structure validation passed")
    
    def _clean_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Clean and validate a single sample.
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Cleaned sample or None if invalid
        """
        try:
            # Extract required fields
            report = sample.get(self.report_column, "")
            summary = sample.get(self.summary_column, "")
            
            # Basic validation
            if not report or not summary:
                return None
            
            if not isinstance(report, str) or not isinstance(summary, str):
                return None
            
            # Clean text (basic cleaning)
            report = report.strip()
            summary = summary.strip()
            
            # Length validation - be reasonable but not too strict
            if len(report) < 50 or len(summary) < 10:
                return None
            
            # Prevent extremely long texts that might cause memory issues
            if len(report) > 50000 or len(summary) > 5000:
                return None
            
            # Create cleaned sample
            cleaned_sample = {
                self.report_column: report,
                self.summary_column: summary,
                "sample_id": sample.get("id", hash(report + summary) % 1000000)
            }
            
            return cleaned_sample
            
        except Exception as e:
            self.logger.debug(f"Error cleaning sample: {e}")
            return None
    
    def get_dataset_info(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the dataset without loading it fully.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_name is None:
            dataset_name = self.config.dataset_name
        
        try:
            info = {
                "dataset_name": dataset_name,
                "configs": ["default"],
                "splits": {"default": ["train", "test", "validation"]},
                "sample_structure": None,
                "has_required_columns": False
            }
            
            # Try to get a sample to understand structure
            try:
                sample_dataset = load_dataset(dataset_name, split="train", streaming=True)
                sample = next(iter(sample_dataset))
                info["sample_structure"] = {
                    "columns": list(sample.keys()),
                    "column_types": {k: type(v).__name__ for k, v in sample.items()}
                }
                
                # Check if our required columns exist
                info["has_required_columns"] = all(
                    col in sample for col in self.required_columns
                )
                
            except Exception as e:
                self.logger.warning(f"Could not get sample structure: {e}")
                info["sample_structure"] = None
                info["has_required_columns"] = False
            
            return info
            
        except Exception as e:
            error_msg = f"Failed to get dataset info for {dataset_name}: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg) from e
    
    def estimate_dataset_size(
        self,
        dataset_name: Optional[str] = None,
        split: str = "test",
        sample_count: int = 100
    ) -> Dict[str, Any]:
        """
        Estimate dataset size and processing requirements.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to analyze
            sample_count: Number of samples to analyze for estimation
            
        Returns:
            Dictionary with size estimates
        """
        if dataset_name is None:
            dataset_name = self.config.dataset_name
        
        try:
            self.logger.info(f"Estimating size for {dataset_name} ({split})")
            
            # Sample items to estimate
            samples = list(self.load_dataset_streaming(
                dataset_name=dataset_name,
                split=split,
                num_samples=sample_count
            ))
            
            if not samples:
                return {"error": "No valid samples found"}
            
            # Calculate statistics
            report_lengths = [len(s[self.report_column]) for s in samples]
            summary_lengths = [len(s[self.summary_column]) for s in samples]
            
            estimates = {
                "sample_count": len(samples),
                "avg_report_length": sum(report_lengths) / len(report_lengths),
                "avg_summary_length": sum(summary_lengths) / len(summary_lengths),
                "max_report_length": max(report_lengths),
                "max_summary_length": max(summary_lengths),
                "min_report_length": min(report_lengths),
                "min_summary_length": min(summary_lengths)
            }
            
            # Estimate memory requirements (rough)
            avg_chars_per_sample = estimates["avg_report_length"] + estimates["avg_summary_length"]
            bytes_per_sample = avg_chars_per_sample * 4  # Rough estimate for UTF-8
            
            estimates["estimated_bytes_per_sample"] = bytes_per_sample
            estimates["estimated_mb_per_1k_samples"] = (bytes_per_sample * 1000) / (1024 * 1024)
            
            self.logger.info(f"Estimated {bytes_per_sample:.0f} bytes per sample")
            
            return estimates
            
        except Exception as e:
            error_msg = f"Failed to estimate dataset size: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg) from e


class DataValidator:
    """
    Simple validator for dataset quality.
    """
    
    def __init__(self, config: Config):
        """Initialize the validator."""
        self.config = config
        self.logger = get_logger("data_validator")
    
    def validate_samples(
        self,
        samples: List[Dict[str, Any]],
        max_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Validate a collection of samples and return quality metrics.
        
        Args:
            samples: List of data samples to validate
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary with validation results
        """
        try:
            if len(samples) > max_samples:
                import random
                samples = random.sample(samples, max_samples)
            
            results = {
                "total_samples": len(samples),
                "valid_samples": 0,
                "issues": {
                    "empty_reports": 0,
                    "empty_summaries": 0,
                    "too_short_reports": 0,
                    "too_short_summaries": 0,
                    "too_long_reports": 0,
                    "too_long_summaries": 0,
                    "encoding_errors": 0
                },
                "statistics": {
                    "report_lengths": [],
                    "summary_lengths": [],
                    "report_word_counts": [],
                    "summary_word_counts": []
                }
            }
            
            for sample in samples:
                try:
                    report = sample.get("report", "")
                    summary = sample.get("summary", "")
                    
                    # Check for issues
                    if not report:
                        results["issues"]["empty_reports"] += 1
                        continue
                    
                    if not summary:
                        results["issues"]["empty_summaries"] += 1
                        continue
                    
                    if len(report) < 100:
                        results["issues"]["too_short_reports"] += 1
                    
                    if len(summary) < 20:
                        results["issues"]["too_short_summaries"] += 1
                    
                    if len(report) > 20000:
                        results["issues"]["too_long_reports"] += 1
                    
                    if len(summary) > 2000:
                        results["issues"]["too_long_summaries"] += 1
                    
                    # Collect statistics
                    results["statistics"]["report_lengths"].append(len(report))
                    results["statistics"]["summary_lengths"].append(len(summary))
                    results["statistics"]["report_word_counts"].append(len(report.split()))
                    results["statistics"]["summary_word_counts"].append(len(summary.split()))
                    
                    results["valid_samples"] += 1
                    
                except UnicodeError:
                    results["issues"]["encoding_errors"] += 1
                except Exception as e:
                    self.logger.debug(f"Error validating sample: {e}")
            
            # Calculate summary statistics
            if results["statistics"]["report_lengths"]:
                import statistics
                for stat_type in ["report_lengths", "summary_lengths", "report_word_counts", "summary_word_counts"]:
                    values = results["statistics"][stat_type]
                    if values:
                        results["statistics"][f"{stat_type}_mean"] = statistics.mean(values)
                        results["statistics"][f"{stat_type}_median"] = statistics.median(values)
                        results["statistics"][f"{stat_type}_std"] = statistics.stdev(values) if len(values) > 1 else 0
            
            results["quality_score"] = results["valid_samples"] / len(samples) if samples else 0
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to validate samples: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg) from e