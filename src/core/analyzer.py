"""
Core similarity analysis logic for report-summary pairs.
Handles embedding generation and similarity calculation with tranches analysis.
Clean implementation that trusts Model2Vec's internal batching.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from ..core.distiller import ModelDistiller
from ..core.data_loader import DatasetLoader
from ..utils.logging import get_logger, ProgressTracker, TimedOperation, AnalysisError
from ..utils.config import Config


class SimilarityAnalyzer:
    """
    Analyzes semantic similarity between government reports and their summaries.
    Uses distilled embedding models for efficient similarity computation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the similarity analyzer.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger("analyzer")
        self.distiller = ModelDistiller(config)
        self.data_loader = DatasetLoader(config)
        
        # Analysis settings
        self.similarity_metric = "cosine"
    
    def analyze_dataset(
        self,
        model_path: Path,
        output_path: Optional[Path] = None,
        num_samples: Optional[int] = None,
        dataset_name: Optional[str] = None,
        max_workers: int = 1  # Keep simple - no multiprocessing complexity
    ) -> Dict[str, Any]:
        """
        Analyze similarity for the entire dataset.
        
        Args:
            model_path: Path to distilled model
            output_path: Where to save results (optional)
            num_samples: Maximum number of samples to analyze
            dataset_name: Dataset identifier (uses config default if None)
            max_workers: Number of worker processes (ignored - single threaded)
            
        Returns:
            Analysis results dictionary
        """
        try:
            with TimedOperation("Dataset similarity analysis", self.logger):
                # Load the distilled model
                model = self.distiller.load_distilled_model(model_path)
                self.logger.info(f"Loaded model from {model_path}")
                
                # Setup data streaming
                data_stream = self.data_loader.load_dataset_streaming(
                    dataset_name=dataset_name,
                    split=self.config.dataset_split,
                    num_samples=num_samples
                )
                
                # Initialize results collection
                results = {
                    "analysis_metadata": {
                        "model_path": str(model_path),
                        "dataset_name": dataset_name or self.config.dataset_name,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "num_samples_requested": num_samples,
                        "similarity_metric": self.similarity_metric,
                        "processing_mode": "single_worker"
                    },
                    "similarity_scores": [],
                    "sample_metadata": [],
                    "processing_stats": {
                        "samples_processed": 0,
                        "samples_failed": 0,
                        "average_processing_time": 0.0
                    }
                }
                
                # Process data - let Model2Vec handle batching internally
                results = self._analyze_streaming(model, data_stream, results)
                
                # Add tranches analysis
                if results["similarity_scores"]:
                    self.logger.info("Creating distance tranches analysis")
                    results["distance_tranches"] = self.create_distance_tranches_analysis(
                        results["similarity_scores"]
                    )
                
                # Save results if output path provided
                if output_path:
                    self._save_results(results, output_path)
                
                self.logger.info(f"Analysis completed: {results['processing_stats']['samples_processed']} samples")
                
                return results
                
        except Exception as e:
            error_msg = f"Failed to analyze dataset: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def _analyze_streaming(
        self,
        model,
        data_stream: Iterator[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze data using streaming approach with efficient batching.
        
        Model2Vec handles batching internally, so collect reasonable chunks
        and let the library optimize the processing.
        
        Args:
            model: Loaded distilled model
            data_stream: Iterator of data samples
            results: Results dictionary to update
            
        Returns:
            Updated results dictionary
        """
        import time
        
        # Collect samples in chunks for efficient processing
        chunk_size = 100  # Process samples in chunks, but let Model2Vec handle batching
        current_chunk = []
        processing_times = []
        
        for sample in data_stream:
            current_chunk.append(sample)
            
            # Process when we have a full chunk
            if len(current_chunk) >= chunk_size:
                chunk_start_time = time.time()
                
                try:
                    chunk_results = self._process_chunk_efficiently(model, current_chunk)
                    self._add_chunk_results(chunk_results, results)
                    
                    chunk_time = time.time() - chunk_start_time
                    processing_times.append(chunk_time)
                    
                    # Log progress periodically
                    total_processed = results["processing_stats"]["samples_processed"]
                    if total_processed % 500 == 0 and total_processed > 0:
                        avg_time = np.mean(processing_times[-5:]) if processing_times else 0
                        self.logger.info(f"Processed {total_processed} samples (avg: {avg_time:.2f}s/chunk)")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process chunk of {len(current_chunk)} samples: {e}")
                    results["processing_stats"]["samples_failed"] += len(current_chunk)
                
                # Reset for next chunk
                current_chunk = []
        
        # Process remaining samples
        if current_chunk:
            try:
                chunk_results = self._process_chunk_efficiently(model, current_chunk)
                self._add_chunk_results(chunk_results, results)
            except Exception as e:
                self.logger.warning(f"Failed to process final chunk: {e}")
                results["processing_stats"]["samples_failed"] += len(current_chunk)
        
        # Update processing statistics
        if processing_times:
            results["processing_stats"]["average_processing_time"] = np.mean(processing_times)
        
        return results
    
    def _process_chunk_efficiently(
        self,
        model,
        chunk: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of samples efficiently using Model2Vec's internal batching.
        
        Args:
            model: Distilled model
            chunk: List of samples to process
            
        Returns:
            List of analysis results
        """
        try:
            import time
            
            # Extract texts for batch processing
            reports = [sample["report"] for sample in chunk]
            summaries = [sample["summary"] for sample in chunk]
            
            chunk_start = time.time()
            
            # Let Model2Vec handle the batching internally 
            self.logger.debug(f"Encoding {len(reports)} reports and summaries")
            
            # Generate embeddings - Model2Vec handles batching optimization
            report_embeddings = self.distiller.encode_texts(model, reports)
            summary_embeddings = self.distiller.encode_texts(model, summaries)
            
            # Calculate similarities
            similarities = self._calculate_similarities(
                report_embeddings, summary_embeddings
            )
            
            chunk_time = time.time() - chunk_start
            per_sample_time = chunk_time / len(chunk)
            
            # Create results
            results = []
            for i, (sample, similarity) in enumerate(zip(chunk, similarities)):
                results.append({
                    "sample_id": sample.get("sample_id", f"sample_{i}"),
                    "similarity_score": float(similarity),
                    "report_length": len(sample["report"]),
                    "summary_length": len(sample["summary"]),
                    "processing_time": per_sample_time
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process chunk efficiently: {e}")
            # Return error results for all samples
            return [
                {
                    "sample_id": sample.get("sample_id", f"error_{i}"),
                    "similarity_score": None,
                    "error": str(e)
                }
                for i, sample in enumerate(chunk)
            ]
    
    def _add_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> None:
        """
        Add chunk results to the main results dictionary.
        
        Args:
            chunk_results: Results from processing a chunk
            results: Main results dictionary to update
        """
        for result in chunk_results:
            if result["similarity_score"] is not None:
                results["similarity_scores"].append(result["similarity_score"])
                results["sample_metadata"].append({
                    "sample_id": result["sample_id"],
                    "report_length": result.get("report_length"),
                    "summary_length": result.get("summary_length"),
                    "processing_time": result.get("processing_time", 0)
                })
                results["processing_stats"]["samples_processed"] += 1
            else:
                results["processing_stats"]["samples_failed"] += 1
    
    def create_distance_tranches_analysis(
        self, 
        similarity_scores: List[float],
        num_tranches: int = 5
    ) -> Dict[str, Any]:
        """
        Create distance distribution analysis in tranches (equally-sized bins).
        This addresses the specific assignment requirement for "Distance distribution (in tranches)".
        
        Args:
            similarity_scores: List of similarity scores (0-1 range)
            num_tranches: Number of equally-sized tranches to create
            
        Returns:
            Dictionary containing tranche analysis
        """
        try:
            if not similarity_scores:
                return {"error": "No similarity scores provided"}
            
            scores = np.array(similarity_scores)
            
            # Convert similarity to distance (distance = 1 - similarity)
            distances = 1.0 - scores
            
            # Create equal-width tranches for distance distribution
            min_distance = float(np.min(distances))
            max_distance = float(np.max(distances))
            
            # Create tranche boundaries
            tranche_width = (max_distance - min_distance) / num_tranches if max_distance > min_distance else 0.1
            tranche_boundaries = [min_distance + i * tranche_width for i in range(num_tranches + 1)]
            tranche_boundaries[-1] = max_distance  # Ensure last boundary captures max value
            
            # Assign distances to tranches
            tranche_counts = [0] * num_tranches
            
            for distance in distances:
                # Find which tranche this distance belongs to
                if tranche_width > 0:
                    tranche_idx = min(int((distance - min_distance) / tranche_width), num_tranches - 1)
                else:
                    tranche_idx = 0  # All distances in first tranche if no variation
                tranche_counts[tranche_idx] += 1
            
            # Calculate percentages
            total_samples = len(distances)
            tranche_percentages = [(count / total_samples) * 100 for count in tranche_counts]
            
            # Create tranche labels and statistics
            tranches_analysis = {
                "num_tranches": num_tranches,
                "total_samples": total_samples,
                "distance_range": {
                    "min_distance": min_distance,
                    "max_distance": max_distance,
                    "range_width": max_distance - min_distance
                },
                "tranches": []
            }
            
            # Create detailed tranche information
            for i in range(num_tranches):
                tranche_info = {
                    "tranche_id": i + 1,
                    "distance_range": {
                        "lower_bound": tranche_boundaries[i],
                        "upper_bound": tranche_boundaries[i + 1]
                    },
                    "similarity_range": {
                        "lower_bound": 1.0 - tranche_boundaries[i + 1],  # Convert back to similarity
                        "upper_bound": 1.0 - tranche_boundaries[i]
                    },
                    "count": tranche_counts[i],
                    "percentage": tranche_percentages[i],
                    "description": self._get_tranche_description(i, num_tranches)
                }
                tranches_analysis["tranches"].append(tranche_info)
            
            # Add summary statistics
            tranches_analysis["summary"] = {
                "most_populated_tranche": max(range(num_tranches), key=lambda i: tranche_counts[i]) + 1,
                "least_populated_tranche": min(range(num_tranches), key=lambda i: tranche_counts[i]) + 1,
                "concentration_metric": max(tranche_percentages) / (100 / num_tranches) if tranche_percentages else 1.0
            }
            
            return tranches_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to create tranches analysis: {e}")
            return {"error": str(e)}

    def _get_tranche_description(self, tranche_idx: int, total_tranches: int) -> str:
        """
        Generate human-readable description for each tranche.
        
        Args:
            tranche_idx: Zero-based index of the tranche
            total_tranches: Total number of tranches
            
        Returns:
            Human-readable description of the tranche
        """
        if total_tranches == 5:
            descriptions = [
                "Very High Similarity (Excellent summaries)",
                "High Similarity (Good summaries)", 
                "Medium Similarity (Acceptable summaries)",
                "Low Similarity (Poor summaries)",
                "Very Low Similarity (Very poor summaries)"
            ]
            return descriptions[tranche_idx] if tranche_idx < len(descriptions) else f"Tranche {tranche_idx + 1}"
        else:
            # Generic descriptions for non-standard tranche counts
            position = (tranche_idx + 1) / total_tranches
            if position <= 0.2:
                return "Very High Similarity"
            elif position <= 0.4:
                return "High Similarity"
            elif position <= 0.6:
                return "Medium Similarity"
            elif position <= 0.8:
                return "Low Similarity"
            else:
                return "Very Low Similarity"
    
    def _calculate_similarities(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> np.ndarray:
        """
        Calculate pairwise similarities between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Array of similarity scores
        """
        try:
            # Convert to numpy for sklearn
            emb1_np = embeddings1.detach().cpu().numpy()
            emb2_np = embeddings2.detach().cpu().numpy()
            
            # Calculate pairwise similarities (diagonal of similarity matrix)
            similarities = np.diag(cosine_similarity(emb1_np, emb2_np))
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarities: {e}")
            raise AnalysisError(f"Similarity calculation failed: {e}") from e
    
    def _save_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results dictionary
            output_path: Where to save results
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create DataFrame for easy analysis
            df_data = []
            for i, score in enumerate(results["similarity_scores"]):
                row = {"similarity_score": score}
                
                # Add metadata if available
                if i < len(results["sample_metadata"]):
                    row.update(results["sample_metadata"][i])
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Save to CSV
            csv_path = output_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            
            # Save full results as JSON (including tranches)
            import json
            json_path = output_path.with_suffix(".json")
            
            # Prepare results for JSON (remove any numpy arrays)
            json_results = results.copy()
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=self._json_serializer)
            
            self.logger.info(f"Results saved to {csv_path} and {json_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise AnalysisError(f"Result saving failed: {e}") from e
    
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
    
    def get_similarity_statistics(
        self,
        similarity_scores: List[float]
    ) -> Dict[str, float]:
        """
        Calculate statistics for similarity scores.
        
        Args:
            similarity_scores: List of similarity scores
            
        Returns:
            Dictionary of statistics
        """
        try:
            if not similarity_scores:
                return {"error": "No similarity scores provided"}
            
            scores = np.array(similarity_scores)
            
            stats = {
                "count": len(scores),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
                "iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25))
            }
            
            # Add distribution analysis
            stats["high_similarity_ratio"] = float(np.mean(scores > 0.8))
            stats["medium_similarity_ratio"] = float(np.mean((scores > 0.5) & (scores <= 0.8)))
            stats["low_similarity_ratio"] = float(np.mean(scores <= 0.5))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {e}")
            return {"error": str(e)}