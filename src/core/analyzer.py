"""
Core similarity analysis logic for report-summary pairs.
Handles embedding generation and similarity calculation with tranches analysis.
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
        self.batch_size = config.batch_size
    
    def analyze_dataset(
        self,
        model_path: Path,
        output_path: Optional[Path] = None,
        num_samples: Optional[int] = None,
        dataset_name: Optional[str] = None,
        max_workers: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze similarity for the entire dataset.
        
        Args:
            model_path: Path to distilled model
            output_path: Where to save results (optional)
            num_samples: Maximum number of samples to analyze
            dataset_name: Dataset identifier (uses config default if None)
            max_workers: Number of worker processes (single worker only - a brave attempt was made but alas, I gave up)
            
        Returns:
            Analysis results dictionary
        """
        try:
            with TimedOperation("Dataset similarity analysis", self.logger):
                # Load the distilled model
                model = self.distiller.load_distilled_model(model_path)
                self.logger.info(f"Loaded model from {model_path}")
                
                # Log if user requested multiprocessing
                if max_workers > 1:
                    self.logger.info(f"Note: Multiprocessing requested ({max_workers} workers) but using single-worker for reliability")
                
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
                        "processing_mode": "single_worker" # hardcoded for now
                    },
                    "similarity_scores": [],
                    "sample_metadata": [],
                    "processing_stats": {
                        "samples_processed": 0,
                        "samples_failed": 0,
                        "average_processing_time": 0.0
                    }
                }
                
                # Process data sequentially (single-worker)
                results = self._analyze_sequential(model, data_stream, results)
                
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
            tranche_assignments = []
            
            for distance in distances:
                # Find which tranche this distance belongs to
                if tranche_width > 0:
                    tranche_idx = min(int((distance - min_distance) / tranche_width), num_tranches - 1)
                else:
                    tranche_idx = 0  # All distances in first tranche if no variation
                tranche_counts[tranche_idx] += 1
                tranche_assignments.append(tranche_idx)
            
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
                "high_similarity_tranches": sum(1 for i in range(num_tranches) if i < num_tranches // 2),
                "low_similarity_tranches": sum(1 for i in range(num_tranches) if i >= num_tranches // 2),
                "most_populated_tranche": max(range(num_tranches), key=lambda i: tranche_counts[i]) + 1,
                "least_populated_tranche": min(range(num_tranches), key=lambda i: tranche_counts[i]) + 1,
                "concentration_metric": max(tranche_percentages) / (100 / num_tranches) if tranche_percentages else 1.0  # How concentrated vs uniform
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
    
    def analyze_sample_batch(
        self,
        model_path: Path,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze similarity for a batch of samples.
        
        Args:
            model_path: Path to distilled model
            samples: List of report-summary pairs
            
        Returns:
            List of analysis results
        """
        try:
            # Load model
            model = self.distiller.load_distilled_model(model_path)
            
            # Process samples
            results = []
            
            with ProgressTracker("Analyzing sample batch", len(samples)) as progress:
                for i, sample in enumerate(samples):
                    try:
                        result = self._analyze_single_sample(model, sample)
                        results.append(result)
                        progress.update()
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze sample {i}: {e}")
                        # Add failed result
                        results.append({
                            "sample_id": sample.get("sample_id", i),
                            "similarity_score": None,
                            "error": str(e),
                            "report_embedding": None,
                            "summary_embedding": None
                        })
                        progress.update()
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to analyze sample batch: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def _analyze_sequential(
        self,
        model,
        data_stream: Iterator[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze data sequentially (single-threaded).
        
        Args:
            model: Loaded distilled model
            data_stream: Iterator of data samples
            results: Results dictionary to update
            
        Returns:
            Updated results dictionary
        """
        import time
        
        processing_times = []
        
        # Convert stream to batched processing for efficiency
        batch_iterator = self.data_loader.batch_iterator(
            data_stream, self.batch_size
        )
        
        for batch in batch_iterator:
            batch_start_time = time.time()
            
            try:
                # Process entire batch for embedding efficiency
                batch_results = self._process_batch(model, batch)
                
                # Add results
                for result in batch_results:
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
                
                batch_time = time.time() - batch_start_time
                processing_times.append(batch_time)
                
                # Log progress periodically
                total_processed = results["processing_stats"]["samples_processed"]
                if total_processed % 100 == 0:
                    avg_time = np.mean(processing_times[-10:]) if processing_times else 0
                    self.logger.info(f"Processed {total_processed} samples (avg: {avg_time:.2f}s/batch)")
                
            except Exception as e:
                self.logger.warning(f"Failed to process batch: {e}")
                results["processing_stats"]["samples_failed"] += len(batch)
        
        # Update processing statistics
        if processing_times:
            results["processing_stats"]["average_processing_time"] = np.mean(processing_times)
        
        return results
    
    def _process_batch(
        self,
        model,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of samples efficiently.
        
        Args:
            model: Distilled model
            batch: List of samples to process
            
        Returns:
            List of analysis results
        """
        try:
            import time
            
            # Extract texts
            reports = [sample["report"] for sample in batch]
            summaries = [sample["summary"] for sample in batch]
            
            batch_start = time.time()
            
            # Generate embeddings in batch for efficiency
            report_embeddings = self.distiller.encode_with_safety(
                model, reports, batch_size=len(reports)
            )
            summary_embeddings = self.distiller.encode_with_safety(
                model, summaries, batch_size=len(summaries)
            )
            
            # Calculate similarities
            similarities = self._calculate_similarities(
                report_embeddings, summary_embeddings
            )
            
            batch_time = time.time() - batch_start
            per_sample_time = batch_time / len(batch)
            
            # Create results
            results = []
            for i, (sample, similarity) in enumerate(zip(batch, similarities)):
                results.append({
                    "sample_id": sample.get("sample_id", f"batch_{i}"),
                    "similarity_score": float(similarity),
                    "report_length": len(sample["report"]),
                    "summary_length": len(sample["summary"]),
                    "processing_time": per_sample_time,
                    "report_embedding": report_embeddings[i].detach().cpu().numpy(),
                    "summary_embedding": summary_embeddings[i].detach().cpu().numpy()
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
            # Return error results for all samples
            return [
                {
                    "sample_id": sample.get("sample_id", f"error_{i}"),
                    "similarity_score": None,
                    "error": str(e),
                    "report_embedding": None,
                    "summary_embedding": None
                }
                for i, sample in enumerate(batch)
            ]
    
    def _analyze_single_sample(
        self,
        model,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single report-summary pair.
        
        Args:
            model: Distilled model
            sample: Sample containing report and summary
            
        Returns:
            Analysis result dictionary
        """
        try:
            import time
            
            start_time = time.time()
            
            # Extract texts
            report = sample["report"]
            summary = sample["summary"]
            
            # Generate embeddings
            report_embedding = self.distiller.encode_with_safety(model, [report])
            summary_embedding = self.distiller.encode_with_safety(model, [summary])
            
            # Calculate similarity
            similarity = cosine_similarity(
                report_embedding.detach().cpu().numpy(),
                summary_embedding.detach().cpu().numpy()
            )[0][0]
            
            processing_time = time.time() - start_time
            
            return {
                "sample_id": sample.get("sample_id", "unknown"),
                "similarity_score": float(similarity),
                "report_length": len(report),
                "summary_length": len(summary),
                "processing_time": processing_time,
                "report_embedding": report_embedding.squeeze().detach().cpu().numpy(),
                "summary_embedding": summary_embedding.squeeze().detach().cpu().numpy()
            }
            
        except Exception as e:
            return {
                "sample_id": sample.get("sample_id", "unknown"),
                "similarity_score": None,
                "error": str(e),
                "report_embedding": None,
                "summary_embedding": None
            }
    
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
            
            # Calculate pairwise similarities
            if self.similarity_metric == "cosine":
                # Calculate diagonal of similarity matrix (pairwise similarities)
                similarities = np.diag(cosine_similarity(emb1_np, emb2_np))
            else:
                raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
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
            
            # Prepare results for JSON (remove numpy arrays)
            json_results = results.copy()
            json_results.pop("embeddings", None)  # Remove embeddings if present
            
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
    
    def compare_models(
        self,
        model_paths: List[Path],
        test_samples: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test samples.
        
        Args:
            model_paths: List of paths to distilled models
            test_samples: Samples to test on
            output_path: Where to save comparison results
            
        Returns:
            Comparison results
        """
        try:
            self.logger.info(f"Comparing {len(model_paths)} models on {len(test_samples)} samples")
            
            comparison_results = {
                "models": [],
                "sample_results": [],
                "summary_statistics": {}
            }
            
            # Test each model
            for i, model_path in enumerate(model_paths):
                self.logger.info(f"Testing model {i+1}/{len(model_paths)}: {model_path}")
                
                try:
                    # Analyze samples with this model
                    model_results = self.analyze_sample_batch(model_path, test_samples)
                    
                    # Extract similarity scores
                    similarities = [
                        r["similarity_score"] for r in model_results 
                        if r["similarity_score"] is not None
                    ]
                    
                    # Calculate statistics
                    stats = self.get_similarity_statistics(similarities)
                    
                    # Store model results
                    model_info = {
                        "model_path": str(model_path),
                        "model_name": model_path.name,
                        "valid_samples": len(similarities),
                        "failed_samples": len(model_results) - len(similarities),
                        "statistics": stats
                    }
                    
                    comparison_results["models"].append(model_info)
                    comparison_results["sample_results"].append(model_results)
                    
                except Exception as e:
                    self.logger.error(f"Failed to test model {model_path}: {e}")
                    comparison_results["models"].append({
                        "model_path": str(model_path),
                        "model_name": model_path.name,
                        "error": str(e)
                    })
                    comparison_results["sample_results"].append([])
            
            # Create summary comparison
            valid_models = [m for m in comparison_results["models"] if "error" not in m]
            if valid_models:
                comparison_results["summary_statistics"] = {
                    "best_mean_similarity": max(
                        m["statistics"]["mean"] for m in valid_models
                    ),
                    "model_rankings": sorted(
                        valid_models,
                        key=lambda x: x["statistics"]["mean"],
                        reverse=True
                    )
                }
            
            # Save results if requested
            if output_path:
                self._save_comparison_results(comparison_results, output_path)
            
            return comparison_results
            
        except Exception as e:
            error_msg = f"Model comparison failed: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def _save_comparison_results(
        self,
        comparison_results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save model comparison results.
        
        Args:
            comparison_results: Comparison results to save
            output_path: Where to save results
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save full results as JSON
            import json
            json_path = output_path.with_suffix(".json")
            with open(json_path, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=self._json_serializer)
            
            # Create summary CSV
            csv_path = output_path.with_suffix("_summary.csv")
            summary_data = []
            
            for model_info in comparison_results["models"]:
                if "error" not in model_info:
                    row = {
                        "model_name": model_info["model_name"],
                        "model_path": model_info["model_path"],
                        "valid_samples": model_info["valid_samples"],
                        "failed_samples": model_info["failed_samples"]
                    }
                    row.update(model_info["statistics"])
                    summary_data.append(row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Comparison results saved to {json_path} and {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save comparison results: {e}")


class EmbeddingAnalyzer:
    """
    Additional analysis tools for embeddings and similarity patterns.
    """
    
    def __init__(self, config: Config):
        """Initialize the embedding analyzer."""
        self.config = config
        self.logger = get_logger("embedding_analyzer")
    
    def analyze_embedding_quality(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze quality and properties of embeddings.
        
        Args:
            embeddings: Array of embeddings to analyze
            labels: Optional labels for embeddings
            
        Returns:
            Analysis results
        """
        try:
            results = {
                "embedding_shape": embeddings.shape,
                "dimension_statistics": {},
                "quality_metrics": {}
            }
            
            # Dimension-wise statistics
            results["dimension_statistics"] = {
                "mean_values": np.mean(embeddings, axis=0).tolist(),
                "std_values": np.std(embeddings, axis=0).tolist(),
                "dimension_variance": np.var(embeddings, axis=0).tolist()
            }
            
            # Overall quality metrics
            norms = np.linalg.norm(embeddings, axis=1)
            results["quality_metrics"] = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
                "zero_embeddings": int(np.sum(norms < 1e-6)),
                "nan_values": int(np.sum(np.isnan(embeddings))),
                "inf_values": int(np.sum(np.isinf(embeddings)))
            }
            
            # Diversity analysis
            if len(embeddings) > 1:
                pairwise_similarities = cosine_similarity(embeddings)
                # Remove diagonal (self-similarities)
                mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
                off_diagonal_sims = pairwise_similarities[mask]
                
                results["quality_metrics"]["diversity"] = {
                    "mean_pairwise_similarity": float(np.mean(off_diagonal_sims)),
                    "std_pairwise_similarity": float(np.std(off_diagonal_sims)),
                    "min_pairwise_similarity": float(np.min(off_diagonal_sims)),
                    "max_pairwise_similarity": float(np.max(off_diagonal_sims))
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze embedding quality: {e}")
            return {"error": str(e)}
    
    def find_outlier_samples(
        self,
        similarity_scores: List[float],
        sample_ids: List[str],
        threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Find outlier samples based on similarity scores.
        
        Args:
            similarity_scores: List of similarity scores
            sample_ids: List of sample identifiers
            threshold_std: Standard deviations from mean for outlier detection
            
        Returns:
            Outlier analysis results
        """
        try:
            scores = np.array(similarity_scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Find outliers
            outlier_mask = np.abs(scores - mean_score) > (threshold_std * std_score)
            outlier_indices = np.where(outlier_mask)[0]
            
            outliers = {
                "low_similarity_outliers": [],
                "high_similarity_outliers": [],
                "statistics": {
                    "total_outliers": len(outlier_indices),
                    "outlier_ratio": len(outlier_indices) / len(scores),
                    "threshold_used": threshold_std,
                    "mean_score": mean_score,
                    "std_score": std_score
                }
            }
            
            for idx in outlier_indices:
                outlier_info = {
                    "sample_id": sample_ids[idx] if idx < len(sample_ids) else f"sample_{idx}",
                    "similarity_score": similarity_scores[idx],
                    "z_score": (similarity_scores[idx] - mean_score) / std_score
                }
                
                if similarity_scores[idx] < mean_score:
                    outliers["low_similarity_outliers"].append(outlier_info)
                else:
                    outliers["high_similarity_outliers"].append(outlier_info)
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Failed to find outliers: {e}")
            return {"error": str(e)}