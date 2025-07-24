"""
Core distillation logic using Model2Vec.
Handles the conversion of transformer models to static embeddings.

Clean, professional implementation that avoids over-engineering.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

from model2vec.distill import distill
from model2vec import StaticModel

from ..utils.logging import get_logger, TimedOperation, DistillationError
from ..utils.config import Config


class ModelDistiller:
    """
    Handles distillation of transformer models to static embeddings using Model2Vec.
    """

    def __init__(self, config: Config):
        """
        Initialize the distiller.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger("distiller")

    def distill_model(
        self,
        model_name: str,
        output_path: Optional[Path] = None,
        pca_dims: Optional[int] = None,
        custom_vocab: Optional[List[str]] = None,
        **distill_kwargs
    ) -> StaticModel:
        """
        Distill a transformer model to static embeddings.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "BAAI/bge-m3")
            output_path: Where to save the distilled model
            pca_dims: Dimensionality for PCA reduction
            custom_vocab: Optional custom vocabulary for distillation
            **distill_kwargs: Additional arguments for Model2Vec distill function
            
        Returns:
            Distilled StaticModel instance
            
        Raises:
            DistillationError: If distillation fails
        """
        try:
            with TimedOperation(f"Distilling {model_name}", self.logger):
                self.logger.info(f"Starting distillation of {model_name}")
                
                # Validate model exists
                self._validate_model_availability(model_name)
                
                # Prepare distillation parameters - only use what we need
                distill_params = {}
                
                if pca_dims is not None:
                    distill_params["pca_dims"] = pca_dims
                else:
                    distill_params["pca_dims"] = self.config.default_pca_dims
                
                # Add any additional parameters passed by user
                distill_params.update(distill_kwargs)
                
                self.logger.debug(f"Distillation parameters: {distill_params}")
                
                # Perform distillation - let Model2Vec handle the complexity
                if custom_vocab:
                    self.logger.info(f"Using custom vocabulary with {len(custom_vocab)} tokens")
                    distilled_model = distill(
                        model_name=model_name,
                        vocabulary=custom_vocab,
                        **distill_params
                    )
                else:
                    self.logger.info("Using model's original vocabulary")
                    distilled_model = distill(
                        model_name=model_name,
                        **distill_params
                    )
                
                # Save model if output path specified
                if output_path:
                    output_path = Path(output_path)
                    output_path.mkdir(parents=True, exist_ok=True)
                    distilled_model.save_pretrained(str(output_path))
                    self.logger.info(f"Saved distilled model to {output_path}")
                
                # Log model information
                self._log_model_info(distilled_model, model_name)
                
                return distilled_model
                
        except Exception as e:
            error_msg = f"Failed to distill model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e

    def load_distilled_model(self, model_path: Union[str, Path]) -> StaticModel:
        """
        Load a previously distilled model.
        
        Args:
            model_path: Path to the distilled model directory
            
        Returns:
            Loaded StaticModel instance
            
        Raises:
            DistillationError: If model loading fails
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            self.logger.info(f"Loading distilled model from {model_path}")
            
            # Load the model - let Model2Vec handle the details
            model = StaticModel.from_pretrained(str(model_path))
            
            self.logger.info(f"Successfully loaded model from {model_path}")
            self._log_model_info(model, str(model_path))
            
            return model
            
        except Exception as e:
            error_msg = f"Failed to load distilled model from {model_path}: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e

    def encode_texts(
        self,
        model: StaticModel,
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Encode texts using the distilled model.
        
        Args:
            model: The distilled model to use for encoding
            texts: Text(s) to encode
            
        Returns:
            Tensor of embeddings
            
        Raises:
            DistillationError: If encoding fails
        """
        try:
            # Ensure we have a list of texts
            if isinstance(texts, str):
                texts = [texts]
            
            self.logger.debug(f"Encoding {len(texts)} texts")
            
            # Use Model2Vec's encode method 
            embeddings = model.encode(texts)
            
            # Convert to PyTorch tensor if needed, maintaining data type
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
            elif not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to encode texts: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e

    def _validate_model_availability(self, model_name: str) -> None:
        """
        Validate that the source model is available for distillation.
        
        Args:
            model_name: HuggingFace model identifier
            
        Raises:
            DistillationError: If model is not available
        """
        try:
            self.logger.debug(f"Validating model availability: {model_name}")
            
            # Check if we can load the tokenizer and config
            AutoTokenizer.from_pretrained(model_name)
            AutoConfig.from_pretrained(model_name)
            
            self.logger.debug(f"Model {model_name} is available")
            
        except Exception as e:
            error_msg = f"Model {model_name} is not available: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e

    def _log_model_info(self, model: StaticModel, source_name: str) -> None:
        """
        Log information about the distilled model.
        
        Args:
            model: The distilled model
            source_name: Source model name or path
        """
        try:
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'shape'):
                vocab_size, embed_dim = model.embedding.shape
                self.logger.info(f"Model info for {source_name}:")
                self.logger.info(f"  Vocabulary size: {vocab_size:,}")
                self.logger.info(f"  Embedding dimension: {embed_dim}")
                
                # Calculate approximate model size
                param_count = vocab_size * embed_dim
                size_mb = (param_count * 4) / (1024 * 1024)  # Assuming float32
                self.logger.info(f"  Approximate size: {size_mb:.1f} MB")
                self.logger.info(f"  Parameters: ~{param_count:,}")
            
            if hasattr(model, 'tokenizer'):
                self.logger.debug(f"  Tokenizer type: {type(model.tokenizer).__name__}")
                
        except Exception as e:
            self.logger.debug(f"Could not extract complete model info: {e}")

    def get_model_metrics(self, model: StaticModel) -> Dict[str, Any]:
        """
        Get metrics about a distilled model.
        
        Args:
            model: The distilled model
            
        Returns:
            Dictionary of model metrics
        """
        metrics = {}
        
        try:
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'shape'):
                vocab_size, embed_dim = model.embedding.shape
                metrics.update({
                    'vocabulary_size': int(vocab_size),
                    'embedding_dimension': int(embed_dim),
                    'parameter_count': int(vocab_size * embed_dim),
                    'size_mb': float((vocab_size * embed_dim * 4) / (1024 * 1024))
                })
            
            if hasattr(model, 'tokenizer'):
                metrics['tokenizer_type'] = type(model.tokenizer).__name__
                
        except Exception as e:
            self.logger.debug(f"Could not extract all model metrics: {e}")
        
        return metrics

    def create_vocabulary_from_dataset(
        self,
        dataset_name: str,
        text_column: str = "report",
        max_vocab_size: int = 10000,
        min_frequency: int = 2
    ) -> List[str]:
        """
        Create a custom vocabulary from a dataset for domain-specific distillation.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            text_column: Column containing text data
            max_vocab_size: Maximum vocabulary size
            min_frequency: Minimum token frequency for inclusion
            
        Returns:
            List of vocabulary tokens sorted by frequency
        """
        try:
            from collections import Counter
            from datasets import load_dataset
            import re
            
            self.logger.info(f"Creating vocabulary from {dataset_name}")
            
            # Load dataset in streaming mode for memory efficiency
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            
            # Tokenize and count words
            word_counts = Counter()
            processed_samples = 0
            
            for sample in dataset:
                if text_column in sample and sample[text_column]:
                    # Simple tokenization - can be enhanced with proper tokenizer
                    text = sample[text_column].lower()
                    words = re.findall(r'\b\w+\b', text)
                    word_counts.update(words)
                    
                    processed_samples += 1
                    if processed_samples % 1000 == 0:
                        self.logger.debug(f"Processed {processed_samples} samples")
                    
                    # Limit processing to avoid excessive memory usage
                    if processed_samples >= 100000:
                        break
            
            # Filter by frequency and limit size
            vocab = [
                word for word, count in word_counts.most_common(max_vocab_size)
                if count >= min_frequency
            ]
            
            self.logger.info(f"Created vocabulary with {len(vocab)} tokens from {processed_samples} samples")
            
            return vocab
            
        except Exception as e:
            error_msg = f"Failed to create vocabulary from dataset: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e