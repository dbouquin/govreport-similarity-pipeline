"""
Core distillation logic using Model2Vec.
Handles the conversion of transformer models to static embeddings.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model2vec.distill import distill
from model2vec import StaticModel

from ..utils.logging import get_logger, TimedOperation, DistillationError
from ..utils.config import Config


class ModelDistiller:
    """
    Handles distillation of transformer models to static embeddings using Model2Vec.
    
    Implements careful handling of Model2Vec parameters to avoid the "unexpected behaviors"
    mentioned in the assignment instructions, using only verified API parameters.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the distiller.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger("distiller")
        
        # Model2Vec distillation defaults - ONLY using verified parameters from the API
        self.distill_defaults = {
            "pca_dims": config.default_pca_dims,
            "device": config.device,
            "apply_zipf": True,  # SIF/Zipf weighting - key performance parameter
            "use_subword": True,  # Use subword tokenization by default
            "quantize_to": "float16",  # Default quantization
        }
    
    def distill_model(
        self,
        model_name: str,
        output_path: Optional[Path] = None,
        pca_dims: Optional[int] = None,
        custom_vocab: Optional[list] = None,
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
                # Prepare distillation parameters
                distill_params = self.distill_defaults.copy()
                if pca_dims is not None:
                    distill_params["pca_dims"] = pca_dims
                
                # Override with any custom parameters (but filter to known safe ones)
                safe_kwargs = self._filter_distill_kwargs(distill_kwargs)
                distill_params.update(safe_kwargs)
                
                self.logger.info(f"Starting distillation of {model_name}")
                self.logger.debug(f"Distillation parameters: {distill_params}")
                
                # Validate model exists
                self._validate_model_availability(model_name)
                
                # Perform distillation
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
            
            # Load the model
            model = StaticModel.from_pretrained(str(model_path))
            
            self.logger.info(f"Successfully loaded model from {model_path}")
            self._log_model_info(model, str(model_path))
            
            return model
            
        except Exception as e:
            error_msg = f"Failed to load distilled model from {model_path}: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e
    
    def encode_with_safety(
        self,
        model: StaticModel,
        texts: Union[str, list],
        batch_size: Optional[int] = None,
        **encode_kwargs
    ) -> torch.Tensor:
        """
        Safely encode texts using the distilled model with explicit parameter handling.
        
        This method addresses the "unexpected behaviors" warning by being explicit
        about encoding parameters and only using verified Model2Vec parameters.
        
        Args:
            model: The distilled model to use for encoding
            texts: Text(s) to encode
            batch_size: Batch size for processing
            **encode_kwargs: Additional encoding parameters (filtered for safety)
            
        Returns:
            Tensor of embeddings (always returns PyTorch tensor)
        """
        try:
            # Ensure we have a list of texts
            if isinstance(texts, str):
                texts = [texts]
            
            # Prepare encoding parameters - be explicit and conservative
            encode_params = {}
            
            # Handle batch_size - the main parameter we know works
            if batch_size is not None:
                encode_params["batch_size"] = batch_size
            elif hasattr(self.config, 'batch_size'):
                encode_params["batch_size"] = self.config.batch_size
            
            # For additional safety, only allow batch_size parameter
            # This is the most conservative approach based on the assignment warning
            self.logger.debug(f"Encoding {len(texts)} texts with parameters: {encode_params}")
            
            # Use Model2Vec's encode method - let it handle batching internally
            embeddings = model.encode(texts, **encode_params)
            
            # Ensure we return a PyTorch tensor with proper type handling
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings)
            elif not isinstance(embeddings, torch.Tensor):
                # Handle any other type by converting via numpy
                embeddings = torch.from_numpy(np.array(embeddings))
            
            # Ensure proper dtype
            if embeddings.dtype != torch.float32:
                embeddings = embeddings.float()
                
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to encode texts: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e
    
    def _filter_distill_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter distillation kwargs to only include known safe parameters.
        
        Args:
            kwargs: Raw keyword arguments
            
        Returns:
            Filtered dictionary with only safe parameters
        """
        # Only allow verified Model2Vec distill() parameters
        allowed_distill_params = [
            "pca_dims",
            "device", 
            "apply_zipf",
            "use_subword",
            "quantize_to",
            "vocabulary"
        ]
        
        safe_kwargs = {}
        for key, value in kwargs.items():
            if key in allowed_distill_params:
                safe_kwargs[key] = value
            else:
                self.logger.warning(f"Ignoring unrecognized distill parameter: {key}")
        
        return safe_kwargs
    
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
            
            # Try to load tokenizer first (lighter operation)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.debug(f"Tokenizer loaded successfully for {model_name}")
            
            # Load config to validate model exists
            config = AutoConfig.from_pretrained(model_name)
            self.logger.debug(f"Model config validated for {model_name}")
            
        except Exception as e:
            error_msg = f"Model {model_name} is not available or accessible: {str(e)}"
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
            # Get embedding matrix info
            if hasattr(model, 'embedding'):
                vocab_size, embed_dim = model.embedding.shape
                self.logger.info(f"Model info for {source_name}:")
                self.logger.info(f"  Vocabulary size: {vocab_size:,}")
                self.logger.info(f"  Embedding dimension: {embed_dim}")
                
                # Calculate approximate model size
                param_count = vocab_size * embed_dim
                size_mb = (param_count * 4) / (1024 * 1024)  # Assuming float32
                self.logger.info(f"  Approximate size: {size_mb:.1f} MB")
                self.logger.info(f"  Parameters: ~{param_count:,}")
            
            # Log tokenizer info if available
            if hasattr(model, 'tokenizer'):
                self.logger.debug(f"  Tokenizer type: {type(model.tokenizer).__name__}")
            
        except Exception as e:
            self.logger.warning(f"Could not extract model info: {e}")
    
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
            if hasattr(model, 'embedding'):
                vocab_size, embed_dim = model.embedding.shape
                metrics['vocabulary_size'] = vocab_size
                metrics['embedding_dimension'] = embed_dim
                metrics['parameter_count'] = vocab_size * embed_dim
                metrics['size_mb'] = (vocab_size * embed_dim * 4) / (1024 * 1024)
            
            # Add device info
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'device'):
                metrics['device'] = str(model.embedding.device)
            
            # Add tokenizer info
            if hasattr(model, 'tokenizer'):
                metrics['tokenizer_type'] = type(model.tokenizer).__name__
            
        except Exception as e:
            self.logger.warning(f"Could not extract all model metrics: {e}")
        
        return metrics
    
    def create_vocabulary_from_dataset(
        self,
        dataset_name: str,
        text_column: str = "report",
        max_vocab_size: int = 10000,
        min_frequency: int = 5
    ) -> list:
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
            
            # Load dataset
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            
            # Tokenize and count
            word_counts = Counter()
            processed_samples = 0
            
            for sample in dataset:
                if text_column in sample and sample[text_column]:
                    # Simple tokenization (can be improved with proper tokenizer)
                    words = re.findall(r'\b\w+\b', sample[text_column].lower())
                    word_counts.update(words)
                    
                    processed_samples += 1
                    if processed_samples % 1000 == 0:
                        self.logger.debug(f"Processed {processed_samples} samples")
            
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
    
    def compare_models(
        self,
        original_model_name: str,
        distilled_model_path: Path,
        test_texts: list,
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Compare original and distilled models on test texts.
        
        Args:
            original_model_name: Original transformer model name
            distilled_model_path: Path to distilled model
            test_texts: List of texts for comparison
            similarity_threshold: Minimum similarity threshold for "good" distillation
            
        Returns:
            Comparison metrics
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            self.logger.info("Comparing original and distilled models")
            
            # Load models
            original_model = AutoModel.from_pretrained(original_model_name)
            original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
            distilled_model = self.load_distilled_model(distilled_model_path)
            
            # Get embeddings from both models
            original_embeddings = []
            distilled_embeddings = []
            
            for text in test_texts:
                # Original model (simplified - would need proper pooling in practice)
                inputs = original_tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = original_model(**inputs)
                    # Use mean pooling for simplicity
                    orig_emb = outputs.last_hidden_state.mean(dim=1).squeeze()
                    original_embeddings.append(orig_emb.numpy())
                
                # Distilled model
                dist_emb = self.encode_with_safety(distilled_model, [text])
                distilled_embeddings.append(dist_emb.squeeze().numpy())
            
            # Calculate similarities
            similarities = []
            for orig, dist in zip(original_embeddings, distilled_embeddings):
                sim = cosine_similarity([orig], [dist])[0][0]
                similarities.append(sim)
            
            # Compile metrics
            similarities = np.array(similarities)
            metrics = {
                "mean_similarity": float(similarities.mean()),
                "min_similarity": float(similarities.min()),
                "max_similarity": float(similarities.max()),
                "std_similarity": float(similarities.std()),
                "above_threshold": float((similarities >= similarity_threshold).mean()),
                "test_samples": len(test_texts)
            }
            
            self.logger.info(f"Model comparison completed: mean similarity = {metrics['mean_similarity']:.3f}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to compare models: {str(e)}"
            self.logger.error(error_msg)
            raise DistillationError(error_msg) from e