"""
CLI command for model distillation using Model2Vec.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from ..core.distiller import ModelDistiller
from ..utils.logging import get_logger, log_system_info
from ..utils.config import Config

console = Console()
logger = get_logger("distill_command")


@click.command()
@click.option(
    "--model", "-m",
    default="BAAI/bge-m3",
    help="Source model to distill (default: BAAI/bge-m3)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output directory for distilled model (default: auto-generated)"
)
@click.option(
    "--pca-dims",
    type=int,
    default=256,
    help="PCA dimensions for embedding reduction (default: 256)"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    default="auto",
    help="Device to use for distillation (default: auto)"
)
@click.option(
    "--custom-vocab",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom vocabulary file (one token per line)"
)
@click.option(
    "--vocab-from-dataset",
    is_flag=True,
    help="Create vocabulary from the target dataset"
)
@click.option(
    "--vocab-size",
    type=int,
    default=10000,
    help="Maximum vocabulary size when creating from dataset (default: 10000)"
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing distilled model"
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate distilled model after creation"
)
@click.pass_context
def distill_command(
    ctx: click.Context,
    model: str,
    output: Optional[Path],
    pca_dims: int,
    device: str,
    custom_vocab: Optional[Path],
    vocab_from_dataset: bool,
    vocab_size: int,
    force: bool,
    validate: bool
) -> None:
    """
    Distill a transformer model into a fast static embedding model.
    
    This command uses Model2Vec to convert large transformer models like BAAI/bge-m3
    into lightweight static embedding models that are 100x+ faster for inference.
    
    Examples:
    
        # Basic distillation with default settings
        python main.py distill
        
        # Distill with custom PCA dimensions
        python main.py distill --pca-dims 512
        
        # Create vocabulary from target dataset
        python main.py distill --vocab-from-dataset --vocab-size 15000
        
        # Distill different model
        python main.py distill --model "sentence-transformers/all-mpnet-base-v2"
    """
    config: Config = ctx.obj["config"]
    
    # Override config with command-line arguments
    if device != "auto":
        config.device = device
    if pca_dims != 256:
        config.default_pca_dims = pca_dims
    
    # Determine output path
    if output is None:
        output = config.get_model_path(model)
    
    console.print(f"\n[bold blue]Distilling Model: {model}[/bold blue]")
    console.print(f"Output path: {output}")
    console.print(f"PCA dimensions: {pca_dims}")
    console.print(f"Device: {config.device}")
    
    # Check if model already exists
    if output.exists() and not force:
        console.print(f"\n[yellow]Warning: Model already exists at {output}[/yellow]")
        if not click.confirm("Do you want to overwrite it?"):
            console.print("[red]Distillation cancelled[/red]")
            return
    
    try:
        # Log system information for debugging
        log_system_info(logger)
        
        # Initialize distiller
        distiller = ModelDistiller(config)
        
        # Handle vocabulary options
        vocabulary = None
        if custom_vocab:
            console.print(f"Loading custom vocabulary from {custom_vocab}")
            vocabulary = load_vocabulary_file(custom_vocab)
            logger.info(f"Loaded {len(vocabulary)} tokens from custom vocabulary")
            
        elif vocab_from_dataset:
            console.print("Creating vocabulary from target dataset...")
            with console.status("[spinner]Creating vocabulary..."):
                vocabulary = distiller.create_vocabulary_from_dataset(
                    dataset_name=config.dataset_name,
                    text_column="report",
                    max_vocab_size=vocab_size,
                    min_frequency=2
                )
            console.print(f"Created vocabulary with {len(vocabulary)} tokens")
        
        # Perform distillation
        console.print("\n[bold green]Starting distillation...[/bold green]")
        
        distilled_model = distiller.distill_model(
            model_name=model,
            output_path=output,
            pca_dims=pca_dims,
            custom_vocab=vocabulary,
            device=config.device
        )
        
        # Get model metrics
        metrics = distiller.get_model_metrics(distilled_model)
        
        # Display results
        console.print("\n[bold green]✓ Distillation completed successfully![/bold green]")
        console.print("\n[bold]Model Information:[/bold]")
        console.print(f" Saved to: {output}")
        console.print(f" Vocabulary size: {metrics.get('vocabulary_size', 'Unknown'):,}")
        console.print(f" Embedding dimension: {metrics.get('embedding_dimension', 'Unknown')}")
        console.print(f" Approximate size: {metrics.get('size_mb', 0):.1f} MB")
        console.print(f" Parameters: ~{metrics.get('parameter_count', 0):,}")
        
        # Validate model if requested
        if validate:
            console.print("\n[bold]Validating distilled model...[/bold]")
            validate_distilled_model(distiller, distilled_model, model)
        
        # Save distillation metadata
        save_distillation_metadata(output, model, pca_dims, vocabulary, metrics)
        
        console.print(f"\n[bold green]Distilled model ready for use![/bold green]")
        console.print(f"Use it with: [code]python main.py analyze --model {output}[/code]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Distillation failed: {e}[/red]")
        logger.error(f"Distillation failed", exc_info=True)
        raise click.ClickException(f"Distillation failed: {e}")


def load_vocabulary_file(vocab_path: Path) -> list:
    """
    Load vocabulary from a text file.
    
    Args:
        vocab_path: Path to vocabulary file (one token per line)
        
    Returns:
        List of vocabulary tokens
    """
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(vocab)} tokens from {vocab_path}")
        return vocab
        
    except Exception as e:
        raise click.ClickException(f"Failed to load vocabulary from {vocab_path}: {e}")


def validate_distilled_model(
    distiller: ModelDistiller,
    distilled_model,
    original_model_name: str
) -> None:
    """
    Validate the distilled model by testing basic functionality.
    
    Args:
        distiller: ModelDistiller instance
        distilled_model: The distilled model to validate
        original_model_name: Original model name for comparison
    """
    try:
        # Test basic encoding
        test_texts = [
            "This is a test sentence.",
            "Government report on economic policy.",
            "Summary of financial regulations."
        ]
        
        with console.status("[spinner]Testing model encoding..."):
            embeddings = distiller.encode_with_safety(
                distilled_model, 
                test_texts,
                batch_size=len(test_texts)
            )
        
        # Validate embedding properties
        if embeddings.shape[0] != len(test_texts):
            raise ValueError(f"Expected {len(test_texts)} embeddings, got {embeddings.shape[0]}")
        
        # Check for reasonable embedding values
        if embeddings.isnan().any():
            raise ValueError("Model produced NaN embeddings")
        
        if embeddings.isinf().any():
            raise ValueError("Model produced infinite embeddings")
        
        # Calculate basic statistics
        mean_norm = embeddings.norm(dim=1).mean().item()
        
        console.print(f" Successfully encoded {len(test_texts)} test sentences")
        console.print(f" Embedding dimension: {embeddings.shape[1]}")
        console.print(f" Mean embedding norm: {mean_norm:.3f}")
        console.print(" No NaN or infinite values detected")
        
        logger.info("Model validation completed successfully")
        
    except Exception as e:
        console.print(f" ✗ Model validation failed: {e}")
        logger.error(f"Model validation failed: {e}")
        raise


def save_distillation_metadata(
    output_path: Path,
    source_model: str,
    pca_dims: int,
    vocabulary: Optional[list],
    metrics: dict
) -> None:
    """
    Save metadata about the distillation process.
    
    Args:
        output_path: Where the model was saved
        source_model: Original model name
        pca_dims: PCA dimensions used
        vocabulary: Custom vocabulary if used
        metrics: Model metrics
    """
    try:
        import json
        from datetime import datetime
        
        metadata = {
            "distillation_info": {
                "source_model": source_model,
                "pca_dimensions": pca_dims,
                "custom_vocabulary": vocabulary is not None,
                "vocabulary_size": len(vocabulary) if vocabulary else None,
                "distillation_timestamp": datetime.now().isoformat(),
                "model2vec_version": None  # Could be added if needed
            },
            "model_metrics": metrics
        }
        
        # Try to get Model2Vec version
        try:
            import model2vec
            metadata["distillation_info"]["model2vec_version"] = model2vec.__version__
        except (ImportError, AttributeError):
            pass
        
        metadata_path = output_path / "distillation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved distillation metadata to {metadata_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save distillation metadata: {e}")