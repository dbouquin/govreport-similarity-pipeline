"""
CLI command for similarity analysis between reports and summaries.
Updated to work with simplified Model2Vec batching approach.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ..core.analyzer import SimilarityAnalyzer
from ..core.data_loader import DatasetLoader
from ..utils.logging import get_logger
from ..utils.config import Config

console = Console()
logger = get_logger("analyze_command")


@click.command()
@click.option(
    "--model", "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to distilled model directory"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output path for results (default: auto-generated)"
)
@click.option(
    "--dataset",
    default="ccdv/govreport-summarization",
    help="Dataset to analyze (default: ccdv/govreport-summarization)"
)
@click.option(
    "--split",
    default="test",
    help="Dataset split to use (default: test)"
)
@click.option(
    "--num-samples", "-n",
    type=int,
    help="Maximum number of samples to analyze (default: all)"
)
@click.pass_context
def analyze_command(
    ctx: click.Context,
    model: Path,
    output: Optional[Path],
    dataset: str,
    split: str,
    num_samples: Optional[int]
) -> None:
    """
    Analyze semantic similarity between reports and summaries.
    
    This command uses a distilled embedding model to calculate cosine similarity
    between government reports and their AI-generated summaries, providing insights
    into summarization quality including distance distribution in tranches.
    
    The updated implementation trusts Model2Vec's internal batching for optimal
    performance and memory efficiency.
    """
    config: Config = ctx.obj["config"]
    
    # Determine output path with timestamp
    if output is None:
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.name
        output = config.results_dir / f"analysis_{model_name}_{timestamp}"
    
    console.print(f"\n[bold blue]Analyzing Semantic Similarity[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Dataset: {dataset} ({split})")
    console.print(f"Output: {output}")
    console.print(f"Max samples: {num_samples or 'all'}")
    
    try:
        # Initialize components
        analyzer = SimilarityAnalyzer(config)
        data_loader = DatasetLoader(config)
        
        # Validate model exists and is loadable
        console.print("\n[bold]Validating model...[/bold]")
        try:
            test_model = analyzer.distiller.load_distilled_model(model)
            model_metrics = analyzer.distiller.get_model_metrics(test_model)
            
            console.print("✓ Model loaded successfully")
            console.print(f"  Vocabulary size: {model_metrics.get('vocabulary_size', 'Unknown'):,}")
            console.print(f"  Embedding dimension: {model_metrics.get('embedding_dimension', 'Unknown')}")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            raise click.ClickException(f"Model validation failed: {e}")
        
        # Get dataset info
        console.print("\n[bold]Dataset Information:[/bold]")
        try:
            dataset_info = data_loader.get_dataset_info(dataset)
            console.print(f"  Dataset: {dataset_info['dataset_name']}")
            console.print(f"  Required columns present: {dataset_info.get('has_required_columns', 'Unknown')}")
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not get dataset info: {e}[/yellow]")
        
        # Perform analysis
        console.print(f"\n[bold green]Starting similarity analysis...[/bold green]")
        console.print("[dim]Using Model2Vec's optimized internal batching[/dim]")
        
        results = analyzer.analyze_dataset(
            model_path=model,
            output_path=output,
            num_samples=num_samples,
            dataset_name=dataset
        )
        
        # Display results summary
        display_analysis_results(results, console)
        
        # Display tranches analysis
        if results.get("distance_tranches") and "error" not in results["distance_tranches"]:
            display_tranches_analysis(results["distance_tranches"], console)
        
        # Show analysis summary
        show_analysis_summary(results, console)
        
        # Generate summary report
        console.print(f"\n[bold green]✓ Analysis completed successfully![/bold green]")
        console.print(f"Results saved to: {output}")
        console.print(f"Samples processed: {results['processing_stats']['samples_processed']}")
        console.print(f"Samples failed: {results['processing_stats']['samples_failed']}")
        
        if results.get("similarity_scores"):
            mean_similarity = sum(results["similarity_scores"]) / len(results["similarity_scores"])
            console.print(f"Mean similarity: {mean_similarity:.3f}")
        
        # Provide next steps
        console.print(f"\n[dim]Next: [code]python main.py report --input {output}.csv[/code][/dim]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Analysis failed: {e}[/red]")
        logger.error(f"Analysis failed", exc_info=True)
        raise click.ClickException(f"Analysis failed: {e}")


def display_analysis_results(results: dict, console: Console) -> None:
    """
    Display analysis results in a formatted table.
    
    Args:
        results: Analysis results dictionary
        console: Rich console for output
    """
    try:
        console.print("\n[bold]Analysis Results:[/bold]")
        
        # Create results table
        table = Table(title="Similarity Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Processing statistics
        stats = results["processing_stats"]
        table.add_row("Samples Processed", f"{stats['samples_processed']:,}")
        table.add_row("Samples Failed", f"{stats['samples_failed']:,}")
        
        if stats["samples_processed"] > 0:
            success_rate = stats["samples_processed"] / (stats["samples_processed"] + stats["samples_failed"])
            table.add_row("Success Rate", f"{success_rate:.1%}")
        
        # Similarity statistics
        if results.get("similarity_scores"):
            scores = results["similarity_scores"]
            table.add_section()
            table.add_row("Mean Similarity", f"{sum(scores) / len(scores):.3f}")
            table.add_row("Median Similarity", f"{sorted(scores)[len(scores)//2]:.3f}")
            table.add_row("Min Similarity", f"{min(scores):.3f}")
            table.add_row("Max Similarity", f"{max(scores):.3f}")
            
            # Quality categories
            high_quality = sum(1 for s in scores if s > 0.8)
            medium_quality = sum(1 for s in scores if 0.5 < s <= 0.8)
            low_quality = sum(1 for s in scores if s <= 0.5)
            
            table.add_section()
            table.add_row("High Similarity (>0.8)", f"{high_quality} ({high_quality/len(scores):.1%})")
            table.add_row("Medium Similarity (0.5-0.8)", f"{medium_quality} ({medium_quality/len(scores):.1%})")
            table.add_row("Low Similarity (≤0.5)", f"{low_quality} ({low_quality/len(scores):.1%})")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not display results summary: {e}[/yellow]")


def display_tranches_analysis(tranches_data: dict, console: Console) -> None:
    """
    Display distance tranches analysis in a formatted table.
    
    Args:
        tranches_data: Tranches analysis results
        console: Rich console for output
    """
    try:
        console.print("\n[bold]Distance Distribution in Tranches:[/bold]")
        
        # Create tranches table
        table = Table(title="Distance Distribution Tranches")
        table.add_column("Tranche #", style="cyan", justify="center")
        table.add_column("Distance Range", style="yellow")
        table.add_column("Similarity Range", style="green")
        table.add_column("Count", style="blue", justify="right")
        table.add_column("Percentage", style="magenta", justify="right")
        table.add_column("Quality Level", style="white")
        
        # Add tranche data
        for tranche in tranches_data["tranches"]:
            # Color code based on similarity level
            if tranche["percentage"] > 0:
                if tranche["similarity_range"]["lower_bound"] > 0.8:
                    style = "green"
                elif tranche["similarity_range"]["lower_bound"] > 0.6:
                    style = "blue"  
                elif tranche["similarity_range"]["lower_bound"] > 0.4:
                    style = "yellow"
                else:
                    style = "red"
            else:
                style = "dim"
            
            table.add_row(
                f"[{style}]{tranche['tranche_id']}[/{style}]",
                f"[{style}]{tranche['distance_range']['lower_bound']:.3f} - {tranche['distance_range']['upper_bound']:.3f}[/{style}]",
                f"[{style}]{tranche['similarity_range']['lower_bound']:.3f} - {tranche['similarity_range']['upper_bound']:.3f}[/{style}]",
                f"[{style}]{tranche['count']:,}[/{style}]",
                f"[{style}]{tranche['percentage']:.1f}%[/{style}]",
                f"[{style}]{tranche['description']}[/{style}]"
            )
        
        console.print(table)
        
        # Display summary
        summary = tranches_data["summary"]
        console.print(f"\n[bold]Tranches Summary:[/bold]")
        console.print(f"  Total tranches: {tranches_data['num_tranches']}")
        console.print(f"  Total samples: {tranches_data['total_samples']:,}")
        console.print(f"  Distance range: {tranches_data['distance_range']['min_distance']:.3f} - {tranches_data['distance_range']['max_distance']:.3f}")
        console.print(f"  Most populated tranche: #{summary['most_populated_tranche']}")
        console.print(f"  Least populated tranche: #{summary['least_populated_tranche']}")
        console.print(f"  Concentration metric: {summary['concentration_metric']:.2f} (1.0 = uniform distribution)")
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not display tranches analysis: {e}[/yellow]")


def show_analysis_summary(results: dict, console: Console) -> None:
    """
    Show objective analysis summary for Harvard IDI context.
    
    Args:
        results: Analysis results
        console: Rich console for output
    """
    try:
        if not results.get("similarity_scores"):
            return
        
        scores = results["similarity_scores"]
        summary_metrics = []
        
        # Quality distribution - objective metrics only
        high_quality_count = sum(1 for s in scores if s > 0.8)
        high_quality_pct = (high_quality_count / len(scores)) * 100
        summary_metrics.append(f"High-quality samples (>0.8 similarity): {high_quality_count}/{len(scores)} ({high_quality_pct:.1f}%)")
        
        # Consistency assessment
        import numpy as np
        std_sim = np.std(scores)
        summary_metrics.append(f"Standard deviation: {std_sim:.3f} (consistency indicator)")
        
        # Tranches distribution - objective summary
        if results.get("distance_tranches") and "error" not in results["distance_tranches"]:
            tranches = results["distance_tranches"]
            top_two_percentage = sum(
                t["percentage"] for t in tranches["tranches"][:2]  # Top 2 tranches
            )
            summary_metrics.append(f"Top 2 tranches contain: {top_two_percentage:.1f}% of samples")
        
        if summary_metrics:
            console.print("\n[bold]Analysis Summary:[/bold]")
            for metric in summary_metrics:
                console.print(f"  • {metric}")
        
    except Exception as e:
        console.print(f"[yellow]Could not generate summary metrics: {e}[/yellow]")