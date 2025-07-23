"""
CLI command for generating statistical reports from analysis results.
Focuses on data and metrics without subjective interpretations.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ..core.reporter import SimilarityReporter
from ..utils.logging import get_logger
from ..utils.config import Config

console = Console()
logger = get_logger("report_command")


@click.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to analysis results file (JSON or CSV)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output path for report files (default: auto-generated)"
)
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "both"]),
    default="both",
    help="Output format for report (default: both)"
)
@click.pass_context
def report_command(
    ctx: click.Context,
    input: Path,
    output: Optional[Path],
    format: str
) -> None:
    """
    Generate statistical reports from similarity analysis results.
    
    This command analyzes the results from the 'analyze' command and produces
    detailed statistical reports focused on data and metrics.
        
    """
    config: Config = ctx.obj["config"]
    
    # Determine output path with better naming convention
    if output is None:
        # Extract info from input filename to create matching report name
        input_stem = input.stem
        if input_stem.startswith("analysis_"):
            # Replace "analysis_" with "report_" to keep model and timestamp
            report_name = input_stem.replace("analysis_", "report_", 1)
        else:
            # Fallback if input doesn't follow expected naming convention
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"report_{timestamp}"
        
        output = config.results_dir / report_name
    
    console.print(f"\n[bold blue]Generating Statistical Report[/bold blue]")
    console.print(f"Input: {input}")
    console.print(f"Output: {output}")
    console.print(f"Format: {format}")
    
    try:
        # Initialize reporter
        reporter = SimilarityReporter(config)
        
        # Validate input file
        console.print("\n[bold]Validating input file...[/bold]")
        validate_input_file(input)
        console.print("✓ Input file validation passed")
        
        # Generate report
        console.print("\n[bold]Generating report...[/bold]")
        with console.status("[spinner]Processing analysis results..."):
            report = reporter.generate_report(
                input_path=input,
                output_path=output if format in ["both", "json"] else None,
                include_plots=False  # Simplified - no plots
            )
        
        # Display results in console
        display_report_summary(report, console)
        
        # Save in requested formats
        save_report_formats(report, output, format, reporter)
        
        console.print(f"\n[bold green]✓ Report generation completed successfully![/bold green]")
        console.print(f"Report files saved to: {output}")
        
        # Show file locations
        show_output_files(output, format)
        
    except Exception as e:
        console.print(f"\n[red]✗ Report generation failed: {e}[/red]")
        logger.error(f"Report generation failed", exc_info=True)
        raise click.ClickException(f"Report generation failed: {e}")


def validate_input_file(input_path: Path) -> None:
    """
    Validate that the input file contains the expected analysis results.
    
    Args:
        input_path: Path to the input file
        
    Raises:
        click.ClickException: If validation fails
    """
    try:
        if input_path.suffix.lower() not in [".csv", ".json"]:
            raise click.ClickException(f"Unsupported file format: {input_path.suffix}")
        
        # Basic content validation
        if input_path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(input_path)
            if "similarity_score" not in df.columns:
                raise click.ClickException("CSV file missing required 'similarity_score' column")
            
            if df["similarity_score"].dropna().empty:
                raise click.ClickException("No valid similarity scores found in CSV file")
        
        elif input_path.suffix.lower() == ".json":
            import json
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise click.ClickException("JSON file should contain a dictionary")
            
            if "similarity_scores" not in data and "sample_metadata" not in data:
                raise click.ClickException("JSON file missing expected analysis result structure")
        
    except Exception as e:
        if isinstance(e, click.ClickException):
            raise
        raise click.ClickException(f"Failed to validate input file: {e}")


def display_report_summary(report: dict, console: Console) -> None:
    """
    Display data-focused report summary in the console.
    
    Args:
        report: Generated report dictionary
        console: Rich console for output
    """
    try:
        console.print("\n[bold]Report Summary[/bold]")
        
        # Statistical Analysis
        stats = report.get("statistical_analysis", {})
        if "error" not in stats:
            display_statistical_summary(stats, console)
        
        # Quality Metrics
        quality_metrics = report.get("quality_metrics", {})
        if "error" not in quality_metrics:
            display_quality_metrics(quality_metrics, console)
        
        # Tranches Analysis
        tranches = report.get("tranches_analysis", {})
        if tranches and "error" not in tranches:
            display_tranches_summary(tranches, console)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not display full report summary: {e}[/yellow]")


def display_statistical_summary(stats: dict, console: Console) -> None:
    """Display statistical analysis section."""
    try:
        desc_stats = stats.get("descriptive_statistics", {})
        if desc_stats:
            table = Table(title="Descriptive Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Count", f"{desc_stats.get('count', 0):,}")
            table.add_row("Mean", f"{desc_stats.get('mean', 0):.4f}")
            table.add_row("Median", f"{desc_stats.get('median', 0):.4f}")
            table.add_row("Standard Deviation", f"{desc_stats.get('standard_deviation', 0):.4f}")
            table.add_row("Minimum", f"{desc_stats.get('minimum', 0):.4f}")
            table.add_row("Maximum", f"{desc_stats.get('maximum', 0):.4f}")
            table.add_row("Range", f"{desc_stats.get('range', 0):.4f}")
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[yellow]Could not display statistical summary: {e}[/yellow]")


def display_quality_metrics(quality_metrics: dict, console: Console) -> None:
    """Display quality metrics section."""
    try:
        threshold_analysis = quality_metrics.get("threshold_analysis", {})
        
        if threshold_analysis:
            table = Table(title="Threshold Analysis", show_header=True, header_style="bold magenta")
            table.add_column("Similarity Level", style="cyan")
            table.add_column("Threshold", justify="center")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right")
            
            for metric_name, data in threshold_analysis.items():
                level = metric_name.replace("_similarity", "").replace("_", " ").title()
                threshold = data.get("threshold", 0)
                count = data.get("count", 0)
                percentage = data.get("percentage", 0)
                
                table.add_row(
                    level,
                    f"≥ {threshold}",
                    f"{count:,}",
                    f"{percentage:.1f}%"
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[yellow]Could not display quality metrics: {e}[/yellow]")


def display_tranches_summary(tranches: dict, console: Console) -> None:
    """Display tranches analysis summary."""
    try:
        table = Table(title="Distance Tranches Summary", show_header=True, header_style="bold magenta")
        table.add_column("Tranche", style="cyan", justify="center")
        table.add_column("Distance Range", style="yellow")
        table.add_column("Count", style="blue", justify="right")
        table.add_column("Percentage", style="green", justify="right")
        
        for tranche in tranches.get("tranches", []):
            table.add_row(
                f"#{tranche['tranche_id']}",
                f"{tranche['distance_range']['lower_bound']:.3f} - {tranche['distance_range']['upper_bound']:.3f}",
                f"{tranche['count']:,}",
                f"{tranche['percentage']:.1f}%"
            )
        
        console.print(table)
        
        # Show basic summary stats
        summary = tranches.get("summary", {})
        if summary:
            console.print(f"\nMost populated tranche: #{summary.get('most_populated_tranche', 'N/A')}")
            console.print(f"Total samples: {tranches.get('total_samples', 0):,}")
        
    except Exception as e:
        console.print(f"[yellow]Could not display tranches summary: {e}[/yellow]")


def save_report_formats(
    report: dict,
    output_path: Path,
    format: str,
    reporter: SimilarityReporter
) -> None:
    """
    Save report in requested formats.
    
    Args:
        report: Report dictionary
        output_path: Base output path
        format: Requested format(s)
        reporter: SimilarityReporter instance
    """
    try:
        if format in ["json", "both"]:
            json_path = output_path.with_suffix(".json")
            import json
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=reporter._json_serializer)
        
        if format in ["csv", "both"]:
            csv_path = output_path.parent / f"{output_path.stem}_summary.csv"
            reporter._save_summary_csv(report, csv_path)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save all requested formats: {e}[/yellow]")


def show_output_files(output_path: Path, format: str) -> None:
    """Show the locations of generated files."""
    try:
        console.print("\n[bold]Generated Files:[/bold]")
        
        files_to_check = []
        
        if format in ["json", "both"]:
            files_to_check.append((output_path.with_suffix(".json"), "Statistical report (JSON)"))
        
        if format in ["csv", "both"]:
            files_to_check.append((output_path.parent / f"{output_path.stem}_summary.csv", "Summary statistics (CSV)"))
        
        for file_path, description in files_to_check:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                console.print(f"  ✓ {description}: {file_path} ({size_mb:.2f} MB)")
            else:
                console.print(f"  ✗ {description}: {file_path} (not created)")
        
    except Exception as e:
        console.print(f"[yellow]Could not show output file information: {e}[/yellow]")