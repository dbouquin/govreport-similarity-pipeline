#!/usr/bin/env python3
"""
Government Report Summarization Quality Assessment Pipeline
Harvard IDI Principal Engineer Technical Assessment

(Daina's first time building a CLI interface so this took closer to 5 hours)
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Simple path setup
sys.path.insert(0, str(Path(__file__).parent / "src"))

console = Console()

# Import our validation and config
try:
    from src.utils.validation import validate_environment, ValidationError
    from src.utils.config import Config
except ImportError as e:
    console.print(f"[red]Import error: {e}[/red]")
    console.print("Ensure you're in the project root directory")
    sys.exit(1)

# Import commands only when needed to avoid import issues
def get_commands():
    """Lazy import commands to avoid early import failures."""
    try:
        from src.commands.distill import distill_command
        from src.commands.analyze import analyze_command  
        from src.commands.report import report_command
        return distill_command, analyze_command, report_command
    except ImportError as e:
        console.print(f"[red]Command import error: {e}[/red]")
        console.print("Run 'python main.py validate' to check your environment")
        sys.exit(1)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps", "auto"]), help="Device to use")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, device: str) -> None:
    """
    Government Report Summarization Quality Assessment Pipeline.
    
    A CLI for evaluating semantic similarity between 
    AI-generated summaries and original government reports.
    
    Commands:
        validate - Check environment and dependencies
        info     - Show pipeline information  
        distill  - Create distilled models using Model2Vec
        analyze  - Calculate similarity between reports and summaries  
        report   - Generate statistics and quality metrics
    
    Examples:
        python main.py validate                    # Check setup
        python main.py distill --model BAAI/bge-m3 # Create model
        python main.py analyze --model models/my_model --num-samples 10
    """
    # Initialize configuration
    config_overrides = {}
    if verbose:
        config_overrides["verbose"] = verbose
    if device:
        config_overrides["device"] = device
    
    try:
        config = Config(**config_overrides)
        config.validate()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    
    # Store in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.option("--check-env", is_flag=True, help="Run environment checks")
def validate(check_env: bool) -> None:
    """Validate pipeline setup and environment."""
    console.print("\n[bold blue]Pipeline Validation[/bold blue]")
    
    try:
        # Run environment validation
        results = validate_environment()
        
        # Display results
        if results["status"] == "passed":
            console.print("[bold green]✓ Environment validation passed![/bold green]")
        elif results["status"] == "warning":
            console.print("[bold yellow]⚠ Environment validation passed with warnings[/bold yellow]")
        else:
            console.print("[bold red]✗ Environment validation failed[/bold red]")
        
        # Show details
        console.print(f"\nPython version: {results['python_version']}")
        console.print(f"Device: {results['device']}")
        
        # Package status
        available_packages = [pkg for pkg, info in results["packages"].items() if info["available"]]
        missing_packages = [pkg for pkg, info in results["packages"].items() if not info["available"]]
        
        if available_packages:
            console.print(f"\n[green]Available packages ({len(available_packages)}):[/green]")
            for pkg in available_packages[:5]:  # Show first 5
                version = results["packages"][pkg]["version"]
                console.print(f"  ✓ {pkg} ({version})")
            if len(available_packages) > 5:
                console.print(f"  ... and {len(available_packages) - 5} more")
        
        if missing_packages:
            console.print(f"\n[red]Missing packages ({len(missing_packages)}):[/red]")
            for pkg in missing_packages:
                console.print(f"  ✗ {pkg}")
            console.print("\nInstall missing packages with:")
            console.print(f"pip install {' '.join(missing_packages)}")
        
        # Show errors/warnings
        if results["errors"]:
            console.print(f"\n[red]Errors:[/red]")
            for error in results["errors"]:
                console.print(f"  • {error}")
        
        if results["warnings"]:
            console.print(f"\n[yellow]Warnings:[/yellow]")
            for warning in results["warnings"]:
                console.print(f"  • {warning}")
        
        # Next steps
        if results["status"] == "passed":
            console.print(f"\n[bold green]Ready to use![/bold green]")
            console.print("Next steps:")
            console.print("  1. python main.py distill --model BAAI/bge-m3")
            console.print("  2. python main.py analyze --model models/BAAI_bge-m3_distilled --num-samples 10") 
            console.print("  3. python main.py report --input results/analysis_*.csv")
        
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)


@cli.command() 
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display pipeline information and status."""
    config = ctx.obj["config"]
    
    console.print("\n[bold blue]Government Report Similarity Pipeline[/bold blue]")
    console.print("Harvard IDI Principal Engineer Assessment")
    
    # Configuration table
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Models Directory", str(config.models_dir))
    table.add_row("Results Directory", str(config.results_dir))
    table.add_row("Default Model", config.default_model)
    table.add_row("Dataset", config.dataset_name)
    table.add_row("Device", config.device)
    table.add_row("Batch Size", str(config.batch_size))
    table.add_row("Max Workers", str(config.max_workers))
    
    console.print(table)
    
    # Directory status
    console.print(f"\n[bold]Directory Status:[/bold]")
    dirs = {
        "Models": config.models_dir,
        "Results": config.results_dir,
        "Cache": config.cache_dir
    }
    
    for name, path in dirs.items():
        status = "✓" if path.exists() else "✗"
        console.print(f"  {status} {name}: {path}")
    
    # Quick examples
    examples = Panel.fit(
        "[bold]Quick Start Examples:[/bold]\n\n"
        "1. Check environment:\n"
        "   [code]python main.py validate[/code]\n\n"
        "2. Create a model:\n"
        "   [code]python main.py distill --model BAAI/bge-m3[/code]\n\n"
        "3. Test analysis:\n"
        "   [code]python main.py analyze --model models/my_model --num-samples 10[/code]",
        title="Usage"
    )
    console.print(examples)


# Add commands dynamically to avoid import issues
def add_commands():
    """Add commands after validating imports."""
    try:
        distill_cmd, analyze_cmd, report_cmd = get_commands()
        cli.add_command(distill_cmd, name="distill")
        cli.add_command(analyze_cmd, name="analyze")
        cli.add_command(report_cmd, name="report")
    except SystemExit:
        pass  # Commands couldn't be imported, but validate/info still work


if __name__ == "__main__":
    try:
        # Add commands if possible
        add_commands()
        
        # Run CLI
        cli()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)