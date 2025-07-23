"""
Logging configuration for the pipeline.
Provides structured logging with rich console output and file logging.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

log_file = os.getenv("GOVREPORT_LOGFILE", "govreport.log")

console = Console()


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    logger_name: str = "govreport_pipeline"
) -> logging.Logger:
    """
    Setup logging configuration with rich console handler.
    
    Args:
        verbose: Enable debug level logging
        log_file: Optional file to write logs to
        logger_name: Name of the logger
    
    Returns:
        Configured logger instance
    """
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    
    # Get logger for this package
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"govreport_pipeline.{name}")


class ProgressTracker:
    """
    Progress tracking utility for long-running operations.
    Provides both rich progress bars and logging integration.
    """
    
    def __init__(self, description: str, total: Optional[int] = None):
        """
        Initialize progress tracker.
        
        Args:
            description: Description of the operation
            total: Total number of items to process (if known)
        """
        self.description = description
        self.total = total
        self.logger = get_logger("progress")
        self._progress = None
        self._task_id = None
    
    def __enter__(self):
        """Enter context manager."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn() if self.total else "",
            MofNCompleteColumn() if self.total else "",
            console=console
        )
        self._progress.__enter__()
        
        self._task_id = self._progress.add_task(
            self.description, 
            total=self.total
        )
        
        self.logger.info(f"Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            self.logger.info(f"Completed: {self.description}")
        else:
            self.logger.error(f"Failed: {self.description} - {exc_val}")
        
        self._progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, advance: int = 1, **kwargs):
        """Update progress."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=advance, **kwargs)
    
    def set_description(self, description: str):
        """Update the progress description."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=description)


class TimedOperation:
    """Context manager for timing operations with logging."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timed operation.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Logger to use (defaults to module logger)
        """
        self.operation_name = operation_name
        self.logger = logger or get_logger("timing")
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results."""
        import time
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(
                    f"Completed {self.operation_name} in {elapsed:.2f} seconds"
                )
            else:
                self.logger.error(
                    f"Failed {self.operation_name} after {elapsed:.2f} seconds: {exc_val}"
                )


def log_memory_usage(logger: Optional[logging.Logger] = None) -> None:
    """Log current memory usage."""
    logger = logger or get_logger("memory")
    
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"Memory usage: {memory_mb:.1f} MB")
    except ImportError:
        logger.debug("psutil not available for memory monitoring")


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """Log system information."""
    logger = logger or get_logger("system")
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        logger.warning("PyTorch not available")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.warning("Transformers not available")
    
    try:
        import model2vec
        logger.info(f"Model2Vec version: {model2vec.__version__}")
    except ImportError:
        logger.warning("Model2Vec not available")


# Custom exception classes for better error handling
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DistillationError(PipelineError):
    """Error during model distillation."""
    pass


class AnalysisError(PipelineError):
    """Error during similarity analysis."""
    pass


class DataLoadError(PipelineError):
    """Error loading or processing data."""
    pass


class ModelLoadError(PipelineError):
    """Error loading models."""
    pass