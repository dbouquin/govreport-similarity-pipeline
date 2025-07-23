"""
Simplified validation utilities for the pipeline.
Professional but focused on what actually matters.
"""

import sys
from pathlib import Path
from typing import Dict, Any


def validate_environment() -> Dict[str, Any]:
    """
    Simple environment validation - just check what we actually need.
    
    Returns:
        Environment validation results
    """
    results = {
        "status": "unknown",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {},
        "device": "cpu",
        "errors": [],
        "warnings": []
    }
    
    # Check Python version (minimum requirement)
    if sys.version_info < (3, 8):
        results["errors"].append(f"Python 3.8+ required, found {results['python_version']}")
    
    # Check essential packages (just try to import them)
    essential_packages = [
        "torch",
        "transformers", 
        "datasets",
        "model2vec",
        "numpy",
        "pandas",
        "click",
        "rich"
    ]
    
    for package in essential_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            results["packages"][package] = {"available": True, "version": version}
        except ImportError as e:
            results["packages"][package] = {"available": False, "error": str(e)}
            results["errors"].append(f"Missing package: {package}")
    
    # Check device availability (simple)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            results["device"] = "mps"
        elif torch.cuda.is_available():
            results["device"] = "cuda"
        else:
            results["device"] = "cpu"
    except:
        results["device"] = "cpu"
    
    # Set overall status
    if results["errors"]:
        results["status"] = "failed"
    elif results["warnings"]:
        results["status"] = "warning"
    else:
        results["status"] = "passed"
    
    return results


def simple_validate_path(path_str: str) -> Path:
    """Simple path validation - just convert and create if needed."""
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


class SimpleConfig:
    """
    Simplified configuration - just the essentials with sensible defaults.
    """
    
    def __init__(self, **overrides):
        """Initialize with defaults and apply overrides."""
        # Set sensible defaults
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.cache_dir = Path(".cache")
        
        # Core settings
        self.default_model = "BAAI/bge-m3"
        self.dataset_name = "ccdv/govreport-summarization"
        self.batch_size = 32
        self.max_workers = 1
        self.device = "auto"
        self.verbose = False
        
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if self.device == "auto":
            self.device = self._detect_device()
    
    def _detect_device(self) -> str:
        """Simple device detection."""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except:
            return "cpu"
    
    def validate(self):
        """Simple validation - just check that directories exist."""
        for dir_path in [self.models_dir, self.results_dir, self.cache_dir]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)


class ValidationError(Exception):
    """Simple validation error."""
    pass