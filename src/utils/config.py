"""
Configuration management for the pipeline.
Handles paths, settings, and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field

import json


@dataclass
class Config:
    """Configuration class for the pipeline."""
    
    # Core directories
    project_root: Path = field(default_factory=lambda: Path.cwd())
    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models")
    results_dir: Path = field(default_factory=lambda: Path.cwd() / "results")
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / ".cache")
    
    # Model settings
    default_model: str = "BAAI/bge-m3"
    default_pca_dims: int = 256
    
    # Dataset settings
    dataset_name: str = "ccdv/govreport-summarization"
    dataset_split: str = "test"
    
    # Processing settings
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
    
    # Performance
    max_workers: int = 4
    streaming: bool = True
    
    def __post_init__(self):
        """Initialize configuration after object creation."""
        # Convert string paths to Path objects
        self.models_dir = Path(self.models_dir)
        self.results_dir = Path(self.results_dir)
        self.cache_dir = Path(self.cache_dir)
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device automatically if needed
        if self.device == "auto":
            self.device = self._detect_device()
        
        # Update logging level based on verbose flag
        if self.verbose:
            self.log_level = "DEBUG"
    
    def __init__(self, **kwargs):
        """Initialize configuration with keyword arguments."""
        # Set defaults first
        self.project_root = kwargs.get('project_root', Path.cwd())
        self.models_dir = kwargs.get('models_dir', Path.cwd() / "models")
        self.results_dir = kwargs.get('results_dir', Path.cwd() / "results")
        self.cache_dir = kwargs.get('cache_dir', Path.cwd() / ".cache")
        
        self.default_model = kwargs.get('default_model', "BAAI/bge-m3")
        self.default_pca_dims = kwargs.get('default_pca_dims', 256)
        
        self.dataset_name = kwargs.get('dataset_name', "ccdv/govreport-summarization")
        self.dataset_split = kwargs.get('dataset_split', "test")
        
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_length = kwargs.get('max_length', 512)
        self.device = kwargs.get('device', "auto")
        
        self.verbose = kwargs.get('verbose', False)
        self.log_level = kwargs.get('log_level', "INFO")
        
        self.max_workers = kwargs.get('max_workers', 4)
        self.streaming = kwargs.get('streaming', True)
        
        # Run post-init
        self.__post_init__()
    
    @classmethod
    def from_file(cls, config_file: Path) -> "Config":
        """Load configuration from JSON file."""
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    @classmethod 
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "GOVREPORT_MODELS_DIR": "models_dir",
            "GOVREPORT_RESULTS_DIR": "results_dir", 
            "GOVREPORT_DEFAULT_MODEL": "default_model",
            "GOVREPORT_BATCH_SIZE": "batch_size",
            "GOVREPORT_MAX_WORKERS": "max_workers",
            "GOVREPORT_DEVICE": "device",
            "GOVREPORT_VERBOSE": "verbose"
        }
        
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if config_key in ["batch_size", "max_workers", "default_pca_dims"]:
                    value = int(value)
                elif config_key == "verbose":
                    value = value.lower() in ("true", "1", "yes")
                elif config_key in ["models_dir", "results_dir", "cache_dir"]:
                    value = Path(value)
                
                env_config[config_key] = value
        
        return cls(**env_config)
    
    def _detect_device(self) -> str:
        """Automatically detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def save(self, config_file: Path) -> None:
        """Save configuration to file."""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path where a distilled model should be saved."""
        # Sanitize model name for filesystem
        safe_name = model_name.replace("/", "_").replace(":", "_")
        return self.models_dir / f"{safe_name}_distilled"
    
    def get_results_path(self, experiment_name: str) -> Path:
        """Get the path where analysis results should be saved."""
        return self.results_dir / f"{experiment_name}.csv"
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if self.default_pca_dims <= 0:
            errors.append("default_pca_dims must be positive")
        
        if self.device not in ["cpu", "cuda", "mps", "auto"]:
            errors.append(f"Invalid device: {self.device}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")