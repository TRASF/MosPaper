import yaml
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Generic, Type, cast
import glob # Added for path_globs
import numpy as np # Added for log_experiment

# Configure module logger
logger = logging.getLogger(__name__)

# Helper to resolve expressions like "n_fft // 2 + 1"
def _resolve_expression(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            # Try to evaluate if it looks like an expression
            if any(op in value for op in ['+', '-', '*', '/', '%', '//']):
                return eval(value, {}, context) # Provide context for evaluation
        except Exception:
            pass # If eval fails, return original string
    return value

class AudioConfig:
    """Audio processing configuration with validation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize audio processing configuration with validation.
        
        Args:
            config_dict: Dictionary containing audio configuration
                         (typically from the main config's 'audio' key).
        """
        self.sample_rate: int = int(config_dict.get('sample_rate', 8000))
        self.audio_length: float = float(config_dict.get('audio_length', 0.3)) # in seconds
        # Default filter settings if not provided
        self.filter_lowcut: int = int(config_dict.get('filter_lowcut', 100)) 
        self.filter_highcut: int = int(config_dict.get('filter_highcut', 3000))
        self.filter_type: str = str(config_dict.get('filter_type', 'bandpass'))
        self.filter_order: int = int(config_dict.get('filter_order', 4))
        self.normalize_method: str = str(config_dict.get('normalize_method', 'rms'))
        self.pad_mode: str = str(config_dict.get('pad_mode', 'reflect'))
        
        self._validate()
    
    def _validate(self) -> None:
        """Validate audio configuration parameters."""
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.audio_length <= 0:
            raise ValueError(f"Audio length must be positive, got {self.audio_length}")
        # Add other validations as needed (e.g., filter cutoffs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audio configuration to dictionary."""
        return self.__dict__


class STFTConfig:
    """STFT processing configuration with validation."""
    
    def __init__(self, config_dict: Dict[str, Any], audio_config: AudioConfig):
        """
        Initialize STFT processing configuration.
        
        Args:
            config_dict: Dictionary containing STFT configuration 
                         (typically from the main config's 'stft' key).
            audio_config: The AudioConfig instance for dependent calculations.
        """
        self.n_fft: int = int(config_dict.get('n_fft', 512))
        self.hop_length: int = int(config_dict.get('hop_length', 256))
        self.window: str = str(config_dict.get('window', 'hann'))
        self.center: bool = bool(config_dict.get('center', True))
        self.pad_mode: str = str(config_dict.get('pad_mode_stft', 'reflect')) # Use a distinct key if different from audio padding
        self.power: float = float(config_dict.get('power', 2.0)) # For magnitude spectrogram
        self.norm: Optional[str] = config_dict.get('norm', None) # e.g. 'slaney' for librosa mel

        # Resolve freq_bins and time_frames, potentially using expressions
        stft_context = {'n_fft': self.n_fft}
        self.freq_bins: int = int(_resolve_expression(config_dict.get('freq_bins', self.n_fft // 2 + 1), stft_context))

        # Calculate time_frames based on audio properties
        samples = int(audio_config.sample_rate * audio_config.audio_length)
        if samples >= self.n_fft:
            self.time_frames: int = 1 + (samples - self.n_fft) // self.hop_length
        else:
            # If audio is shorter than n_fft, result is 1 frame (librosa behavior with padding)
            self.time_frames: int = 1 
            logger.warning(
                f"Audio length ({audio_config.audio_length}s at {audio_config.sample_rate}Hz = {samples} samples) "
                f"is shorter than n_fft ({self.n_fft}). STFT will produce 1 time frame."
            )
        
        # Allow override from config if explicitly provided
        if 'time_frames' in config_dict:
            self.time_frames = int(config_dict['time_frames'])

        self.feature_type: str = str(config_dict.get('feature_type', 'stft')) # 'stft', 'mel_spectrogram', etc.
        
        self._validate()

    def _validate(self) -> None:
        """Validate STFT configuration parameters."""
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.freq_bins <= 0:
             raise ValueError(f"freq_bins must be positive, got {self.freq_bins}")
        if self.time_frames <= 0:
             raise ValueError(f"time_frames must be positive, got {self.time_frames}")
        # Add other validations (e.g., window type if using a library that restricts it)

    def to_dict(self) -> Dict[str, Any]:
        """Convert STFT configuration to dictionary."""
        return self.__dict__

class DatasetConfig:
    """Dataset configuration with validation."""
    def __init__(self, config_dict: Dict[str, Any], project_root: Path):
        self.project_root = project_root
        self.path_globs: List[str] = config_dict.get('path_globs', [])
        self.class_dict: Dict[str, int] = config_dict.get('class_dict', {})
        self.num_classes: int = int(config_dict.get('num_classes', len(self.class_dict)))
        
        self.split_mode: str = str(config_dict.get('split_mode', 'ratio'))
        self.train_ratio: float = float(config_dict.get('train_ratio', 0.7))
        self.val_ratio: float = float(config_dict.get('val_ratio', 0.15))
        self.test_ratio: float = float(config_dict.get('test_ratio', 0.15))
        
        self.train_samples: int = int(config_dict.get('train_samples', 100)) # Per class for 'fixed' mode
        self.val_samples: int = int(config_dict.get('val_samples', 20))   # Per class for 'fixed' mode
        self.test_samples: int = int(config_dict.get('test_samples', 20))  # Per class for 'fixed' mode
        
        self.unlabeled_data_dir: Optional[str] = config_dict.get('unlabeled_data_dir', None)
        
        self.ssl_labels_per_class: int = int(config_dict.get('ssl_labels_per_class', 50))
        self.save_ssl_indices: bool = bool(config_dict.get('save_ssl_indices', False))
        self.load_ssl_indices_path: Optional[str] = config_dict.get('load_ssl_indices_path', None)

        self.paths: List[Path] = self._resolve_file_paths()
        self._validate()

    def _resolve_file_paths(self) -> List[Path]:
        resolved_paths = []
        for pattern in self.path_globs:
            # Ensure glob patterns are resolved relative to the project root
            abs_pattern = self.project_root / pattern
            for path_str in glob.glob(str(abs_pattern)):
                 resolved_paths.append(Path(path_str))
        
        # No explicit log for missing files; attempt fallback silently
        try:
            from config.path import dataset as fallback_list
            resolved_paths = [Path(p) for p in fallback_list]
        except ImportError:
            # Ignore fallback import errors silently
            pass

        # Ensure uniqueness of resolved paths
        return list(set(resolved_paths)) # Ensure uniqueness

    def _validate(self) -> None:
        if not self.class_dict:
            raise ValueError("class_dict cannot be empty.")
        if self.num_classes != len(self.class_dict):
            raise ValueError(f"num_classes ({self.num_classes}) does not match the number of "
                             f"entries in class_dict ({len(self.class_dict)}).")
        if self.split_mode not in ['ratio', 'fixed', 'max_train']:
            raise ValueError(f"Invalid split_mode: {self.split_mode}")
        if self.split_mode == 'ratio' and not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1 and 0 < self.test_ratio < 1 and \
                                              abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-9):
            raise ValueError("Train, val, and test ratios must be between 0 and 1 and sum to 1.")
        # Add more specific validations for fixed counts if needed

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset configuration to dictionary, excluding project_root."""
        d = self.__dict__.copy()
        d.pop('project_root', None) 
        d['paths'] = [str(p) for p in self.paths] # Convert Path objects to strings for serialization
        return d

class AugmentationConfig:
    """Augmentation configuration."""
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is None:
            config_dict = {} # Default to empty if no config provided
        self.add_noise: bool = bool(config_dict.get('add_noise', False))
        self.noise_level: float = float(config_dict.get('noise_level', 0.005))
        self.time_shift: bool = bool(config_dict.get('time_shift', False))
        self.shift_factor: float = float(config_dict.get('shift_factor', 0.1)) # Max shift as fraction of audio length
        # Strong augmentation specific
        self.time_stretch: bool = bool(config_dict.get('time_stretch', False))
        self.stretch_factor_range: List[float] = config_dict.get('stretch_factor_range', [0.8, 1.2])
        self.pitch_shift: bool = bool(config_dict.get('pitch_shift', False))
        self.pitch_shift_range: List[float] = config_dict.get('pitch_shift_range', [-2.0, 2.0]) # In semitones
        self.freq_mask: bool = bool(config_dict.get('freq_mask', False))
        self.freq_mask_width: int = int(config_dict.get('freq_mask_width', 10)) # Number of freq bins to mask
        self.time_mask: bool = bool(config_dict.get('time_mask', False)) # Added time masking
        self.time_mask_width: int = int(config_dict.get('time_mask_width', 10)) # Number of time steps to mask
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class SSLConfig:
    """SSL algorithm specific configuration."""
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is None:
            config_dict = {}
        self.mu: int = int(config_dict.get('mu', 7)) # Multiplier for unlabeled batch size
        self.confidence_threshold: float = float(config_dict.get('confidence_threshold', 0.95))
        self.lambda_u: float = float(config_dict.get('lambda_u', 1.0)) # Weight for unsupervised loss
        self.warmup_steps: int = int(config_dict.get('warmup_steps', 0)) # For lambda_u or learning rate scheduling
        
        # Temperature for sharpening - used in FixMatch/FlexMatch
        self.T: float = float(config_dict.get('T', 1.0))
        
        # Distribution Alignment (DA) settings for FlexMatch  
        self.use_DA: bool = bool(config_dict.get('use_DA', False))
        self.p_target_uniform: bool = bool(config_dict.get('p_target_uniform', True))
        
        # Exponential Moving Average settings
        self.use_ema: bool = bool(config_dict.get('use_ema', False))
        self.ema_decay: float = float(config_dict.get('ema_decay', 0.999))
        self.evaluate_ema_model: bool = bool(config_dict.get('evaluate_ema_model', True))
        
        # Dynamic training configurations
        self.dynamic_lambda_u: bool = bool(config_dict.get('dynamic_lambda_u', False))
        self.dynamic_threshold: bool = bool(config_dict.get('dynamic_threshold', False))
        
        # Steps per epoch configuration for SSL
        self.steps_per_epoch_unlabeled: Optional[int] = config_dict.get('steps_per_epoch_unlabeled', None)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class Config:
    """
    Main configuration class. Loads from YAML, allows overrides, and manages nested configs.
    """
    _instance: Optional['Config'] = None

    def __new__(cls, *args, **kwargs):
        # Allow re-initialization for testing or dynamic changes if needed,
        # but typically, it behaves like a singleton after first init.
        # if cls._instance is None:
        # cls._instance = super(Config, cls).__new__(cls)
        # return cls._instance
        # For this refactor, let's allow new instances if path or overrides change.
        return super(Config, cls).__new__(cls)

    def __init__(self, 
                 config_path: Union[str, Path] = 'config/base_config.yaml',
                 initial_dataset_paths: Optional[List[str]] = None,                 auto_save: bool = False,
                 **kwargs: Any):
        """
        Initialize configuration.

        Args:
            config_path: Path to the YAML configuration file.
            initial_dataset_paths: Legacy, prefer path_globs in YAML.
            auto_save: If True, saves the final config to the results directory.
            **kwargs: Overrides for config values.
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent # Project root: MosPaper/
        self.config_path = self.project_root / config_path
        
        # Load base configuration from YAML
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self._config_dict: Dict[str, Any] = yaml.safe_load(f)

        # Apply command-line overrides
        self._apply_overrides(kwargs)
        
        # Initialize general settings
        self.experiment_name: str = str(self._config_dict.get('experiment_name', 'mosquito_experiment'))
        self.seed: int = int(self._config_dict.get('seed', 42))
        self.device: str = str(self._config_dict.get('device', 'auto')) # 'cuda', 'cpu', 'auto'
        self.results_dir_base: Path = self.project_root / str(self._config_dict.get('results_dir', 'results'))
        self.log_interval: int = int(self._config_dict.get('log_interval', 10))
        self.data_type: str = str(self._config_dict.get('data_type', 'raw')) # 'raw' or 'stft'

        # Initialize nested configuration objects
        # Pass the relevant section of the main config dict to each sub-config
        self.audio = AudioConfig(self._config_dict) # AudioConfig now picks 'sample_rate', 'audio_length' from top level
        self.dataset = DatasetConfig(self._config_dict.get('dataset', {}), self.project_root)
        
        # STFTConfig depends on AudioConfig for some calculations
        # It also needs its own section from the config dict ('n_fft', 'hop_length', etc.)
        # and the top-level audio_length and sample_rate for time_frames calc if not in its own section.
        # For now, STFTConfig constructor is adapted to take the full _config_dict and audio_config.
        stft_params_from_main_config = {
            k: v for k, v in self._config_dict.items() 
            if k in ['n_fft', 'hop_length', 'window', 'center', 'pad_mode_stft', 
                     'power', 'norm', 'freq_bins', 'time_frames', 'feature_type']
        }
        self.stft = STFTConfig(stft_params_from_main_config, self.audio)

        # Correctly initialize weak and strong augmentation from the 'augmentation' block
        augmentation_settings = self._config_dict.get('augmentation', {})
        self.weak_augmentation = AugmentationConfig(augmentation_settings.get('weak', {}))
        self.strong_augmentation = AugmentationConfig(augmentation_settings.get('strong', {}))
        
        self.ssl = SSLConfig(self._config_dict.get('ssl'))

        # Training settings
        self.training_mode: str = str(self._config_dict.get('training_mode', 'supervised'))
        self.batch_size: int = int(self._config_dict.get('batch_size', 32))
        self.unlabeled_batch_size: int = int(self._config_dict.get('unlabeled_batch_size', self.batch_size))
        self.num_epochs: int = int(self._config_dict.get('num_epochs', 100))
        self.learning_rate: float = float(self._config_dict.get('learning_rate', 0.001))
        self.early_stop_patience: int = int(self._config_dict.get('early_stop_patience', 10))
        self.checkpoint_interval: int = int(self._config_dict.get('checkpoint_interval', 10))
        # Steps per epoch might be better derived from dataset size / batch_size
        self.waveform_steps_per_epoch: Optional[int] = self._config_dict.get('waveform_steps_per_epoch')
        self.stft_steps_per_epoch: Optional[int] = self._config_dict.get('stft_steps_per_epoch')

        # Early Stopping configuration
        self.early_stop_monitor: str = str(self._config_dict.get('early_stop_monitor', 'val_loss'))
        self.early_stop_min_delta: float = float(self._config_dict.get('early_stop_min_delta', 0))
        self.early_stop_restore_best: bool = bool(self._config_dict.get('early_stop_restore_best', True))
        
        # ReduceLROnPlateau configuration  
        self.reduce_lr_patience: int = int(self._config_dict.get('reduce_lr_patience', 0))
        self.reduce_lr_monitor: str = str(self._config_dict.get('reduce_lr_monitor', 'val_loss'))
        self.reduce_lr_factor: float = float(self._config_dict.get('reduce_lr_factor', 0.1))
        self.reduce_lr_min_delta: float = float(self._config_dict.get('reduce_lr_min_delta', 0))
        self.reduce_lr_min_lr: float = float(self._config_dict.get('reduce_lr_min_lr', 1e-6))
        
        # ModelCheckpoint configuration
        self.checkpoint_dir: Optional[str] = self._config_dict.get('checkpoint_dir', None)
        self.checkpoint_monitor: str = str(self._config_dict.get('checkpoint_monitor', 'val_loss'))
        self.checkpoint_save_best_only: bool = bool(self._config_dict.get('checkpoint_save_best_only', True))
        self.checkpoint_save_weights_only: bool = bool(self._config_dict.get('checkpoint_save_weights_only', False))
        
        # TensorBoard configuration
        self.tensorboard_log_dir: Optional[str] = self._config_dict.get('tensorboard_log_dir', None)
        
        # Learning rate scheduling configuration
        self.lr_decay_steps: int = int(self._config_dict.get('lr_decay_steps', 10000))
        self.lr_decay_rate: float = float(self._config_dict.get('lr_decay_rate', 0.95))
        
        # Steps per epoch configuration
        self.steps_per_epoch: Optional[int] = self._config_dict.get('steps_per_epoch', None)
        
        # Error handling configuration
        self.raise_training_exceptions: bool = bool(self._config_dict.get('raise_training_exceptions', False))

        # Model settings
        self.model_type: str = str(self._config_dict.get('model_type', 'PureWingbeat'))
        
        # Hyperparameter tuning (can be its own class if it grows complex)
        self.hyperparameter_tuning: Dict[str, Any] = self._config_dict.get('hyperparameter_tuning', {'enabled': False})

        # Create a unique directory for this experiment run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_run_name: str = f"{self.experiment_name}_{timestamp}"
        self.results_actual_dir: Path = self.results_dir_base / self.experiment_run_name
        self.results_actual_dir.mkdir(parents=True, exist_ok=True)
        
        if auto_save:
            self.save_config(self.results_actual_dir / 'final_config.yaml')

        self._validate_config()
        logger.info(f"Configuration loaded for experiment: {self.experiment_run_name}")
        logger.info(f"Results will be saved to: {self.results_actual_dir}")

    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply kwargs overrides to the loaded config_dict. Supports nested keys via dot notation."""
        for key, value in overrides.items():
            if value is None: # Skip None overrides from argparse defaults
                continue
            
            parts = key.split('.')
            d = self._config_dict
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
            logger.info(f"Config override: {key} = {value}")

    def _validate_config(self):
        """Perform basic validation of top-level config elements."""
        if self.data_type not in ['raw', 'stft']:
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be 'raw' or 'stft'.")
        if self.training_mode not in ['supervised', 'fixmatch', 'flexmatch', 'evaluate']:
            raise ValueError(f"Invalid training_mode: {self.training_mode}.")
        if self.model_type not in ['MosSong+', 'PureWingbeat']: # Add other models as they are created
            raise ValueError(f"Invalid model_type: {self.model_type}")
        if not self.dataset.paths and self.training_mode != 'evaluate': # Allow no paths for eval if checkpoint is given
             logger.warning("No data paths found in dataset configuration. This might be an issue unless in evaluation mode with a checkpoint.")


    def get(self, key: str, default: Any = None) -> Any:
        """Access config values using dot notation for nesting."""
        try:
            value = self._config_dict
            for k in key.split('.'):
                value = value[k]
            return value
        except KeyError:
            return default
        except TypeError: # Handle cases where a part of the path is not a dict
            logger.warning(f"Config key '{key}' part resolved to a non-dictionary type.")
            return default


    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access config.get('some.key')."""
        val = self.get(key)
        if val is None:
            raise KeyError(f"Configuration key '{key}' not found.")
        return val

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-like access, e.g., config.some_value or config.nested.value."""
        # This makes it easy to access top-level keys as attributes.
        # For nested access, it's better to use config.section.key or config.get('section.key')
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self._config_dict:
            # If it's a dictionary, we might want to return a sub-config object or the dict itself
            # For simplicity, let's return the raw value from _config_dict if it's a top-level key
            return self._config_dict[name]
        # Check for log_experiment explicitly to avoid confusion if it's not found elsewhere
        if name == 'log_experiment' and not hasattr(self, '_log_experiment_defined_elsewhere'):
             # This path should ideally not be taken if the method is defined below.
             # This is a safeguard or for clarity during debugging.
             pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' and it's not a top-level config key.")

    def log_experiment(self, metrics_to_log: Dict[str, Any]):
        """Logs a dictionary of metrics to a JSONL file in the experiment's results directory."""
        log_file_path = self.results_actual_dir / 'experiment_log.jsonl'
        
        serializable_metrics = {}
        for key, value in metrics_to_log.items():
            if isinstance(value, np.generic): # Handles numpy scalars like np.float32, np.int64
                serializable_metrics[key] = value.item()
            elif isinstance(value, np.ndarray): # Handles numpy arrays
                serializable_metrics[key] = value.tolist() # Convert array to list
            elif isinstance(value, Path): # Handle Path objects
                serializable_metrics[key] = str(value)
            else:
                serializable_metrics[key] = value
                
        try:
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(serializable_metrics) + '\\n')
            logger.debug(f"Logged metrics to {log_file_path}: {serializable_metrics}")
        except Exception as e:
            logger.error(f"Failed to log experiment metrics to {log_file_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation of the full config."""
        serializable_dict = {}
        # Top-level attributes
        for key, value in self.__dict__.items():
            if key.startswith('_') or key == 'project_root' or key == 'config_path' or key == 'results_dir_base':
                continue
            if hasattr(value, 'to_dict') and callable(value.to_dict):
                serializable_dict[key] = value.to_dict()
            elif isinstance(value, Path):
                serializable_dict[key] = str(value)
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                 serializable_dict[key] = value
            # else:
                # logger.debug(f"Skipping non-serializable attribute {key} of type {type(value)}")

        # Ensure all original _config_dict items that weren't converted to objects are included
        # This can be tricky if overrides changed types. A safer way is to build from _config_dict
        # and then update with the object versions.
        
        # Rebuild from the (potentially overridden) _config_dict and then update with structured objects
        # This ensures that any keys not explicitly handled by a sub-config class are still present.
        output_dict = self._config_dict.copy()
        
        output_dict['experiment_name'] = self.experiment_name
        output_dict['seed'] = self.seed
        output_dict['device'] = self.device
        output_dict['results_actual_dir'] = str(self.results_actual_dir) # Add the actual results dir
        output_dict['log_interval'] = self.log_interval
        output_dict['data_type'] = self.data_type
        
        output_dict['audio'] = self.audio.to_dict()
        output_dict['dataset'] = self.dataset.to_dict()
        output_dict['stft'] = self.stft.to_dict()
        output_dict['weak_augmentation'] = self.weak_augmentation.to_dict()
        output_dict['strong_augmentation'] = self.strong_augmentation.to_dict()
        output_dict['ssl'] = self.ssl.to_dict()
        
        output_dict['training_mode'] = self.training_mode
        output_dict['batch_size'] = self.batch_size
        output_dict['num_epochs'] = self.num_epochs
        output_dict['learning_rate'] = self.learning_rate
        output_dict['early_stop_patience'] = self.early_stop_patience
        output_dict['checkpoint_interval'] = self.checkpoint_interval
        if self.waveform_steps_per_epoch is not None:
            output_dict['waveform_steps_per_epoch'] = self.waveform_steps_per_epoch
        if self.stft_steps_per_epoch is not None:
            output_dict['stft_steps_per_epoch'] = self.stft_steps_per_epoch
            
        output_dict['model_type'] = self.model_type
        output_dict['hyperparameter_tuning'] = self.hyperparameter_tuning
        
        return output_dict

    def save_config(self, path: Union[str, Path]):
        """Save the current configuration to a YAML file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config_to_save = self.to_dict()
        
        # Remove non-essential or runtime-generated paths from the saved file if desired
        # For example, dataset.paths is resolved at runtime, might not need to save it if path_globs is saved.
        if 'dataset' in config_to_save and 'paths' in config_to_save['dataset']:
            # Keep path_globs, remove resolved paths for cleaner saved config
            # config_to_save['dataset'].pop('paths', None) 
            pass # Decided to keep resolved paths for full reproducibility of what was run

        with open(save_path, 'w') as f:
            yaml.dump(config_to_save, f, sort_keys=False, default_flow_style=False)
        logger.info(f"Configuration saved to {save_path}")

# Example usage (for testing within this file)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy base_config.yaml for testing
    dummy_config_content = {
        'experiment_name': 'test_experiment',
        'seed': 123,
        'data_type': 'stft',
        'sample_rate': 16000, # For AudioConfig
        'audio_length': 0.5,  # For AudioConfig
        'dataset': {
            'path_globs': ['dummy_data/*.wav'], # This will be empty unless files exist
            'class_dict': {'class_a': 0, 'class_b': 1},
            'num_classes': 2,
            'split_mode': 'ratio',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'ssl_labels_per_class': 10 # Added for testing nested override
        },
        'n_fft': 1024, # For STFTConfig
        'hop_length': 512, # For STFTConfig
        'model_type': 'MosSong+',
        'training_mode': 'supervised',
        'results_dir': 'temp_results'
    }
    dummy_yaml_path = Path(__file__).parent.parent.parent / 'config' / 'dummy_test_config.yaml'
    with open(dummy_yaml_path, 'w') as f:
        yaml.dump(dummy_config_content, f)

    try:
        # Test with CLI overrides
        cfg = Config(
            config_path='config/dummy_test_config.yaml', 
            auto_save=True,
            training_mode='fixmatch',  # Override
            # For nested overrides, the key should be a flat string with dot notation
            **{'dataset.ssl_labels_per_class': 20, 'learning_rate': 0.005}
        )
        print("Config loaded successfully.")
        print(f"Experiment name: {cfg.experiment_name}")
        print(f"Training mode: {cfg.training_mode}") # Should be fixmatch
        print(f"Audio sample rate: {cfg.audio.sample_rate}")
        print(f"STFT n_fft: {cfg.stft.n_fft}")
        print(f"STFT time_frames: {cfg.stft.time_frames}")
        print(f"Dataset paths: {cfg.dataset.paths}") # Will be empty if dummy_data/*.wav doesn't exist
        print(f"Dataset SSL labels per class: {cfg.dataset.ssl_labels_per_class}") # Should be 20
        print(f"Learning rate: {cfg.learning_rate}") # Should be 0.005
        
        # Test serialization
        cfg_dict = cfg.to_dict()
        print("\\nSerialized Config:")
        # print(json.dumps(cfg_dict, indent=2)) # Path objects are not JSON serializable by default
        
        # Check if results dir was created
        print(f"\\nResults directory: {cfg.results_actual_dir}")
        assert cfg.results_actual_dir.exists()
        assert (cfg.results_actual_dir / 'final_config.yaml').exists()
        print("final_config.yaml was saved.")

    except Exception as e:
        logger.error(f"Error during Config testing: {e}", exc_info=True)
    finally:
        # Clean up dummy config
        if dummy_yaml_path.exists():
            # os.remove(dummy_yaml_path) # Keep for inspection if needed
            pass
        # Clean up dummy results
        # import shutil
        # dummy_results_path = Path(__file__).resolve().parent.parent.parent / 'temp_results'
        # if dummy_results_path.exists():
        #     shutil.rmtree(dummy_results_path)
        pass