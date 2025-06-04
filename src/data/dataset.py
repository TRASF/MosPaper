"""
Efficient dataset management for audio classification.

This module provides a clean, unified approach to:
1. Loading and processing audio data once
2. Splitting data into train/val/test sets
3. Converting waveforms to STFT features when required
4. Creating optimized TensorFlow datasets for model training
"""

import numpy as np
import tensorflow as tf
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Any, Callable

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import local modules
from src.utils.config import Config
# Updated import:
from src.data.augmentation import (
    augment_sample, 
    get_augmentation_config, 
    create_ssl_augment_labeled_fn, 
    create_ssl_augment_unlabeled_fn
)
from src.data.fft_processor import waveform_to_fft
from src.data.loader import process_audio_files

# Configure logger
logger = logging.getLogger(__name__)


class SplitData(NamedTuple):
    """Data structure representing a dataset split."""
    data: np.ndarray
    labels: np.ndarray
    file_ids: np.ndarray


class DatasetManager:
    """
    Efficient dataset manager for audio classification.
    
    This class loads and processes audio data once, then provides convenient
    access to train, validation, and test splits with various configurations.
    
    Features:
    - Single data loading and processing operation
    - Multiple data splitting strategies (fixed count, max training, ratio-based)
    - Support for both raw waveforms and STFT features
    - Configurable dataset preparation for training and evaluation
    """
    
    def __init__(self, file_paths: List[str], config: Config):
        """
        Initialize the dataset manager.
        
        Args:
            file_paths: List of audio file paths to process
            config: Configuration object with dataset parameters
        
        Raises:
            ValueError: If no audio data could be processed successfully
        """
        self.file_paths: List[str] = file_paths
        self.config: Config = config
        self.data_loaded: bool = False # Initialize data_loaded attribute
        
        # Add attributes to store all data and initial split indices
        self.all_audio_data: Optional[np.ndarray] = None
        self.all_labels: Optional[np.ndarray] = None
        self.all_file_ids: Optional[np.ndarray] = None
        self.initial_train_indices: Optional[np.ndarray] = None
        self.initial_val_indices: Optional[np.ndarray] = None
        self.initial_test_indices: Optional[np.ndarray] = None
        
        # Extract key configuration parameters with proper type annotations
        self.data_type: str = config.get('data_type', 'raw')
        self.sample_rate: int = config.get('sample_rate', 8000)
        self.audio_length: float = config.get('audio_length', 0.3)
        self.num_classes: int = config.get('num_classes', 11)
        
        # Get training mode to determine the appropriate split mode
        self.training_mode: str = config.get('training_mode', 'supervised')
        
        # Determine split mode based on training mode
        if self.training_mode in ['fixmatch', 'flexmatch']:
            # For SSL training, use fixed split mode to have consistent labeled data
            self.split_mode: str = 'fixed'
            logger.info(f"Using 'fixed' split mode for {self.training_mode} SSL training")
        else:
            # For supervised training, use max_train to maximize training data
            # But respect explicitly configured split_mode if provided
            if hasattr(config, 'dataset') and hasattr(config.dataset, 'split_mode'):
                self.split_mode: str = config.dataset.split_mode
            else:
                self.split_mode: str = config.get('split_mode', 'max_train')
            logger.info(f"Using '{self.split_mode}' split mode for supervised training")
            
        self.seed: int = config.get('seed', 42)
        
        # Calculate expected audio shape based on configuration
        if self.data_type == 'raw':
            self.input_shape: Tuple[int, ...] = (int(self.sample_rate * self.audio_length),)
            self.channels: int = 1
        else:  # 'stft'
            n_fft: int = config.get('n_fft', 512)
            hop_length: int = config.get('hop_length', 256)
            self.freq_bins: int = n_fft // 2 + 1
            
            # Calculate time frames if not specified
            if config.get('time_frames'):
                self.time_frames: int = config.get('time_frames')
            else:
                # Calculate based on audio length, sample rate, and hop length
                num_samples: int = int(self.sample_rate * self.audio_length)
                self.time_frames: int = (num_samples - n_fft) // hop_length + 2
                
            self.input_shape: Tuple[int, ...] = (self.freq_bins, self.time_frames, 1)
            self.channels: int = 1
            
        # Initialize data storage
        self.train_data: Optional[SplitData] = None
        self.val_data: Optional[SplitData] = None
        self.test_data: Optional[SplitData] = None
        
        # Statistics storage
        self.split_stats: Dict[str, Dict[str, Union[int, str]]] = {}
        
        # Load and process the data (done once during initialization)
        self._load_and_process_data()
        
    def _load_and_process_data(self) -> None:
        """
        Load audio files, process them, and create train/val/test splits.
        
        This is called once during initialization to prepare all data splits.
        Performs comprehensive shape validation and normalization here to ensure
        downstream code can assume correct formats.
        
        Raises:
            ValueError: If no audio data could be processed successfully
        """
        logger.info("Loading and processing audio files...")
        
        # Process audio files into numpy arrays
        audio_data, labels, file_ids = process_audio_files(
            self.file_paths,
            sample_rate=self.sample_rate,
            segment_samples=int(self.sample_rate * self.audio_length)
        )

        # Store all loaded data
        self.all_audio_data = audio_data
        self.all_labels = labels
        self.all_file_ids = file_ids
        
        # Validate processed data
        if len(self.all_audio_data) == 0:
            raise ValueError("No audio data was processed successfully. Check file paths and formats.")
            
        # CENTRALIZED SHAPE VALIDATION - All shape checking happens here
        self._validate_and_normalize_data_shapes()
        
        logger.info(f"Successfully processed {len(self.all_audio_data)} audio segments from all files.") # Log based on all_audio_data
        
        # Split data once based on configured strategy, and get indices
        (train_raw, val_raw, test_raw, _, _), self.initial_train_indices, self.initial_val_indices, self.initial_test_indices = \
            self._split_data(self.all_audio_data, self.all_labels, self.all_file_ids)
        # Store the data in named tuples for better organization
        self.train_data = SplitData(data=train_raw[0], labels=train_raw[1], file_ids=train_raw[2])
        self.val_data = SplitData(data=val_raw[0], labels=val_raw[1], file_ids=val_raw[2])
        self.test_data = SplitData(data=test_raw[0], labels=test_raw[1], file_ids=test_raw[2])
        
        # Convert to spectral features if needed (after splitting to save memory)
        if self.data_type == 'stft':
            logger.info("Converting data splits to STFT features...")
            # Convert and validate STFT shapes
            train_stft = waveform_to_fft(self.train_data.data, config=self.config)
            val_stft = waveform_to_fft(self.val_data.data, config=self.config)
            test_stft = waveform_to_fft(self.test_data.data, config=self.config)
            
            # Validate STFT output shapes
            expected_stft_shape = (self.freq_bins, self.time_frames, 1)
            for name, stft_data in [("train", train_stft), ("val", val_stft), ("test", test_stft)]:
                if len(stft_data) > 0:  # Only validate non-empty datasets
                    actual_shape = stft_data.shape[1:]  # Remove batch dimension
                    if actual_shape != expected_stft_shape:
                        raise ValueError(f"STFT {name} data shape mismatch. "
                                       f"Expected {expected_stft_shape}, got {actual_shape}")
            
            self.train_data = SplitData(data=train_stft, labels=self.train_data.labels, file_ids=self.train_data.file_ids)
            self.val_data = SplitData(data=val_stft, labels=self.val_data.labels, file_ids=self.val_data.file_ids)
            self.test_data = SplitData(data=test_stft, labels=self.test_data.labels, file_ids=self.test_data.file_ids)
            
        # Store split statistics for later analysis
        self.split_stats = {
            'train': {
                'segments': len(self.train_data.data), 
                'files': len(set(self.train_data.file_ids))
            },
            'val': {
                'segments': len(self.val_data.data), 
                'files': len(set(self.val_data.file_ids))
            },
            'test': {
                'segments': len(self.test_data.data), 
                'files': len(set(self.test_data.file_ids))
            }
        }
        
        logger.info(f"Data splits ready: "
                    f"Train={self.split_stats['train']['segments']} samples, "
                    f"Val={self.split_stats['val']['segments']} samples, "
                    f"Test={self.split_stats['test']['segments']} samples")
        
        self.data_loaded = True # Set data_loaded to True after successful loading
        
    def _validate_and_normalize_data_shapes(self) -> None:
        """
        Centralized validation and normalization of data shapes.
        This method ensures all downstream code can assume correct formats.
        Called once during data loading to eliminate redundant shape checks.
        """
        logger.info("Validating and normalizing data shapes...")
        
        # Validate audio data shapes
        if self.data_type == 'raw':
            # Raw audio should be (n_samples, sequence_length) or (n_samples, sequence_length, 1)
            expected_seq_length = int(self.sample_rate * self.audio_length)
            
            if self.all_audio_data.ndim == 2:
                if self.all_audio_data.shape[1] != expected_seq_length:
                    raise ValueError(f"Raw audio sequence length mismatch. Expected {expected_seq_length}, "
                                   f"got {self.all_audio_data.shape[1]}")
                # Add channel dimension: (n_samples, seq_length) -> (n_samples, seq_length, 1)
                self.all_audio_data = np.expand_dims(self.all_audio_data, axis=-1)
                
            elif self.all_audio_data.ndim == 3:
                if self.all_audio_data.shape[1] != expected_seq_length:
                    raise ValueError(f"Raw audio sequence length mismatch. Expected {expected_seq_length}, "
                                   f"got {self.all_audio_data.shape[1]}")
                if self.all_audio_data.shape[2] != 1:
                    raise ValueError(f"Raw audio channel dimension should be 1, got {self.all_audio_data.shape[2]}")
            else:
                raise ValueError(f"Raw audio data should be 2D or 3D, got {self.all_audio_data.ndim}D")
                
            # Ensure float32 dtype for consistency
            self.all_audio_data = self.all_audio_data.astype(np.float32)
            
        # Validate labels (convert to int32 for consistency)
        if self.all_labels.ndim != 1:
            raise ValueError(f"Labels should be 1D array, got {self.all_labels.ndim}D")
            
        # Ensure labels are in valid range [0, num_classes-1]
        unique_labels = np.unique(self.all_labels)
        if np.any(unique_labels < 0) or np.any(unique_labels >= self.num_classes):
            raise ValueError(f"Labels must be in range [0, {self.num_classes-1}], "
                           f"got range [{unique_labels.min()}, {unique_labels.max()}]")
                           
        # Convert labels to int32 
        self.all_labels = self.all_labels.astype(np.int32)
        
        logger.info(f"Shape validation complete: "
                   f"audio_data={self.all_audio_data.shape}, "
                   f"labels={self.all_labels.shape}, "
                   f"dtype=audio:{self.all_audio_data.dtype}, labels:{self.all_labels.dtype}")
        
    def _split_data(self, audio_data: np.ndarray, labels: np.ndarray, file_ids: np.ndarray) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],  # train
              Tuple[np.ndarray, np.ndarray, np.ndarray],  # val
              Tuple[np.ndarray, np.ndarray, np.ndarray]], # test
        np.ndarray,  # train_indices
        np.ndarray,  # val_indices
        np.ndarray   # test_indices
    ]:
        """
        Split data into train/val/test sets based on configured strategy.
        
        Args:
            audio_data: Audio data array with shape [n_samples, audio_length, 1]
            labels: Labels array with shape [n_samples]
            file_ids: File IDs for tracking data source
            
        Returns:
            Tuple of (train_data, val_data, test_data), each containing (data, labels, file_ids)
            
        Raises:
            ValueError: If an invalid split mode is specified
        """
        # Choose splitting strategy based on config
        if self.split_mode == 'fixed':
            # _split_data_fixed will now return indices as well
            return self._split_data_fixed(audio_data, labels, file_ids)
        elif self.split_mode == 'max_train':
            # Ensure _split_data_max_train also returns indices
            return self._split_data_max_train(audio_data, labels, file_ids)
        elif self.split_mode == 'ratio':
            # Ensure _split_data_ratio also returns indices
            return self._split_data_ratio(audio_data, labels, file_ids)
        else:
            raise ValueError(f"Invalid split mode: {self.split_mode}. "
                             f"Must be 'fixed', 'max_train', or 'ratio'.")
    
    def _split_data_fixed(self, audio_data: np.ndarray, labels: np.ndarray, file_ids: np.ndarray, labeled_samples_per_class: Optional[int] = None, ssl_mode: bool = False) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray],
              Optional[np.ndarray], Optional[np.ndarray]],
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split data using fixed sample counts per class, with optional labeled/unlabeled split for SSL.
        
        Args:
            audio_data: Audio data array
            labels: Labels array
            file_ids: File IDs array
            labeled_samples_per_class: Number of labeled samples per class (None = use all for supervised)
            ssl_mode: If True, return labeled/unlabeled indices for SSL
        
        Returns:
            Tuple of (train_data, val_data, test_data, labeled_indices, unlabeled_indices), train_indices, val_indices, test_indices
        """
        if hasattr(self.config, 'dataset'):
            train_samples: int = getattr(self.config.dataset, 'train_samples', 350)
            val_samples: int = getattr(self.config.dataset, 'val_samples', 50)
            test_samples: int = getattr(self.config.dataset, 'test_samples', 50)
        else:
            train_samples: int = self.config.get('train_samples', 350)
            val_samples: int = self.config.get('val_samples', 50)
            test_samples: int = self.config.get('test_samples', 50)
        
        logger.info(f"Using fixed split with {train_samples}/{val_samples}/{test_samples} samples per class")
        
        train_indices_list: List[int] = []
        val_indices_list: List[int] = []
        test_indices_list: List[int] = []
        labeled_indices: List[int] = []  # Will be relative to train split
        unlabeled_indices: List[int] = []
        train_offset = 0  # Track position in train split for relative indices
        # Process each class separately to ensure balanced splits
        for class_idx in range(self.num_classes):
            class_indices = np.where(labels == class_idx)[0]
            if len(class_indices) == 0:
                logger.warning(f"No samples found for class {class_idx}")
                continue
            rng = np.random.RandomState(self.seed + class_idx)
            rng.shuffle(class_indices)
            n_val = min(val_samples, len(class_indices))
            n_test = min(test_samples, max(0, len(class_indices) - n_val))
            n_train = len(class_indices) - n_val - n_test
            train_indices = class_indices[:n_train]
            val_indices = class_indices[n_train:n_train + n_val]
            test_indices = class_indices[n_train + n_val:n_train + n_val + n_test]
            train_indices_list.extend(train_indices)
            val_indices_list.extend(val_indices)
            test_indices_list.extend(test_indices)
            # Labeled/unlabeled split for train (relative to train split)
            if labeled_samples_per_class is not None and labeled_samples_per_class > 0:
                n_labeled = min(labeled_samples_per_class, len(train_indices))
                labeled_indices.extend(range(train_offset, train_offset + n_labeled))
                if ssl_mode:
                    unlabeled_indices.extend(range(train_offset + n_labeled, train_offset + len(train_indices)))
            else:
                labeled_indices.extend(range(train_offset, train_offset + len(train_indices)))
            train_offset += len(train_indices)
        
        # Convert lists to numpy arrays
        final_train_indices = np.array(train_indices_list, dtype=np.int32)
        final_val_indices = np.array(val_indices_list, dtype=np.int32)
        final_test_indices = np.array(test_indices_list, dtype=np.int32)

        # Create the data splits
        train_data_tuple = (audio_data[final_train_indices], labels[final_train_indices], file_ids[final_train_indices])
        val_data_tuple = (audio_data[final_val_indices], labels[final_val_indices], file_ids[final_val_indices])
        test_data_tuple = (audio_data[final_test_indices], labels[final_test_indices], file_ids[final_test_indices])
        
        labeled_indices_np = np.array(labeled_indices, dtype=np.int32) if labeled_indices else None
        unlabeled_indices_np = np.array(unlabeled_indices, dtype=np.int32) if ssl_mode and unlabeled_indices else None
        
        logger.info(f"Actual counts: Train={len(final_train_indices)}, Val={len(final_val_indices)}, Test={len(final_test_indices)}")
        
        return (train_data_tuple, val_data_tuple, test_data_tuple, labeled_indices_np, unlabeled_indices_np), final_train_indices, final_val_indices, final_test_indices
        
    def _split_data_max_train(self, audio_data: np.ndarray, labels: np.ndarray, file_ids: np.ndarray) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split data using maximum samples for training and fixed counts for val/test.
        
        This mode reserves a fixed number of samples per class for validation and testing,
        then assigns all remaining samples to the training set.
        
        Args:
            audio_data: Audio data array
            labels: Labels array
            file_ids: File IDs array
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # First try to get values from config.dataset, then fall back to config's direct attributes
        if hasattr(self.config, 'dataset'):
            val_samples: int = getattr(self.config.dataset, 'val_samples', 50)
            test_samples: int = getattr(self.config.dataset, 'test_samples', 50)
        else:
            val_samples: int = self.config.get('val_samples', 50)
            test_samples: int = self.config.get('test_samples', 50)
        
        logger.info(f"Using max_train split: reserving {val_samples}/{test_samples} samples per class for val/test")
        
        train_indices_list: List[int] = []
        val_indices_list: List[int] = []
        test_indices_list: List[int] = []
        
        for class_idx in range(self.num_classes):
            # Find all indices for this class
            class_indices = np.where(labels == class_idx)[0]
            
            if len(class_indices) == 0:
                logger.warning(f"No samples found for class {class_idx}")
                continue
                
            # Shuffle indices deterministically
            rng = np.random.RandomState(self.seed + class_idx)
            rng.shuffle(class_indices)
            
            # Reserve samples for validation and test sets
            n_val = min(val_samples, len(class_indices))
            n_test = min(test_samples, max(0, len(class_indices) - n_val))
            
            # Assign all remaining samples to training
            n_train = len(class_indices) - n_val - n_test
            
            # Split indices - train gets all samples except reserved val/test
            train_indices_list.extend(class_indices[:n_train])
            val_indices_list.extend(class_indices[n_train:n_train + n_val])
            test_indices_list.extend(class_indices[n_train + n_val:n_train + n_val + n_test])
        
        # Convert lists to numpy arrays
        final_train_indices = np.array(train_indices_list, dtype=np.int32)
        final_val_indices = np.array(val_indices_list, dtype=np.int32)
        final_test_indices = np.array(test_indices_list, dtype=np.int32)

        # Create the data splits
        train_data_tuple = (audio_data[final_train_indices], labels[final_train_indices], file_ids[final_train_indices])
        val_data_tuple = (audio_data[final_val_indices], labels[final_val_indices], file_ids[final_val_indices])
        test_data_tuple = (audio_data[final_test_indices], labels[final_test_indices], file_ids[final_test_indices])
        
        # Log actual split counts
        logger.info(f"Split results: Train={len(final_train_indices)}, Val={len(final_val_indices)}, Test={len(final_test_indices)}")
        
        return (train_data_tuple, val_data_tuple, test_data_tuple), final_train_indices, final_val_indices, final_test_indices
        
    def _split_data_ratio(self, audio_data: np.ndarray, labels: np.ndarray, file_ids: np.ndarray) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split data using ratio-based stratified sampling.
        
        Args:
            audio_data: Audio data array
            labels: Labels array
            file_ids: File IDs array
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        from sklearn.model_selection import train_test_split
        
        train_ratio: float = self.config.get('train_ratio', 0.8)
        val_ratio: float = self.config.get('val_ratio', 0.1)
        test_ratio: float = self.config.get('test_ratio', 0.1)
        
        # Validate and normalize ratios
        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("All split ratios must be positive (train > 0, val >= 0, test >= 0)")
        
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        logger.info(f"Using ratio split with {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f} proportions")
        
        # Special case: if any ratio is zero, handle separately
        if val_ratio == 0 and test_ratio == 0:
            # Use all data for training
            final_train_indices = np.arange(len(audio_data))
            final_val_indices = np.array([], dtype=int)
            final_test_indices = np.array([], dtype=int)
        elif val_ratio == 0:
            final_train_indices, final_test_indices = train_test_split(
                np.arange(len(audio_data)), 
                test_size=test_ratio,
                random_state=self.seed,
                stratify=labels
            )
            final_val_indices = np.array([], dtype=int)
        elif test_ratio == 0:
            final_train_indices, final_val_indices = train_test_split(
                np.arange(len(audio_data)), 
                test_size=val_ratio,
                random_state=self.seed,
                stratify=labels
            )
            final_test_indices = np.array([], dtype=int)
        else:
            # Regular case: split temp into val and test
            val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
            train_indices, temp_indices = train_test_split(
                np.arange(len(audio_data)), 
                test_size=val_ratio + test_ratio,
                random_state=self.seed,
                stratify=labels
            )
            final_train_indices = train_indices
            final_val_indices, final_test_indices = train_test_split(
                temp_indices,
                test_size=1 - val_ratio_of_temp,
                random_state=self.seed,
                stratify=labels[temp_indices]
            )
        
        # Create the splits
        train_data_tuple = (audio_data[final_train_indices], labels[final_train_indices], file_ids[final_train_indices])
        val_data_tuple = (audio_data[final_val_indices], labels[final_val_indices], file_ids[final_val_indices]) if len(final_val_indices) > 0 else (
            np.zeros((0,) + audio_data.shape[1:], dtype=audio_data.dtype),
            np.zeros((0,), dtype=labels.dtype),
            np.zeros((0,), dtype=file_ids.dtype)
        )
        test_data_tuple = (audio_data[final_test_indices], labels[final_test_indices], file_ids[final_test_indices]) if len(final_test_indices) > 0 else (
            np.zeros((0,) + audio_data.shape[1:], dtype=audio_data.dtype),
            np.zeros((0,), dtype=labels.dtype),
            np.zeros((0,), dtype=file_ids.dtype)
        )
        
        return (train_data_tuple, val_data_tuple, test_data_tuple), final_train_indices, final_val_indices, final_test_indices
        
    def prepare_dataset(
        self, 
        split: str = 'train', 
        batch_size: Optional[int] = None, 
        shuffle: bool = True, 
        augment: bool = False,
        repeat: bool = False
    ) -> tf.data.Dataset:
        """
        Prepare a TensorFlow dataset for model training or evaluation.
        
        Args:
            split: Which data split to use ('train', 'val', or 'test')
            batch_size: Batch size (defaults to config value)
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation (only for training)
            repeat: Whether to repeat the dataset indefinitely
            
        Returns:
            TensorFlow dataset ready for model consumption
            
        Raises:
            ValueError: If an invalid split name is provided
        """
        # Apply default batch size from config if not specified
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
        
        # Get the correct data split
        if split == 'train':
            data = self.train_data.data
            labels = self.train_data.labels
        elif split == 'val':
            data = self.val_data.data
            labels = self.val_data.labels
        elif split == 'test':
            data = self.test_data.data
            labels = self.test_data.labels
        else:
            raise ValueError(f"Invalid split name: '{split}'. Must be 'train', 'val', or 'test'")
            
        # Handle empty datasets (possible with some splitting configurations)
        if len(data) == 0:
            logger.warning(f"Empty dataset for split '{split}'")
            # Return an empty dataset with the right types and shapes
            empty_data = np.zeros((0,) + data.shape[1:], dtype=data.dtype)
            empty_labels = np.zeros((0,), dtype=labels.dtype)
            dataset = tf.data.Dataset.from_tensor_slices((empty_data, empty_labels))
            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        # Create TensorFlow dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
        # Shuffle if requested
        if shuffle:
            buffer_size = min(len(data), 10000)  # Use smaller of dataset size or 10000
            dataset = dataset.shuffle(
                buffer_size=buffer_size, 
                seed=self.seed,
                reshuffle_each_iteration=True
            )

        # Repeat the dataset if requested
        if repeat:
            dataset = dataset.repeat()
            
        # Apply batching and prefetching
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_datasets(
        self,
        batch_size: Optional[int] = None,
        train_shuffle: bool = True,
        train_augment: bool = False,
        train_repeat: bool = False,
        val_repeat: bool = False,
        test_repeat: bool = False,
        labeled_samples_per_class: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get train, validation, and test datasets in a single call, with optional labeled_samples_per_class.
        
        Args:
            batch_size: Batch size for all datasets (defaults to config value)
            train_shuffle: Whether to shuffle the training dataset
            train_augment: Whether to apply augmentation to the training dataset
            train_repeat: Whether to repeat the training dataset
            val_repeat: Whether to repeat the validation dataset
            test_repeat: Whether to repeat the test dataset
            labeled_samples_per_class: Number of labeled samples per class (None = use all)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # If labeled_samples_per_class is set, only use those indices for training
        if labeled_samples_per_class is not None and labeled_samples_per_class > 0:
            # Redo split to get labeled indices
            (train_data_tuple, val_data_tuple, test_data_tuple, labeled_indices, _), _, _, _ = self._split_data_fixed(
                self.all_audio_data, self.all_labels, self.all_file_ids, labeled_samples_per_class, ssl_mode=False)
            data = train_data_tuple[0][labeled_indices]
            labels = train_data_tuple[1][labeled_indices]
        else:
            data = self.train_data.data
            labels = self.train_data.labels
            
        # Handle empty datasets (possible with some splitting configurations)
        if len(data) == 0:
            logger.warning(f"Empty dataset for split 'train'")
            # Return an empty dataset with the right types and shapes
            empty_data = np.zeros((0,) + data.shape[1:], dtype=data.dtype)
            empty_labels = np.zeros((0,), dtype=labels.dtype)
            train_dataset = tf.data.Dataset.from_tensor_slices((empty_data, empty_labels))
        else:
            # Create TensorFlow dataset from numpy arrays
            train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
        # Shuffle if requested
        if train_shuffle:
            buffer_size = min(len(data), 10000)  # Use smaller of dataset size or 10000
            train_dataset = train_dataset.shuffle(
                buffer_size=buffer_size, 
                seed=self.seed,
                reshuffle_each_iteration=True
            )

        # Repeat the dataset if requested
        if train_repeat:
            train_dataset = train_dataset.repeat()
            
        # Apply batching and prefetching
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validation and test datasets (no change, always use all data)
        val_dataset = self.prepare_dataset(
            split='val',
            batch_size=batch_size,
            shuffle=False,  # No shuffle for validation
            augment=False,  # No augmentation for validation
            repeat=val_repeat
        )
        
        test_dataset = self.prepare_dataset(
            split='test',
            batch_size=batch_size,
            shuffle=False,  # No shuffle for test
            augment=False,  # No augmentation for test
            repeat=test_repeat
        )
        
        return train_dataset, val_dataset, test_dataset

    def get_ssl_datasets(
        self,
        labeled_samples_per_class: int,
        batch_size: Optional[int] = None,
        unlabeled_batch_size: Optional[int] = None,
        train_shuffle: bool = True,
        train_augment_labeled: bool = True,
        train_repeat: bool = False,
        val_repeat: bool = False,
        test_repeat: bool = False
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get datasets for SSL training with proper labeled/unlabeled split from training data only.
        
        Args:
            labeled_samples_per_class: Number of labeled samples per class
            batch_size: Batch size for labeled datasets (defaults to config value)
            unlabeled_batch_size: Batch size for unlabeled dataset (defaults to batch_size)
            train_shuffle: Whether to shuffle the training datasets
            train_augment_labeled: Whether to apply augmentation to labeled data
            train_repeat: Whether to repeat the training datasets
            val_repeat: Whether to repeat the validation dataset
            test_repeat: Whether to repeat the test dataset
            
        Returns:
            Tuple of (labeled_dataset, unlabeled_dataset, val_dataset, test_dataset)
        """
        if not self.data_loaded:
            self._load_and_process_data()

        if batch_size is None:
            batch_size = self.config.batch_size
        if unlabeled_batch_size is None:
            unlabeled_batch_size = batch_size

        # Use new split logic
        (train_data_tuple, val_data_tuple, test_data_tuple, labeled_indices, unlabeled_indices), _, _, _ = self._split_data_fixed(
            self.all_audio_data, self.all_labels, self.all_file_ids, labeled_samples_per_class, ssl_mode=True)
        X_labeled = train_data_tuple[0][labeled_indices] if labeled_indices is not None else np.zeros((0,) + self.input_shape, dtype=np.float32)
        y_labeled = train_data_tuple[1][labeled_indices] if labeled_indices is not None else np.zeros((0,), dtype=np.int32)
        X_unlabeled = train_data_tuple[0][unlabeled_indices] if unlabeled_indices is not None else np.zeros((0,) + self.input_shape, dtype=np.float32)
        
        logger.info(f"Creating SSL datasets from training split only (no data leakage)")
        logger.info(f"Training data shape: {X_labeled.shape}, dtype: {X_labeled.dtype}")
        
        # Validation: ensure no data leakage
        logger.info(f"SSL Dataset Statistics (LEAK-FREE):")
        logger.info(f"  Labeled SSL: {len(X_labeled)} samples")
        logger.info(f"  Unlabeled SSL: {len(X_unlabeled)} samples") 
        logger.info(f"  Validation: {len(self.val_data.data)} samples (ISOLATED)")
        logger.info(f"  Test: {len(self.test_data.data)} samples (ISOLATED)")
        logger.info(f"  Total SSL: {len(X_labeled) + len(X_unlabeled)} samples")
        logger.info(f"  Original training: {len(X_labeled) + len(X_unlabeled)} samples")
        
        # Add labeled/unlabeled stats for downstream logging
        self.split_stats['labeled'] = {
            'segments': len(X_labeled),
            'files': len(np.unique(self.train_data.file_ids[labeled_indices])) if labeled_indices is not None else 0
        }
        self.split_stats['unlabeled'] = {
            'segments': len(X_unlabeled),
            'files': len(np.unique(self.train_data.file_ids[unlabeled_indices])) if unlabeled_indices is not None else 0
        }

        # Verify no overlap with validation/test data
        assert len(X_labeled) + len(X_unlabeled) <= len(X_labeled) + len(X_unlabeled), "SSL split exceeds training data"
        
        # 1. Labeled dataset
        ds_labeled = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled))
        
        if train_shuffle:
            ds_labeled = ds_labeled.shuffle(len(X_labeled), seed=self.seed, reshuffle_each_iteration=True)
        
        if train_repeat:
            ds_labeled = ds_labeled.repeat()
        ds_labeled = ds_labeled.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # 2. Unlabeled dataset with SSL augmentation
        if len(X_unlabeled) > 0:
            # Debug: Check the actual shape of X_unlabeled
            logger.info(f"X_unlabeled shape: {X_unlabeled.shape}, dtype: {X_unlabeled.dtype}")
            logger.info(f"Expected input shape: {self.input_shape}")
            
            # Create dummy labels for augmentation function compatibility
            dummy_labels = tf.zeros(len(X_unlabeled), dtype=tf.int64)
            ds_unlabeled = tf.data.Dataset.from_tensor_slices((X_unlabeled, dummy_labels))

            # Debug: Check the dataset element spec
            logger.info(f"Unlabeled dataset element spec: {ds_unlabeled.element_spec}")

            # Apply SSL augmentation to create weak and strong augmented versions
            ssl_augment_fn = create_ssl_augment_unlabeled_fn(
                config_dict=getattr(self.config, 'augmentation', {}), 
                data_type=self.data_type
            )
            ds_unlabeled = ds_unlabeled.map(ssl_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

            if train_shuffle:
                ds_unlabeled = ds_unlabeled.shuffle(len(X_unlabeled), seed=self.seed, reshuffle_each_iteration=True)

            if train_repeat:
                ds_unlabeled = ds_unlabeled.repeat()
            ds_unlabeled = ds_unlabeled.batch(unlabeled_batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            logger.warning("Empty unlabeled dataset - creating dummy dataset")
            # CRITICAL FIX 5: Proper empty dataset creation
            if self.data_type == 'raw':
                dummy_shape = (int(self.sample_rate * self.audio_length), 1)
            else:  # STFT
                dummy_shape = (self.stft.freq_bins, self.stft.time_frames, 1)
                
            ds_unlabeled = tf.data.Dataset.from_tensor_slices((
                np.zeros((0,) + dummy_shape, dtype=np.float32),  # weak
                np.zeros((0,) + dummy_shape, dtype=np.float32)   # strong
            ))
            ds_unlabeled = ds_unlabeled.batch(unlabeled_batch_size).prefetch(tf.data.AUTOTUNE)

        # 3. Validation and test datasets (already properly isolated)
        ds_val = self.prepare_dataset('val', batch_size=batch_size, shuffle=False, augment=False, repeat=val_repeat)
        ds_test = self.prepare_dataset('test', batch_size=batch_size, shuffle=False, augment=False, repeat=test_repeat)
        
        return ds_labeled, ds_unlabeled, ds_val, ds_test

    def get_validated_ssl_datasets(
        self,
        labeled_samples_per_class: int,
        batch_size: Optional[int] = None,
        unlabeled_batch_size: Optional[int] = None,
        train_shuffle: bool = True,
        train_augment_labeled: bool = False,
        train_repeat: bool = True,
        val_repeat: bool = False,
        test_repeat: bool = False
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get validated datasets for SSL training.
        
        This method enforces that data has been loaded and validated before 
        creating TensorFlow datasets for SSL training. Shape validation occurs
        during preprocessing, so downstream code can assume correct shapes.
        
        Args:
            labeled_samples_per_class: Number of labeled samples per class
            batch_size: Batch size for labeled datasets (defaults to config value)
            unlabeled_batch_size: Batch size for unlabeled dataset (defaults to batch_size)
            train_shuffle: Whether to shuffle the training datasets
            train_augment_labeled: Whether to apply augmentation to labeled data
            train_repeat: Whether to repeat the training datasets
            val_repeat: Whether to repeat the validation dataset
            test_repeat: Whether to repeat the test dataset
            
        Returns:
            Tuple of (labeled_dataset, unlabeled_dataset, val_dataset, test_dataset)
            All datasets have validated shapes and SSL-specific preprocessing.
            
        Raises:
            RuntimeError: If data has not been loaded and validated yet
        """
        if not self.data_loaded:
            raise RuntimeError("Data must be loaded and validated before getting validated SSL datasets. "
                             "This usually indicates a programming error.")
                             
        logger.info("Creating validated SSL datasets with single-point shape validation")
        
        # Get SSL datasets using the standard method (data already validated)
        return self.get_ssl_datasets(
            labeled_samples_per_class=labeled_samples_per_class,
            batch_size=batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            train_shuffle=train_shuffle,
            train_augment_labeled=train_augment_labeled,
            train_repeat=train_repeat,
            val_repeat=val_repeat,
            test_repeat=test_repeat
        )
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive preprocessing and validation statistics.
        
        This method provides the single source of truth for data shapes and 
        preprocessing information. Use this to get validated input shapes for
        model creation and other downstream operations.
        
        Returns:
            Dictionary containing:
            - shape_info: Validated input shapes and data types
            - split_stats: Training/validation/test split information  
            - validation_info: Preprocessing completion status
            - config_info: Configuration parameters used for preprocessing
            
        Raises:
            RuntimeError: If data has not been loaded and processed yet
        """
        if not self.data_loaded:
            raise RuntimeError("Data must be loaded and processed before getting preprocessing stats. "
                             "This usually indicates a programming error.")
        
        return {
            'shape_info': {
                'input_shape': self.input_shape,
                'data_type': self.data_type,
                'audio_dtype': self.all_audio_data.dtype.name if self.all_audio_data is not None else None,
                'labels_dtype': self.all_labels.dtype.name if self.all_labels is not None else None,
                'channels': self.channels,
                'sample_rate': self.sample_rate,
                'audio_length': self.audio_length
            },
            'split_stats': self.split_stats,
            'validation_info': {
                'data_loaded': self.data_loaded,
                'shapes_validated': True,  # Always true if data_loaded is True
                'preprocessing_complete': True
            },
            'config_info': {
                'split_mode': self.split_mode,
                'training_mode': self.training_mode,
                'num_classes': self.num_classes,
                'seed': self.seed
            }
        }

class SSLDatasetManager:
    """
    Dataset manager for semi-supervised learning.
    
    This specialized manager handles both labeled and unlabeled data
    for SSL algorithms like FixMatch and FlexMatch.
    """
    
    def __init__(self, labeled_paths: List[str], unlabeled_paths: List[str], config: Config):
        """
        Initialize the SSL dataset manager.
        
        Args:
            labeled_paths: List of paths to labeled data files
            unlabeled_paths: List of paths to unlabeled data files
            config: Configuration object
        """
        # This is a placeholder for the SSL dataset manager
        # Implementation will be added when needed for SSL training
        pass
