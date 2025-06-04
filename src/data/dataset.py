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
        (train_raw, val_raw, test_raw), self.initial_train_indices, self.initial_val_indices, self.initial_test_indices = \
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
    
    def _split_data_fixed(self, audio_data: np.ndarray, labels: np.ndarray, file_ids: np.ndarray) -> Tuple[
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Split data using fixed sample counts per class.
        
        Args:
            audio_data: Audio data array
            labels: Labels array
            file_ids: File IDs array
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # First try to get values from config.dataset, then fall back to config's direct attributes
        if hasattr(self.config, 'dataset'):
            train_samples: int = getattr(self.config.dataset, 'train_samples', 350)
            val_samples: int = getattr(self.config.dataset, 'val_samples', 50)
            test_samples: int = getattr(self.config.dataset, 'test_samples', 50)
        else:
            train_samples: int = self.config.get('train_samples', 350)
            val_samples: int = self.config.get('val_samples', 50)
            test_samples: int = self.config.get('test_samples', 50)
        
        logger.info(f"Using fixed split with {train_samples}/{val_samples}/{test_samples} samples per class")
        
        train_indices_list: List[int] = [] # Use list for extend
        val_indices_list: List[int] = []
        test_indices_list: List[int] = []
        
        # Process each class separately to ensure balanced splits
        for class_idx in range(self.num_classes):
            # Find all indices for this class
            class_indices = np.where(labels == class_idx)[0]
            
            if len(class_indices) == 0:
                logger.warning(f"No samples found for class {class_idx}")
                continue
                
            # Shuffle indices deterministically
            rng = np.random.RandomState(self.seed + class_idx)
            rng.shuffle(class_indices)
            
            # Determine how many samples to take for each split
            n_train = min(train_samples, len(class_indices))
            n_val = min(val_samples, max(0, len(class_indices) - n_train))
            n_test = min(test_samples, max(0, len(class_indices) - n_train - n_val))
            
            # Split indices
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
        logger.info(f"Actual counts: Train={len(final_train_indices)}, Val={len(final_val_indices)}, Test={len(final_test_indices)}")
        
        return (train_data_tuple, val_data_tuple, test_data_tuple), final_train_indices, final_val_indices, final_test_indices
        
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
        
        # Apply augmentation if requested (only for training)
        if augment and split == 'train':
            # Directly use augmentation functions from the augmentation module
            aug_config_weak = get_augmentation_config(self.config.to_dict(), 'weak')
            dataset = dataset.map(
                lambda x_data, y_data: augment_sample(x_data, y_data, self.data_type, aug_config_weak), 
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
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
        test_repeat: bool = False
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get train, validation, and test datasets in a single call.
        
        Args:
            batch_size: Batch size for all datasets (defaults to config value)
            train_shuffle: Whether to shuffle the training dataset
            train_augment: Whether to apply augmentation to the training dataset
            train_repeat: Whether to repeat the training dataset
            val_repeat: Whether to repeat the validation dataset
            test_repeat: Whether to repeat the test dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = self.prepare_dataset(
            split='train',
            batch_size=batch_size,
            shuffle=train_shuffle,
            augment=train_augment,
            repeat=train_repeat
        )
        
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
        Get datasets for semi-supervised learning.

        Splits the training data into labeled and unlabeled sets.
        The unlabeled dataset yields (weak_augment, strong_augment) tuples.

        Args:
            labeled_samples_per_class: Number of labeled samples per class.
            batch_size: Batch size for all datasets. Defaults to config.batch_size.
            train_shuffle: Whether to shuffle labeled and unlabeled training datasets.
            train_augment_labeled: Whether to apply weak augmentation to the labeled training dataset.
            train_repeat: Whether to repeat the training datasets.
            val_repeat: Whether to repeat the validation dataset.
            test_repeat: Whether to repeat the test dataset.

        Returns:
            Tuple of (labeled_train_dataset, unlabeled_train_dataset, val_dataset, test_dataset)
        """
        # Validate that we're using the appropriate split mode for SSL
        if self.split_mode != 'fixed':
            logger.warning(f"SSL dataset requested but split_mode is '{self.split_mode}' instead of 'fixed'. "
                          f"This may lead to inconsistent labeled data pools across runs.")
            
        if not self.data_loaded:
            self._load_and_process_data()

        if batch_size is None: # Add batch_size to local scope if not passed
            batch_size = self.config.batch_size
        if unlabeled_batch_size is None:
            unlabeled_batch_size = batch_size

        # 1. Determine SSL Labeled Set (from the initial training pool)
        # self.train_data was created using self.initial_train_indices from self.all_audio_data
        X_initial_train, y_initial_train, _ = self.train_data.data, self.train_data.labels, self.train_data.file_ids
        
        rng_ssl_split = np.random.default_rng(self.seed) 
        
        ssl_labeled_indices_relative_to_initial_train_list = []
        
        unique_classes_in_train = np.unique(y_initial_train)
        for cls in unique_classes_in_train:
            cls_indices_in_initial_train = np.where(y_initial_train == cls)[0]
            rng_ssl_split.shuffle(cls_indices_in_initial_train)
            
            actual_labeled_count = min(labeled_samples_per_class, len(cls_indices_in_initial_train))
            if actual_labeled_count < labeled_samples_per_class and len(cls_indices_in_initial_train) > 0 : # Check if any samples for this class
                 logger.warning(f"Class {cls}: Requested {labeled_samples_per_class} labeled samples, but only {len(cls_indices_in_initial_train)} available in initial train pool. Using {actual_labeled_count}.")

            ssl_labeled_indices_relative_to_initial_train_list.extend(cls_indices_in_initial_train[:actual_labeled_count])

        ssl_labeled_indices_final = np.array(ssl_labeled_indices_relative_to_initial_train_list, dtype=np.int32)
        rng_ssl_split.shuffle(ssl_labeled_indices_final) 

        X_labeled = X_initial_train[ssl_labeled_indices_final]
        y_labeled = y_initial_train[ssl_labeled_indices_final]

        # 2. Determine SSL Unlabeled Set (from the overall remainder)
        if self.all_audio_data is None or \
           self.initial_train_indices is None or \
           self.initial_val_indices is None or \
           self.initial_test_indices is None:
            # This should not happen if _load_and_process_data was called
            logger.error("Full dataset or initial split indices are not available. Unlabeled set cannot be determined correctly.")
            # Fallback to empty unlabeled set or raise error
            X_unlabeled = np.array([]) 
            # y_unlabeled_for_stats = np.array([]) # if used later
            # Consider raising an error here:
            raise RuntimeError("Initial data splits and indices were not properly prepared for SSL unlabeled set construction.")

        # This is a key improvement: also use remaining training data as unlabeled
        # 1. Get the remaining training indices (those not used for labeled set)
        unused_train_indices_relative = np.setdiff1d(
            np.arange(len(X_initial_train)), 
            ssl_labeled_indices_final, 
            assume_unique=True
        )
        # Convert to absolute indices within all_audio_data
        unused_train_indices_absolute = self.initial_train_indices[unused_train_indices_relative]
        
        # 2. Also get indices outside the initial splits to use as unlabeled
        all_data_indices = np.arange(len(self.all_audio_data))
        indices_in_initial_splits = np.concatenate([
            self.initial_train_indices, 
            self.initial_val_indices, 
            self.initial_test_indices
        ])
        indices_in_initial_splits_unique = np.unique(indices_in_initial_splits)
        external_unlabeled_pool_indices = np.setdiff1d(all_data_indices, indices_in_initial_splits_unique, assume_unique=True)
        
        # 3. Combine both sources of unlabeled data
        ssl_unlabeled_pool_indices = np.concatenate([unused_train_indices_absolute, external_unlabeled_pool_indices])
        rng_ssl_split.shuffle(ssl_unlabeled_pool_indices) # Shuffle the combined unlabeled pool

        X_unlabeled = self.all_audio_data[ssl_unlabeled_pool_indices]
        # Optional: y_unlabeled_for_stats = self.all_labels[ssl_unlabeled_pool_indices]
        
        # If data_type is 'stft', ensure X_unlabeled is also STFT features
        if self.data_type == 'stft':
            logger.info("Converting SSL unlabeled pool to STFT features...")
            X_unlabeled = waveform_to_fft(X_unlabeled, config=self.config)


        # Update split_stats
        self.split_stats['labeled'] = {'segments': len(X_labeled), 'files': 'N/A (derived from train)'}
        
        # Calculate and log the composition of the unlabeled set
        unused_train_count = len(unused_train_indices_absolute) if 'unused_train_indices_absolute' in locals() else 0
        external_data_count = len(external_unlabeled_pool_indices) if 'external_unlabeled_pool_indices' in locals() else 0
        total_unlabeled = len(X_unlabeled)
        
        self.split_stats['unlabeled'] = {
            'segments': total_unlabeled,
            'from_train': unused_train_count,
            'from_external': external_data_count,
            'files': 'N/A (derived from multiple sources)'
        }
        
        logger.info(f"SSL split statistics:")
        logger.info(f"  Labeled train = {len(X_labeled)} samples")
        logger.info(f"  Unlabeled train = {total_unlabeled} samples")
        logger.info(f"    - {unused_train_count} samples from unused training data ({unused_train_count/total_unlabeled*100:.1f}%)")
        logger.info(f"    - {external_data_count} samples from external data ({external_data_count/total_unlabeled*100:.1f}%)")
        logger.info(f"  Validation = {len(self.val_data.data)} samples")
        logger.info(f"  Test = {len(self.test_data.data)} samples")

        # ... (rest of the method: create ds_labeled, ds_unlabeled, ds_val, ds_test from X_labeled, y_labeled, X_unlabeled)
        # Ensure ds_val and ds_test are correctly prepared using self.val_data and self.test_data
        # which were created from the initial splits.

        # 1. Labeled training dataset (from X_labeled, y_labeled)
        ds_labeled = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled))
        
        if train_shuffle: 
            ds_labeled = ds_labeled.shuffle(buffer_size=len(X_labeled), seed=self.seed, reshuffle_each_iteration=True)
        
        if train_augment_labeled: 
            # Use the new centralized augmentation function
            augment_labeled_fn = create_ssl_augment_labeled_fn(self.config.to_dict(), self.data_type)
            ds_labeled = ds_labeled.map(augment_labeled_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if train_repeat: 
            ds_labeled = ds_labeled.repeat()
        ds_labeled = ds_labeled.batch(batch_size).prefetch(tf.data.AUTOTUNE)


        # 2. Unlabeled training dataset (from X_unlabeled)
        if len(X_unlabeled) > 0:
            # For unlabeled data, we only need the features (X_unlabeled)
            # The labels are placeholders and not used by the augmentation functions for unlabeled data.
            # We create dummy labels here to match the expected input signature of from_tensor_slices
            # if we were to pass (features, labels). However, it's cleaner to just pass features
            # and then adapt the mapping function if it expects two arguments.
            # The create_ssl_augment_unlabeled_fn expects (x, y) but ignores y for unlabeled.
            
            # Create dummy labels for the unlabeled dataset to match the structure expected by the map function
            # if we were to pass (features, labels). However, it's cleaner to just pass features
            # and then adapt the mapping function if it expects two arguments.
            # The create_ssl_augment_unlabeled_fn expects (x, y) but ignores y for unlabeled.
            
            # Create dummy labels for the unlabeled dataset to match the structure expected by the map function
            dummy_labels_unlabeled = tf.zeros(len(X_unlabeled), dtype=tf.int64)
            ds_unlabeled = tf.data.Dataset.from_tensor_slices((X_unlabeled, dummy_labels_unlabeled))

            if train_shuffle:
                ds_unlabeled = ds_unlabeled.shuffle(buffer_size=len(X_unlabeled), seed=self.seed, reshuffle_each_iteration=True)

            # Use the new centralized augmentation function
            augment_unlabeled_fn = create_ssl_augment_unlabeled_fn(self.config.to_dict(), self.data_type)
            ds_unlabeled = ds_unlabeled.map(augment_unlabeled_fn, num_parallel_calls=tf.data.AUTOTUNE)
            
            # The output of augment_unlabeled_fn is ((x_weak, x_strong), y_placeholder).
            # For training, we typically only need the augmented views (x_weak, x_strong).
            # So, we map again to discard the placeholder y.
            ds_unlabeled = ds_unlabeled.map(lambda x_augs, y_placeholder: x_augs, num_parallel_calls=tf.data.AUTOTUNE)


            if train_repeat:
                ds_unlabeled = ds_unlabeled.repeat()
            ds_unlabeled = ds_unlabeled.batch(unlabeled_batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            logger.warning("Unlabeled dataset is empty. Creating an empty tf.data.Dataset for unlabeled data.")
            # Create an empty dataset with the expected structure (weak_aug, strong_aug)
            # Determine the expected shape from config or a sample transformation
            # This is a simplified placeholder; a more robust way would be to get shape from a dummy sample
            dummy_shape = self.input_shape 
            if self.data_type == 'raw' and len(dummy_shape) == 1: # if raw (samples,) add channel
                 dummy_shape = dummy_shape + (1,)

            def create_empty_unlabeled_dataset():
                return tf.data.Dataset.from_tensor_slices((
                    np.zeros((0,) + dummy_shape, dtype=np.float32), # weak_aug
                    np.zeros((0,) + dummy_shape, dtype=np.float32)  # strong_aug
                ))
            ds_unlabeled = create_empty_unlabeled_dataset()
            ds_unlabeled = ds_unlabeled.batch(batch_size).prefetch(tf.data.AUTOTUNE)


        # 3. Validation dataset (using self.val_data)
        # val_repeat is an argument to get_ssl_datasets
        ds_val = self.prepare_dataset(
            split='val', batch_size=batch_size, shuffle=False, augment=False, repeat=val_repeat
        )

        # 4. Test dataset (using self.test_data)
        # test_repeat is an argument to get_ssl_datasets
        ds_test = self.prepare_dataset(
            split='test', batch_size=batch_size, shuffle=False, augment=False, repeat=test_repeat
        )
        
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
            raise RuntimeError("Data must be loaded before getting validated SSL datasets. "
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

    # ...existing code...


# For SSL training, we can implement a specific SSL dataset manager when needed
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
