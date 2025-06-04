"""
Augmentation module for mosquito wingbeat classification.
Contains implementations for both weak and strong augmentation of audio data.
Supports both raw waveform and spectrogram augmentation with vectorized operations.
"""

import logging
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, Callable, List, Protocol, TypedDict

logger = logging.getLogger(__name__)


class AugmentationConfig(TypedDict, total=False):
    """Configuration for audio augmentations with type definitions."""
    # Common parameters
    sample_rate: int
    audio_length: float
    
    # Noise parameters
    noise_level: float
    
    # Time shift parameters
    shift_max_sec: float
    
    # Volume parameters
    volume_min: float
    volume_max: float
    
    # Pitch shift parameters
    enable_pitch_shift: bool
    pitch_shift_min: float
    pitch_shift_max: float
    
    # Time stretch parameters
    enable_time_stretch: bool
    time_stretch_min: float
    time_stretch_max: float
    
    # EQ parameters
    enable_eq: bool
    eq_bands: int
    eq_min_gain: float
    eq_max_gain: float
    
    # Mixup parameters
    enable_mixup: bool
    mixup_alpha: float
    
    # Cutout parameters
    enable_cutout: bool
    cutout_max_width: float
    
    # Dynamic range compression parameters
    enable_compression: bool
    compression_threshold: float
    compression_ratio: float
    
    # RIR convolution parameters
    enable_rir: bool
    
    # CutMix parameters
    enable_cutmix: bool
    cutmix_max_ratio: float
    
    # Random gain parameters
    random_gain: bool
    gain_factor_range: List[float]
    
    # Reverb parameters
    reverb: bool
    reverb_factor: float


@tf.function
def augment_sample(x: tf.Tensor, 
                   y: tf.Tensor, 
                   data_type: tf.Tensor, 
                   aug_config: AugmentationConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Placeholder: No augmentation applied. Returns input as-is.
    """
    return x, y


def get_augmentation_config(config_dict: dict, aug_type: str = 'weak') -> dict:
    """
    Placeholder: Returns empty config. No augmentation.
    """
    return {}


def create_ssl_augment_labeled_fn(config_dict: dict, data_type: str):
    """
    Placeholder: Returns identity function for labeled SSL augmentation.
    """
    def identity(x, y):
        return x, y
    return identity


def create_ssl_augment_unlabeled_fn(config_dict: dict, data_type: str):
    """
    Placeholder: Returns function that returns (x, x), y for unlabeled SSL augmentation.
    Ensures both x are tensors, not objects, to avoid tf.data batching errors.
    Downstream code should unpack as: (x_weak, x_strong), y
    """
    def identity(x, y):
        x1 = tf.identity(x)
        x2 = tf.identity(x)
        return (x1, x2), y
    return identity