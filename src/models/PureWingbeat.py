"""
PureWingbeat Model for mosquito wingbeat classification.
Supports both raw waveform and STFT spectrogram inputs.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

# Add the project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import local modules
from src.utils.config import Config

# Configure module logger
logger = logging.getLogger(__name__)

# Default parameters if config not found
DEFAULT_NUM_CLASSES = 11
DEFAULT_AUDIO_LENGTH = 0.3
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_INPUT_SHAPE = int(DEFAULT_SAMPLE_RATE * DEFAULT_AUDIO_LENGTH)
# Default FFT input shape: [freq_bins, time_frames, channels]
DEFAULT_FFT_INPUT_SHAPE = (257, 10, 1)  # Based on n_fft=512


def create_model_from_config(config: Config):
    """
    Create a PureWingbeat model based on configuration settings.
    
    Args:
        config: Configuration object
        
    Returns:
        TensorFlow model configured according to settings
    """
    # Get model parameters from config
    data_type = config.data_type
    num_classes = config.dataset.num_classes
    
    if data_type == 'raw':
        # Calculate input shape for raw waveform
        sample_rate = config.audio.sample_rate
        audio_length = config.audio.audio_length
        input_shape = int(sample_rate * audio_length)
        
        # Create model for raw waveform
        logger.info(f"Creating PureWingbeat waveform model with input shape {input_shape}")
        return create_waveform_model(input_shape=input_shape, num_classes=num_classes)
        
    elif data_type == 'stft':
        # Calculate input shape for STFT features
        freq_bins = config.stft.freq_bins
        time_frames = config.stft.time_frames
        input_shape = (freq_bins, time_frames, 1)
        
        # Create model for STFT features
        logger.info(f"Creating PureWingbeat FFT model with input shape {input_shape}")
        return create_fft_model(input_shape=input_shape, num_classes=num_classes)
        
    else:
        raise ValueError(f"Unsupported data type in config: {data_type}")


def create_waveform_model(input_shape: int = DEFAULT_INPUT_SHAPE, 
                          num_classes: int = DEFAULT_NUM_CLASSES):
    """
    Create a PureWingbeat model for raw waveform input.
    
    Args:
        input_shape: Number of input samples (time dimension)
        num_classes: Number of output classes
        
    Returns:
        TensorFlow model for waveform input
    """
    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Reshape((input_shape, 1)),  # Add channel dimension
        keras.layers.Conv1D(filters=32, kernel_size=350, strides=4, activation='relu'),
        keras.layers.MaxPool1D(pool_size=4),
        keras.layers.Conv1D(filters=32, kernel_size=100, strides=1, activation='relu'),
        keras.layers.MaxPool1D(pool_size=3),
        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])
    
    return model


def create_fft_model(input_shape: Tuple[int, int, int] = DEFAULT_FFT_INPUT_SHAPE,
                     num_classes: int = DEFAULT_NUM_CLASSES):
    """
    Create a PureWingbeat model for FFT spectrogram input.
    
    Args:
        input_shape: Shape of input spectrograms (freq_bins, time_frames, channels)
        num_classes: Number of output classes
        
    Returns:
        TensorFlow model for FFT input
    """
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(35, 3), strides=(4, 1), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(4, 1)),
        keras.layers.Conv2D(32, kernel_size=(10, 3), strides=(1, 1), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(3, 1)),
        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])
    
    return model


def pureWingbeatModel(input_shape=DEFAULT_INPUT_SHAPE, num_classes=DEFAULT_NUM_CLASSES, input_type='waveform'):
    """
    Legacy interface for compatibility with existing code.
    Create a PureWingbeat model for mosquito sound classification.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        input_type: Type of input ('waveform' or 'fft')
        
    Returns:
        TensorFlow model
    """
    if input_type == 'waveform':
        return create_waveform_model(input_shape=input_shape, num_classes=num_classes)
    elif input_type == 'fft':
        # Ensure input_shape is a tuple for FFT
        if isinstance(input_shape, int):
            logger.warning(f"Converting input_shape {input_shape} to FFT shape {DEFAULT_FFT_INPUT_SHAPE}")
            input_shape = DEFAULT_FFT_INPUT_SHAPE
        return create_fft_model(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")