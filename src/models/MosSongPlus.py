"""
MosSong+ Model for mosquito wingbeat classification.

This module defines the MosSong+ model architecture, supporting both
raw waveform and Short-Time Fourier Transform (STFT) spectrogram inputs.
It provides factory functions to create model instances based on
configuration settings.
"""
import logging
import sys
from pathlib import Path
from typing import Tuple, Union # Removed Optional, Dict as they are not used in the current context of this file

import tensorflow as tf
from tensorflow import keras

# Add the project root to the Python path to ensure robust local module imports.
# This allows modules within the 'src' directory to be imported as if 'src' were
# a top-level package, simplifying import statements across the project.
# Example: `from src.utils.config import Config`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config # pylint: disable=import-error,wrong-import-position

# Configure a dedicated logger for this module. This helps in isolating
# log messages and managing log levels specifically for model creation activities.
logger = logging.getLogger(__name__)

# Default parameters for model creation when specific configurations are not provided.
# These serve as fallbacks and ensure the model can be instantiated with sensible defaults.
DEFAULT_NUM_CLASSES: int = 11
DEFAULT_AUDIO_LENGTH_SEC: float = 0.3  # Audio length in seconds
DEFAULT_SAMPLE_RATE_HZ: int = 8000  # Sample rate in Hertz
# Input shape for raw waveform: (time_steps, channels)
# Calculated from default audio length and sample rate.
DEFAULT_WAVEFORM_INPUT_SHAPE: Tuple[int, int] = (
    int(DEFAULT_SAMPLE_RATE_HZ * DEFAULT_AUDIO_LENGTH_SEC), 1
)
# Default input shape for STFT spectrograms: (frequency_bins, time_frames, channels)
# This example is based on n_fft=512, resulting in 257 frequency bins.
DEFAULT_STFT_INPUT_SHAPE: Tuple[int, int, int] = (257, 10, 1)


def create_model_from_config(config: Config) -> keras.Model:
    """
    Factory function to create a Keras model based on the provided configuration.

    This function inspects the `data_type` specified in the configuration
    and dispatches to the appropriate model creation function (either for
    raw waveforms or STFT spectrograms). It ensures that the model's input
    shape and number of output classes are consistent with the configuration.

    Args:
        config: A `Config` object containing all necessary parameters,
                including data type, audio settings, STFT settings,
                and dataset-specific information like the number of classes.

    Returns:
        A compiled Keras model, ready for training or inference.

    Raises:
        ValueError: If the `data_type` in the configuration is not
                    one of 'raw' or 'stft'.
    """
    data_type: str = config.data_type
    num_classes: int = config.dataset.num_classes

    logger.info(f"Attempting to create model for data_type: '{data_type}' with {num_classes} classes.")

    if data_type == 'raw':
        # For raw audio data, the input shape is determined by the sample rate
        # and audio length.
        sample_rate: int = config.audio.sample_rate
        audio_length: float = config.audio.audio_length
        # Input shape is (time_steps, 1) for mono audio.
        input_shape_raw: Tuple[int, int] = (int(sample_rate * audio_length), 1)
        
        logger.info(f"Creating raw waveform model with input shape {input_shape_raw}.")
        return create_waveform_model(input_shape=input_shape_raw, num_classes=num_classes)
        
    elif data_type == 'stft':
        # For STFT data, the input shape is (frequency_bins, time_frames, channels).
        # The number of channels is typically 1 for STFT magnitude spectrograms.
        freq_bins: int = config.stft.freq_bins
        time_frames: int = config.stft.time_frames
        input_shape_stft: Tuple[int, int, int] = (freq_bins, time_frames, 1)
        
        logger.info(f"Creating STFT/FFT model with input shape {input_shape_stft}.")
        return create_fft_model(input_shape=input_shape_stft, num_classes=num_classes)
        
    else:
        # Defensive programming: ensure only supported data types are processed.
        error_msg = f"Unsupported data_type in configuration: '{data_type}'. Must be 'raw' or 'stft'."
        logger.error(error_msg)
        raise ValueError(error_msg)


def create_waveform_model(
    input_shape: Tuple[int, int] = DEFAULT_WAVEFORM_INPUT_SHAPE, 
    num_classes: int = DEFAULT_NUM_CLASSES
) -> keras.Model:
    """
    Creates a Keras Sequential model tailored for raw audio waveform input.

    The architecture consists of several 1D convolutional layers for feature
    extraction, followed by max pooling, flattening, dropout for regularization,
    and dense layers for classification.

    Args:
        input_shape: A tuple representing the input shape of the waveform
                     (time_steps, channels). Defaults to `DEFAULT_WAVEFORM_INPUT_SHAPE`.
        num_classes: The number of output classes for the final dense layer.
                     Defaults to `DEFAULT_NUM_CLASSES`.

    Returns:
        A Keras Sequential model designed for 1D audio signals.
    """
    logger.debug(f"Initializing waveform model with input_shape={input_shape}, num_classes={num_classes}")
    
    # Input layer expects a shape of (time_steps, channels).
    # For raw audio, channels is typically 1 (mono).
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape, name="waveform_input"),
        # Convolutional layers for feature extraction from raw audio.
        # Parameters (filters, kernel_size, strides) are chosen based on common
        # practices for audio processing. 'padding=same' ensures output feature
        # maps can be downsampled without losing information at the edges.
        keras.layers.Conv1D(filters=32, kernel_size=100, strides=4, activation='relu', padding='same', name="conv1d_1"),
        keras.layers.Conv1D(filters=32, kernel_size=64, strides=4, activation='relu', padding='same', name="conv1d_2"),
        keras.layers.Conv1D(filters=64, kernel_size=64, strides=3, activation='relu', padding='same', name="conv1d_3"),
        # MaxPooling reduces dimensionality and provides a degree of translation invariance.
        keras.layers.MaxPooling1D(pool_size=3, name="maxpool1d"),
        # Flatten converts the 2D feature maps into a 1D vector for the dense layers.
        keras.layers.Flatten(name="flatten"),
        # Dropout is a regularization technique to prevent overfitting.
        keras.layers.Dropout(rate=0.5, name="dropout"),
        # Dense layers for classification.
        keras.layers.Dense(256, activation='relu', name="dense_1"),
        keras.layers.Dense(128, activation='relu', name="dense_2"),
        # Output layer with 'num_classes' units. No activation is specified here,
        # implying a linear activation, suitable for use with losses like
        # CategoricalCrossentropy(from_logits=True).
        keras.layers.Dense(num_classes, name="output_logits")
    ])
    
    logger.info("Waveform model created successfully.")
    return model


def create_fft_model(
    input_shape: Tuple[int, int, int] = DEFAULT_STFT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES
) -> keras.Model:
    """
    Creates a Keras Sequential model tailored for STFT spectrogram input.

    The architecture uses 2D convolutional layers suitable for processing
    spectrograms (image-like data), followed by pooling, flattening,
    dropout, and dense layers for classification.

    Args:
        input_shape: A tuple representing the input shape of the spectrogram
                     (frequency_bins, time_frames, channels).
                     Defaults to `DEFAULT_STFT_INPUT_SHAPE`.
        num_classes: The number of output classes for the final dense layer.
                     Defaults to `DEFAULT_NUM_CLASSES`.

    Returns:
        A Keras Sequential model designed for 2D STFT spectrograms.
    """
    logger.debug(f"Initializing FFT model with input_shape={input_shape}, num_classes={num_classes}")
    
    # Input layer expects a shape of (frequency_bins, time_frames, channels).
    # For STFT spectrograms, channels is typically 1.
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape, name="stft_input"),
        # Convolutional layers for feature extraction from spectrograms.
        # Parameters (filters, kernel_size, strides) are chosen based on common
        # practices for image processing. 'padding=same' ensures output feature
        # maps can be downsampled without losing information at the edges.
        keras.layers.Conv2D(32, kernel_size=(10, 10), strides=(4, 1), activation='relu', padding='same', name="conv2d_1"),
        keras.layers.Conv2D(32, kernel_size=(6, 6), strides=(4, 1), activation='relu', padding='same', name="conv2d_2"),
        keras.layers.Conv2D(64, kernel_size=(6, 6), strides=(3, 1), activation='relu', padding='same', name="conv2d_3"),
        # MaxPooling reduces dimensionality and provides a degree of translation invariance.
        keras.layers.MaxPooling2D(pool_size=(3, 1), name="maxpool2d"),
        # Flatten converts the 2D feature maps into a 1D vector for the dense layers.
        keras.layers.Flatten(name="flatten"),
        # Dropout is a regularization technique to prevent overfitting.
        keras.layers.Dropout(rate=0.5, name="dropout"),
        # Dense layers for classification.
        keras.layers.Dense(256, activation='relu', name="dense_1"),
        keras.layers.Dense(128, activation='relu', name="dense_2"),
        # Output layer with 'num_classes' units. No activation is specified here,
        # implying a linear activation, suitable for use with losses like
        # CategoricalCrossentropy(from_logits=True).
        keras.layers.Dense(num_classes, name="output_logits")
    ])
    
    logger.info("FFT model created successfully.")
    return model


def tfModel(input_shape=DEFAULT_WAVEFORM_INPUT_SHAPE, num_classes=DEFAULT_NUM_CLASSES, input_type='waveform'):
    """
    Legacy interface for compatibility with existing code.
    Create a model for mosquito sound classification.
    
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
            logger.warning(f"Converting input_shape {input_shape} to FFT shape {DEFAULT_STFT_INPUT_SHAPE}")
            input_shape = DEFAULT_STFT_INPUT_SHAPE
        return create_fft_model(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

