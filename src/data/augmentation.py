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
    Apply data augmentation to a single sample based on data type.
    
    Args:
        x: Input features (as tensor) [samples] or [samples, channels]
        y: Label (as tensor)
        data_type: Type of data ('raw' or 'stft')
        aug_config: Augmentation configuration dictionary
        
    Returns:
        Tuple of (augmented_features, label)
    """
    # Cast to appropriate types (shapes already validated)
    x = tf.cast(x, tf.float32)
    
    # Apply augmentation based on data type
    if data_type == 'raw':
        return augment_waveform(x, y, aug_config)
    else:
        return augment_spectrogram(x, y, aug_config)


def _random_crop(waveform: tf.Tensor, target_len: int) -> tf.Tensor:
    """
    Randomly crop or pad a 1D waveform tensor to target length.
    
    Args:
        waveform: Input waveform [samples]
        target_len: Target length in samples
        
    Returns:
        Cropped or padded waveform [target_len]
    """
    length = tf.shape(waveform)[0]
    # Crop when longer than target
    def crop():
        start = tf.random.uniform([], 0, length - target_len + 1, dtype=tf.int32)
        return waveform[start:start + target_len]
    # Pad with zeros when shorter
    def pad():
        pad_amt = target_len - length
        return tf.pad(waveform, [[0, pad_amt]], mode='CONSTANT')
    return tf.cond(length > target_len, crop, pad)


@tf.function
def _apply_pitch_shift(waveform: tf.Tensor, 
                       aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply pitch shift using phase vocoder approach.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with pitch shift parameters
        
    Returns:
        Pitch-shifted waveform [samples]
    """
    # Only apply if enabled with probability 0.5
    if not aug_config.get('enable_pitch_shift', False):
        return waveform
    
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Convert to spectrogram
    sr = aug_config.get('sample_rate', 8000)
    n_fft = 512
    
    # Use STFT for pitch shifting
    stft = tf.signal.stft(waveform, 
                          frame_length=n_fft, 
                          frame_step=n_fft//4, 
                          window_fn=tf.signal.hann_window)
    
    # Random pitch shift factor
    shift_min = aug_config.get('pitch_shift_min', 0.98)  # ~-2% pitch
    shift_max = aug_config.get('pitch_shift_max', 1.02)  # ~+2% pitch
    shift_factor = tf.random.uniform([], shift_min, shift_max)
    
    # Apply the shift by modifying the frequency bins
    phase_advance = tf.linspace(0.0, np.pi * shift_factor, tf.shape(stft)[0])
    phase_advance = tf.cast(phase_advance, tf.complex64)
    phase_shift = tf.exp(tf.complex(0.0, 1.0) * phase_advance)
    
    # Reshape for broadcasting
    phase_shift = tf.reshape(phase_shift, [-1, 1])
    
    # Apply the phase shift
    stft_shifted = stft * phase_shift
    
    # Convert back to time domain
    pitched_waveform = tf.signal.inverse_stft(
        stft_shifted,
        frame_length=n_fft,
        frame_step=n_fft//4,
        window_fn=tf.signal.inverse_stft_window_fn(n_fft//4, tf.signal.hann_window)
    )
    
    # Ensure output is same length as input
    pitched_waveform = _random_crop(pitched_waveform, tf.shape(waveform)[0])
    
    return pitched_waveform


@tf.function
def _apply_time_stretch(waveform: tf.Tensor, 
                        aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply time stretching using phase vocoder approach.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with time stretch parameters
        
    Returns:
        Time-stretched waveform [samples]
    """
    # Only apply if enabled with probability 0.5
    if not aug_config.get('enable_time_stretch', False):
        return waveform
    
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Random stretch factor
    stretch_min = aug_config.get('time_stretch_min', 0.9)
    stretch_max = aug_config.get('time_stretch_max', 1.1)
    stretch_factor = tf.random.uniform([], stretch_min, stretch_max)
    
    # Convert to spectrogram
    n_fft = 512
    stft = tf.signal.stft(waveform, 
                          frame_length=n_fft, 
                          frame_step=n_fft//4, 
                          window_fn=tf.signal.hann_window)
    
    # Calculate new frame count
    orig_frames = tf.shape(stft)[0]
    new_frames = tf.cast(tf.cast(orig_frames, tf.float32) / stretch_factor, tf.int32)
    
    # Resample time dimension using linear interpolation
    stft_stretched = tf.image.resize(
        tf.abs(stft)[..., tf.newaxis], 
        [new_frames, tf.shape(stft)[1]], 
        method='bilinear'
    )[..., 0]
    
    # Apply phase reconstruction (simplified)
    phase = tf.random.normal(tf.shape(stft_stretched), stddev=np.pi/2)
    stft_complex = tf.complex(stft_stretched, 0.0) * tf.exp(tf.complex(0.0, 1.0) * phase)
    
    # Convert back to time domain
    stretched_waveform = tf.signal.inverse_stft(
        stft_complex,
        frame_length=n_fft,
        frame_step=n_fft//4,
        window_fn=tf.signal.inverse_stft_window_fn(n_fft//4, tf.signal.hann_window)
    )
    
    # Ensure output is same length as input
    stretched_waveform = _random_crop(stretched_waveform, tf.shape(waveform)[0])
    
    return stretched_waveform


@tf.function
def _apply_eq(waveform: tf.Tensor, 
             aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply random equalization by boosting/cutting frequency bands.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with EQ parameters
        
    Returns:
        Equalized waveform [samples]
    """
    # Only apply if enabled with probability 0.5
    if not aug_config.get('enable_eq', False):
        return waveform
    
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Convert to spectrogram
    n_fft = 512
    stft = tf.signal.stft(waveform, 
                          frame_length=n_fft, 
                          frame_step=n_fft//4, 
                          window_fn=tf.signal.hann_window)
    
    # Generate random EQ gains
    n_bands = aug_config.get('eq_bands', 4)
    min_gain = aug_config.get('eq_min_gain', 0.7)  # -3dB
    max_gain = aug_config.get('eq_max_gain', 1.4)  # +3dB
    
    # Create bands with random gains
    n_freqs = tf.shape(stft)[1]
    eq_gains = tf.random.uniform([n_bands], min_gain, max_gain)
    
    # Create smooth transitions between bands
    band_indices = tf.linspace(0.0, tf.cast(n_freqs, tf.float32), n_bands+1)
    band_indices = tf.cast(band_indices, tf.int32)
    
    # Apply the EQ - create a mask for each band and apply the gain
    eq_mask = tf.ones([n_freqs], dtype=tf.float32)
    
    for i in range(n_bands):
        start_idx = band_indices[i]
        end_idx = band_indices[i+1]
        band_mask = tf.cast(
            tf.logical_and(
                tf.range(n_freqs) >= start_idx,
                tf.range(n_freqs) < end_idx
            ), 
            tf.float32
        )
        eq_mask = eq_mask * (1.0 - band_mask) + band_mask * eq_gains[i]
    
    # Reshape for broadcasting
    eq_mask = tf.reshape(eq_mask, [1, -1])
    
    # Apply EQ to magnitude
    stft_mag = tf.abs(stft) * eq_mask
    stft_phase = tf.math.angle(stft)
    stft_eq = tf.complex(
        stft_mag * tf.cos(stft_phase),
        stft_mag * tf.sin(stft_phase)
    )
    
    # Convert back to time domain
    eq_waveform = tf.signal.inverse_stft(
        stft_eq,
        frame_length=n_fft,
        frame_step=n_fft//4,
        window_fn=tf.signal.inverse_stft_window_fn(n_fft//4, tf.signal.hann_window)
    )
    
    # Ensure output is same length as input
    eq_waveform = _random_crop(eq_waveform, tf.shape(waveform)[0])
    
    return eq_waveform


@tf.function
def _apply_cutout(waveform: tf.Tensor, 
                 aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply cutout by zeroing random sections of the waveform.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with cutout parameters
        
    Returns:
        Waveform with random sections zeroed out [samples]
    """
    # Only apply if enabled with probability 0.5
    if not aug_config.get('enable_cutout', False):
        return waveform
    
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Calculate cutout width in samples
    max_width_ratio = aug_config.get('cutout_max_width', 0.1)  # 10% of audio
    max_width = tf.cast(tf.shape(waveform)[0] * max_width_ratio, tf.int32)
    width = tf.random.uniform([], 1, max_width, dtype=tf.int32)
    
    # Random starting point
    start = tf.random.uniform(
        [], 
        0, 
        tf.shape(waveform)[0] - width, 
        dtype=tf.int32
    )
    
    # Create mask
    mask = tf.concat([
        tf.ones(start),
        tf.zeros(width),
        tf.ones(tf.shape(waveform)[0] - start - width)
    ], axis=0)
    
    # Apply mask
    return waveform * mask


@tf.function
def _apply_dynamic_compression(waveform: tf.Tensor, 
                              aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply dynamic range compression to the waveform.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with compression parameters
        
    Returns:
        Compressed waveform [samples]
    """
    # Only apply if enabled with probability 0.5
    if not aug_config.get('enable_compression', False):
        return waveform
    
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Get compression parameters
    threshold = aug_config.get('compression_threshold', -20.0)
    ratio = aug_config.get('compression_ratio', 4.0)
    
    # Compute signal envelope using peak detection
    abs_signal = tf.abs(waveform)
    
    # Use local maximum as envelope
    window_size = 101  # Samples (odd number)
    padded = tf.pad(abs_signal, [[window_size//2, window_size//2]])
    windows = tf.signal.frame(padded, window_size, 1)
    envelope = tf.reduce_max(windows, axis=1)
    
    # Add small value to avoid division by zero or log(0)
    envelope = tf.maximum(envelope, 1e-10)
    
    # Convert to dB
    envelope_db = 20.0 * tf.math.log(envelope) / tf.math.log(10.0)
    
    # Compute gain reduction
    above_threshold = envelope_db - threshold
    above_threshold = tf.maximum(above_threshold, 0.0)
    gain_reduction_db = above_threshold * (1.0 - 1.0/ratio)
    
    # Convert back to linear gain
    gain_reduction = tf.pow(10.0, -gain_reduction_db / 20.0)
    
    # Apply gain reduction
    compressed = waveform * gain_reduction
    
    # Normalize output level
    input_rms = tf.sqrt(tf.reduce_mean(tf.square(waveform)))
    output_rms = tf.sqrt(tf.reduce_mean(tf.square(compressed)))
    gain = input_rms / (output_rms + 1e-10)
    
    return compressed * gain


@tf.function
def _apply_rir_convolution(waveform: tf.Tensor, 
                          aug_config: AugmentationConfig,
                          rir_bank: tf.Tensor) -> tf.Tensor:
    """
    Apply room impulse response convolution.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with RIR parameters
        rir_bank: Bank of room impulse responses [num_irs, ir_length]
        
    Returns:
        Convolved waveform [samples]
    """
    # Only apply if enabled with probability 0.5
    if not aug_config.get('enable_rir', False):
        return waveform
    
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Select random impulse response
    ir_idx = tf.random.uniform([], 0, tf.shape(rir_bank)[0], dtype=tf.int32)
    ir = rir_bank[ir_idx]
    
    # Normalize IR
    ir = ir / tf.sqrt(tf.reduce_sum(tf.square(ir)) + 1e-10)
    
    # Apply convolution
    padded_waveform = tf.pad(waveform, [[tf.shape(ir)[0] - 1, 0]])
    convolved = tf.nn.conv1d(
        tf.expand_dims(padded_waveform, axis=0),
        tf.expand_dims(tf.expand_dims(ir, axis=1), axis=0),
        stride=1,
        padding="VALID"
    )
    
    # Reshape back to 1D
    convolved = tf.squeeze(convolved, axis=[0, 2])
    
    # Ensure output is same length as input
    convolved = convolved[:tf.shape(waveform)[0]]
    
    return convolved


@tf.function
def _apply_random_gain(waveform: tf.Tensor, 
                      aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply random gain adjustment to waveform.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with gain parameters
        
    Returns:
        Waveform with gain adjustment
    """
    # Only apply if enabled
    if not aug_config.get('random_gain', False):
        return waveform
    
    # Apply with probability 0.7
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.7:
        return waveform
    
    # Get gain range
    gain_min = aug_config.get('gain_factor_range', [0.6, 1.4])[0]
    gain_max = aug_config.get('gain_factor_range', [0.6, 1.4])[1]
    
    # Apply random gain
    gain = tf.random.uniform([], gain_min, gain_max)
    return waveform * gain


@tf.function
def _apply_reverb(waveform: tf.Tensor,
                 aug_config: AugmentationConfig) -> tf.Tensor:
    """
    Apply simple synthetic reverb effect to waveform.
    
    Args:
        waveform: Input waveform [samples]
        aug_config: Augmentation configuration with reverb parameters
        
    Returns:
        Waveform with reverb effect
    """
    # Only apply if enabled
    if not aug_config.get('reverb', False):
        return waveform
    
    # Apply with probability 0.5
    apply_prob = tf.random.uniform([], 0.0, 1.0)
    if apply_prob > 0.5:
        return waveform
    
    # Get reverb factor (controls wet/dry mix)
    reverb_factor = aug_config.get('reverb_factor', 0.3)
    
    # Create simple impulse response
    ir_size = 2000  # Maximum IR length in samples
    decay = tf.linspace(1.0, 0.001, ir_size)
    impulse = tf.random.normal([ir_size], stddev=0.01) * decay
    
    # Apply reverb using convolution
    reverb_signal = tf.nn.conv1d(
        tf.expand_dims(waveform, 0),
        tf.expand_dims(tf.expand_dims(impulse, 0), 2),
        stride=1,
        padding='SAME'
    )[0, :, 0]
    
    # Mix dry and wet signals
    return (1.0 - reverb_factor) * waveform + reverb_factor * reverb_signal


@tf.function
def augment_waveform(waveform: tf.Tensor, 
                     label: tf.Tensor, 
                     aug_config: AugmentationConfig,
                     x_pool: Optional[tf.Tensor] = None,
                     y_pool: Optional[tf.Tensor] = None,
                     rir_bank: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply augmentation to raw waveform data.
    
    Args:
        waveform: Input waveform [samples] or [samples, channels]
        label: Input label
        aug_config: Augmentation configuration dictionary
        x_pool: Optional pool of samples for mixup/cutmix [batch, samples]
        y_pool: Optional pool of labels for mixup/cutmix [batch, classes]
        rir_bank: Optional bank of room impulse responses [num_irs, ir_length]
        
    Returns:
        Tuple of (augmented_waveform, label)
    """
    waveform = tf.cast(waveform, tf.float32)
    
    # Flatten any trailing channel dimension to a 1D vector
    waveform = tf.reshape(waveform, [-1])
    
    # Apply time domain augmentations
    
    # Time shift
    sr = aug_config.get('sample_rate', 8000)
    max_shift = tf.cast(sr * aug_config.get('shift_max_sec', 0.02), tf.int32)
    shift = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
    waveform = tf.roll(waveform, shift, axis=0)
    
    # Pitch shift
    waveform = _apply_pitch_shift(waveform, aug_config)
    
    # Time stretch (formant-preserving)
    waveform = _apply_time_stretch(waveform, aug_config)
    
    # Volume scaling
    scale = tf.random.uniform(
        [], 
        aug_config.get('volume_min', 0.7), 
        aug_config.get('volume_max', 1.3)
    )
    waveform = waveform * scale
    
    # Equalization
    waveform = _apply_eq(waveform, aug_config)
    
    # Dynamic range compression
    waveform = _apply_dynamic_compression(waveform, aug_config)
    
    # Room impulse response
    if aug_config.get('enable_rir', False) and rir_bank is not None:
        waveform = _apply_rir_convolution(waveform, aug_config, rir_bank)
    
    # Additive noise
    noise_std = aug_config.get('noise_level', 0.005)
    if noise_std > 0:
        noise = tf.random.normal(tf.shape(waveform), mean=0.0, stddev=noise_std)
        waveform = waveform + noise
    
    # Cutout
    waveform = _apply_cutout(waveform, aug_config)
    
    # Random gain
    waveform = _apply_random_gain(waveform, aug_config)
    
    # Reverb
    waveform = _apply_reverb(waveform, aug_config)
    
    # Apply mixup or cutmix if enabled and pool is available
    if x_pool is not None and y_pool is not None:
        # Mixup
        if aug_config.get('enable_mixup', False) and tf.random.uniform([], 0, 1) > 0.5:
            alpha = aug_config.get('mixup_alpha', 0.2)
            
            # Sample mixing parameter from Beta distribution
            gamma = tf.random.gamma([1], alpha, 1.0)[0]
            beta = tf.random.gamma([1], alpha, 1.0)[0]
            lam = gamma / (gamma + beta)
            
            # Sample another example
            batch_size = tf.shape(x_pool)[0]
            idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
            x2 = x_pool[idx]
            y2 = y_pool[idx]
            
            # Apply mixup
            waveform = lam * waveform + (1.0 - lam) * x2
            label = lam * label + (1.0 - lam) * y2
        
        # CutMix
        elif aug_config.get('enable_cutmix', False) and tf.random.uniform([], 0, 1) > 0.5:
            max_ratio = aug_config.get('cutmix_max_ratio', 0.3)
            
            # Sample another example
            batch_size = tf.shape(x_pool)[0]
            idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
            x2 = x_pool[idx]
            y2 = y_pool[idx]
            
            # Determine cut region
            length = tf.shape(waveform)[0]
            cut_ratio = tf.random.uniform([], 0.0, max_ratio)
            cut_length = tf.cast(tf.cast(length, tf.float32) * cut_ratio, tf.int32)
            
            # Random starting point
            start = tf.random.uniform([], 0, length - cut_length, dtype=tf.int32)
            
            # Create masks
            mask1 = tf.concat([
                tf.ones(start), 
                tf.zeros(cut_length), 
                tf.ones(length - start - cut_length)
            ], axis=0)
            
            mask2 = 1.0 - mask1
            
            # Apply masks
            waveform = waveform * mask1 + x2 * mask2
            
            # Mix labels based on cut ratio
            label = (1.0 - cut_ratio) * label + cut_ratio * y2
    
    # Random crop/pad to maintain original length
    target_len = tf.shape(waveform)[0]
    waveform = _random_crop(waveform, target_len)
    
    return waveform, label


@tf.function
def augment_spectrogram(spectrogram: tf.Tensor, 
                        label: tf.Tensor, 
                        aug_config: AugmentationConfig,
                        x_pool: Optional[tf.Tensor] = None,
                        y_pool: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply augmentation to spectrogram data.
    
    Args:
        spectrogram: Input spectrogram data [freq_bins, time_frames] or [freq_bins, time_frames, channels] 
                    or [batch, freq_bins, time_frames, channels]
        label: Input label
        aug_config: Augmentation configuration dictionary
        x_pool: Optional pool of spectrograms for mixup/cutmix
        y_pool: Optional pool of labels for mixup/cutmix
        
    Returns:
        Tuple of (augmented_spectrogram, label)
    """
    # Ensure proper tensor format
    if isinstance(spectrogram, np.ndarray):
        spectrogram = tf.convert_to_tensor(spectrogram)
    
    # Ensure spectrogram is float32
    spectrogram = tf.cast(spectrogram, tf.float32)
    
    # Normalize shape to at least 3D for consistency
    rank = tf.rank(spectrogram)
    spectrogram = tf.cond(
        tf.equal(rank, 2),
        lambda: tf.expand_dims(spectrogram, axis=-1),
        lambda: spectrogram
    )
    
    # Time masking - hide random time frames
    if aug_config.get('enable_cutout', False) and tf.random.uniform([], 0, 1) > 0.5:
        time_dim = tf.shape(spectrogram)[1]
        max_mask_width = tf.cast(time_dim * 0.1, tf.int32)  # 10% max
        mask_width = tf.random.uniform([], 1, max_mask_width, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, time_dim - mask_width, dtype=tf.int32)
        
        # Create a mask to zero out time frames
        mask = tf.concat([
            tf.ones([mask_start]),
            tf.zeros([mask_width]),
            tf.ones([time_dim - mask_start - mask_width])
        ], axis=0)
        
        # Apply mask along time dimension
        mask = tf.reshape(mask, [1, -1, 1])  # [1, time, 1]
        spectrogram = spectrogram * mask
    
    # Frequency masking - hide random frequency bins
    if aug_config.get('enable_eq', False) and tf.random.uniform([], 0, 1) > 0.5:
        freq_dim = tf.shape(spectrogram)[0]
        max_mask_height = tf.cast(freq_dim * 0.1, tf.int32)  # 10% max
        mask_height = tf.random.uniform([], 1, max_mask_height, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, freq_dim - mask_height, dtype=tf.int32)
        
        # Create a mask to zero out frequency bins
        mask = tf.concat([
            tf.ones([mask_start]),
            tf.zeros([mask_height]),
            tf.ones([freq_dim - mask_start - mask_height])
        ], axis=0)
        
        # Apply mask along frequency dimension
        mask = tf.reshape(mask, [-1, 1, 1])  # [freq, 1, 1]
        spectrogram = spectrogram * mask
    
    # Gain augmentation
    if tf.random.uniform([], 0, 1) > 0.5:
        gain = tf.random.uniform(
            [], 
            aug_config.get('volume_min', 0.7), 
            aug_config.get('volume_max', 1.3)
        )
        spectrogram = spectrogram * gain
    
    # Apply mixup or cutmix if enabled and pool is available
    if x_pool is not None and y_pool is not None:
        # Mixup for spectrograms
        if aug_config.get('enable_mixup', False) and tf.random.uniform([], 0, 1) > 0.5:
            alpha = aug_config.get('mixup_alpha', 0.2)
            
            # Sample mixing parameter from Beta distribution
            gamma = tf.random.gamma([1], alpha, 1.0)[0]
            beta = tf.random.gamma([1], alpha, 1.0)[0]
            lam = gamma / (gamma + beta)
            
            # Sample another example
            batch_size = tf.shape(x_pool)[0]
            idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
            x2 = x_pool[idx]
            y2 = y_pool[idx]
            
            # Apply mixup
            spectrogram = lam * spectrogram + (1.0 - lam) * x2
            label = lam * label + (1.0 - lam) * y2
        
        # SpecAugment-style CutMix for spectrograms
        elif aug_config.get('enable_cutmix', False) and tf.random.uniform([], 0, 1) > 0.5:
            max_ratio = aug_config.get('cutmix_max_ratio', 0.3)
            
            # Sample another example
            batch_size = tf.shape(x_pool)[0]
            idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
            x2 = x_pool[idx]
            y2 = y_pool[idx]
            
            # Create a rectangular mask
            freq_dim = tf.shape(spectrogram)[0]
            time_dim = tf.shape(spectrogram)[1]
            
            # Determine mask size as percentage of total area
            area = freq_dim * time_dim
            mask_area = tf.cast(tf.cast(area, tf.float32) * max_ratio, tf.int32)
            
            # Calculate mask dimensions
            aspect_ratio = tf.random.uniform([], 0.3, 3.0)  # Random aspect ratio
            mask_width = tf.cast(tf.sqrt(tf.cast(mask_area, tf.float32) * aspect_ratio), tf.int32)
            mask_height = tf.cast(tf.cast(mask_area, tf.float32) / tf.cast(mask_width, tf.float32), tf.int32)
            
            # Ensure mask fits within spectrogram
            mask_width = tf.minimum(mask_width, time_dim)
            mask_height = tf.minimum(mask_height, freq_dim)
            
            # Random starting points
            time_start = tf.random.uniform([], 0, time_dim - mask_width, dtype=tf.int32)
            freq_start = tf.random.uniform([], 0, freq_dim - mask_height, dtype=tf.int32)
            
            # Create mask indices
            time_indices = tf.range(time_start, time_start + mask_width)
            freq_indices = tf.range(freq_start, freq_start + mask_height)
            
            # Calculate area ratio for label mixing
            cut_ratio = tf.cast(mask_width * mask_height, tf.float32) / tf.cast(area, tf.float32)
            
            # Apply the mask (need to use scatter_nd for 2D masking)
            # Create a mask of all ones
            mask = tf.ones([freq_dim, time_dim, 1], dtype=tf.float32)
            
            # Create a grid of indices for the mask region
            time_grid, freq_grid = tf.meshgrid(time_indices, freq_indices)
            mask_indices = tf.stack([freq_grid, time_grid], axis=-1)
            mask_indices = tf.reshape(mask_indices, [-1, 2])
            
            # Set mask region to zeros
            zeros = tf.zeros([tf.shape(mask_indices)[0], 1], dtype=tf.float32)
            mask = tf.tensor_scatter_nd_update(mask, mask_indices, zeros)
            
            # Apply masks to both spectrograms
            spectrogram = spectrogram * mask + x2 * (1.0 - mask)
            
            # Mix labels based on cut ratio
            label = (1.0 - cut_ratio) * label + cut_ratio * y2
    
    return spectrogram, label


def get_augmentation_config(config: Dict[str, Any], mode: str) -> AugmentationConfig:
    """
    Build augmentation parameters for the given mode ('weak' or 'strong') based on config.
    
    Args:
        config: Full configuration dictionary
        mode: Augmentation mode ('weak' or 'strong')
        
    Returns:
        Augmentation configuration dictionary with appropriate parameters
    """
    mode_cfg = config.get('augmentation', {}).get(mode, {})
    aug_config: AugmentationConfig = {}
    
    # Common parameters
    aug_config['sample_rate'] = config.get('sample_rate', 8000)
    aug_config['audio_length'] = config.get('audio_length', 0.3)
    
    # Noise injection
    aug_config['noise_level'] = mode_cfg.get('noise_level', 0.0) if mode_cfg.get('add_noise', False) else 0.0
    
    # Time shift
    aug_config['shift_max_sec'] = mode_cfg.get('shift_factor', 0.0) if mode_cfg.get('time_shift', False) else 0.0
    
    # Volume scaling
    if 'volume_min' in mode_cfg:
        aug_config['volume_min'] = mode_cfg['volume_min']
    if 'volume_max' in mode_cfg:
        aug_config['volume_max'] = mode_cfg['volume_max']
    
    # Pitch shift - weak mode = small shifts, strong mode = larger shifts
    aug_config['enable_pitch_shift'] = mode_cfg.get('enable_pitch_shift', False)
    if mode == 'weak' and aug_config['enable_pitch_shift']:
        aug_config['pitch_shift_min'] = mode_cfg.get('pitch_shift_min', 0.99)  # -1%
        aug_config['pitch_shift_max'] = mode_cfg.get('pitch_shift_max', 1.01)  # +1%
    elif mode == 'strong' and aug_config['enable_pitch_shift']:
        aug_config['pitch_shift_min'] = mode_cfg.get('pitch_shift_min', 0.97)  # -3%
        aug_config['pitch_shift_max'] = mode_cfg.get('pitch_shift_max', 1.03)  # +3%
    
    # Time stretch
    aug_config['enable_time_stretch'] = mode_cfg.get('enable_time_stretch', False)
    if mode == 'weak' and aug_config['enable_time_stretch']:
        aug_config['time_stretch_min'] = mode_cfg.get('time_stretch_min', 0.95)
        aug_config['time_stretch_max'] = mode_cfg.get('time_stretch_max', 1.05)
    elif mode == 'strong' and aug_config['enable_time_stretch']:
        aug_config['time_stretch_min'] = mode_cfg.get('time_stretch_min', 0.9)
        aug_config['time_stretch_max'] = mode_cfg.get('time_stretch_max', 1.1)
    
    # Equalization
    aug_config['enable_eq'] = mode_cfg.get('enable_eq', False)
    if aug_config['enable_eq']:
        aug_config['eq_bands'] = mode_cfg.get('eq_bands', 4)
        if mode == 'weak':
            aug_config['eq_min_gain'] = mode_cfg.get('eq_min_gain', 0.8)  # -2dB
            aug_config['eq_max_gain'] = mode_cfg.get('eq_max_gain', 1.2)  # +2dB
        else:
            aug_config['eq_min_gain'] = mode_cfg.get('eq_min_gain', 0.7)  # -3dB
            aug_config['eq_max_gain'] = mode_cfg.get('eq_max_gain', 1.4)  # +3dB
    
    # Cutout
    aug_config['enable_cutout'] = mode_cfg.get('enable_cutout', False)
    if aug_config['enable_cutout']:
        aug_config['cutout_max_width'] = mode_cfg.get('cutout_max_width', 0.05 if mode == 'weak' else 0.1)
    
    # Dynamic range compression
    aug_config['enable_compression'] = mode_cfg.get('enable_compression', False)
    if aug_config['enable_compression']:
        if mode == 'weak':
            aug_config['compression_threshold'] = mode_cfg.get('compression_threshold', -25.0)
            aug_config['compression_ratio'] = mode_cfg.get('compression_ratio', 2.0)
        else:
            aug_config['compression_threshold'] = mode_cfg.get('compression_threshold', -20.0)
            aug_config['compression_ratio'] = mode_cfg.get('compression_ratio', 4.0)
    
    # Room impulse response convolution
    aug_config['enable_rir'] = mode_cfg.get('enable_rir', False)
    
    # Mixup and CutMix
    aug_config['enable_mixup'] = mode_cfg.get('enable_mixup', False)
    if aug_config['enable_mixup']:
        aug_config['mixup_alpha'] = mode_cfg.get('mixup_alpha', 0.2)
    
    aug_config['enable_cutmix'] = mode_cfg.get('enable_cutmix', False)
    if aug_config['enable_cutmix']:
        aug_config['cutmix_max_ratio'] = mode_cfg.get('cutmix_max_ratio', 
                                                     0.15 if mode == 'weak' else 0.3)
    
    return aug_config


def create_ssl_augment_labeled_fn(config_dict: Dict[str, Any], 
                                 data_type: str,
                                 rir_bank: Optional[tf.Tensor] = None) -> Callable:
    """
    Creates a mapping function for augmenting labeled SSL data.

    Args:
        config_dict: Configuration dictionary (from config.to_dict()).
        data_type: Type of data ('raw' or 'stft').
        rir_bank: Optional bank of room impulse responses.

    Returns:
        A callable function for use with tf.data.Dataset.map().
    """
    # Build actual augmentation configs for weak augmentations
    aug_config = get_augmentation_config(config_dict, 'weak')
    
    @tf.function
    def _augment_labeled(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # apply waveform or spectrogram augmentation depending on data_type
        if data_type == 'raw':
            return augment_waveform(x, y, aug_config, rir_bank=rir_bank)
        else:
            return augment_spectrogram(x, y, aug_config)
    
    return _augment_labeled


def create_ssl_augment_unlabeled_fn(config_dict: Dict[str, Any], 
                                   data_type: str,
                                   rir_bank: Optional[tf.Tensor] = None) -> Callable:
    """
    Creates a mapping function for augmenting unlabeled SSL data.
    
    Returns two versions of each sample - weakly and strongly augmented.

    Args:
        config_dict: Configuration dictionary (from config.to_dict()).
        data_type: Type of data ('raw' or 'stft').
        rir_bank: Optional bank of room impulse responses.

    Returns:
        A callable function for use with tf.data.Dataset.map(), 
        returning ((x_weak, x_strong), y_placeholder).
    """
    aug_config_strong = get_augmentation_config(config_dict, 'strong')
    aug_config_weak = get_augmentation_config(config_dict, 'weak')
    
    @tf.function
    def _augment_unlabeled(x: tf.Tensor, y: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        # y is typically a placeholder for unlabeled data and is passed through
        
        # Apply actual augmentations
        if data_type == 'raw':
            x_strong_augmented, _ = augment_waveform(x, y, aug_config_strong, rir_bank=rir_bank)
            x_weak_augmented, _ = augment_waveform(x, y, aug_config_weak, rir_bank=rir_bank)
        else:
            x_strong_augmented, _ = augment_spectrogram(x, y, aug_config_strong)
            x_weak_augmented, _ = augment_spectrogram(x, y, aug_config_weak)
        
        return (x_weak_augmented, x_strong_augmented), y
    
    return _augment_unlabeled


def create_mixup_augment_fn(config_dict: Dict[str, Any], 
                           data_type: str) -> Callable:
    """
    Creates a mapping function for applying mixup augmentation to a batch of data.

    Args:
        config_dict: Configuration dictionary.
        data_type: Type of data ('raw' or 'stft').

    Returns:
        A callable function for use with tf.data.Dataset.batch().map().
    """
    aug_config = get_augmentation_config(config_dict, 'strong')
    
    @tf.function
    def _apply_batch_mixup(x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply mixup to a batch of data.
        
        Args:
            x_batch: Batch of features [batch, ...]
            y_batch: Batch of labels [batch, classes]
            
        Returns:
            Tuple of (mixed_features, mixed_labels)
        """
        batch_size = tf.shape(x_batch)[0]
        
        # Sample mixing parameter from Beta distribution
        alpha = aug_config.get('mixup_alpha', 0.2)
        gamma = tf.random.gamma([batch_size], alpha, 1.0)
        beta = tf.random.gamma([batch_size], alpha, 1.0)
        lam = gamma / (gamma + beta)
        
        # Create reshaping dimensions for broadcasting
        if data_type == 'raw':
            # For waveform: [batch, time]
            lam_x = tf.reshape(lam, [batch_size, 1])
        else:
            # For spectrogram: [batch, freq, time, channels]
            lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
        
        # For labels: [batch, classes]
        lam_y = tf.reshape(lam, [batch_size, 1])
        
        # Sample indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Apply mixup
        mixed_x = lam_x * x_batch + (1.0 - lam_x) * tf.gather(x_batch, indices)
        mixed_y = lam_y * y_batch + (1.0 - lam_y) * tf.gather(y_batch, indices)
        
        return mixed_x, mixed_y
    
    return _apply_batch_mixup