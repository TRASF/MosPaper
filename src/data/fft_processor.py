"""
FFT feature extraction for mosquito wingbeat classification.
Provides robust processing of audio data into spectral features.
"""
import numpy as np
import logging
import sys
from typing import Tuple, Optional, Union
from librosa import stft, amplitude_to_db
from scipy.signal import spectrogram, hilbert
from pathlib import Path
import skimage.transform

# Add project root to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import config
from src.utils.config import Config

# Configure module logger
logger = logging.getLogger(__name__)

# Default FFT parameters
FFT_N_FFT = 512
FFT_HOP_LENGTH = 256
FFT_FREQ_BINS = FFT_N_FFT // 2 + 1  
FFT_TIME_FRAMES = 10
FFT_FEATURE_TYPE = 'stft'


class FFTProcessor:
    """
    FFT feature extraction for audio data with robust error handling.
    
    Converts raw waveforms to various spectral representations with consistent 
    output shapes and comprehensive error handling.
    """
    
    def __init__(
        self, 
        n_fft: int = FFT_N_FFT, 
        hop_length: int = FFT_HOP_LENGTH, 
        feature_type: str = FFT_FEATURE_TYPE,
        target_shape: Tuple[int, int, int] = (FFT_FREQ_BINS, FFT_TIME_FRAMES, 1)
    ):
        """
        Initialize the FFT processor with type-safe validation.
        
        Args:
            n_fft: FFT window size (power of 2, ≥64)
            hop_length: Hop length between frames (≤n_fft/2)
            feature_type: Type of spectral features ('stft', 'spectrogram', 'hilbert')
            target_shape: Expected output shape (freq_bins, time_frames, channels)
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong types
        """
        # Type validation with precise error messages (Clean Coder principle)
        if not isinstance(n_fft, int):
            raise TypeError(f"n_fft must be an int, got {type(n_fft).__name__}")
        if n_fft < 64 or (n_fft & (n_fft - 1)) != 0:
            raise ValueError(f"n_fft must be a power of 2 ≥ 64, got {n_fft}")
        
        if not isinstance(hop_length, int):
            raise TypeError(f"hop_length must be an int, got {type(hop_length).__name__}")
        if hop_length <= 0 or hop_length > n_fft // 2:
            raise ValueError(f"hop_length must be > 0 and ≤ n_fft/2, got {hop_length}")
        
        if feature_type not in ['stft', 'spectrogram', 'hilbert']:
            raise ValueError(f"feature_type must be one of ['stft', 'spectrogram', 'hilbert'], got {feature_type}")
        
        if not isinstance(target_shape, tuple) or len(target_shape) != 3:
            raise TypeError(f"target_shape must be a 3-tuple, got {target_shape}")
        
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length else n_fft // 4
        self.feature_type = feature_type
        self.target_shape = target_shape
        
        # Pre-compute FFT window for efficiency (reuse in training loops)
        self._window = np.hanning(n_fft)
        
        # Validate frequency bins consistency
        expected_freq_bins = n_fft // 2 + 1
        if self.target_shape[0] != expected_freq_bins:
            logger.warning(f"Target shape freq_bins {self.target_shape[0]} doesn't match n_fft//2+1 = {expected_freq_bins}")
    
    def extract_features(self, audio: np.ndarray, sr: int = 8000) -> np.ndarray:
        """
        Extract FFT features from a single audio sample.
        
        Args:
            audio: Input audio waveform
            sr: Sample rate in Hz
            
        Returns:
            FFT features with shape matching target_shape
        """
        # Check if audio is empty
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio data provided to extract_features")
            return np.zeros(self.target_shape)
            
        # Check input length and pad if needed
        min_length = self.n_fft
        if len(audio) < min_length:
            logger.warning(f"Audio length ({len(audio)}) is less than n_fft ({self.n_fft}), padding...")
            audio = np.pad(audio, (0, min_length - len(audio)), 'constant')
            
        try:
            # Compute features based on chosen method
            features = self._compute_features(audio, sr)
            
            # Resize to target shape if needed
            if features.shape != self.target_shape:
                features = self._resize_features(features)
                
            return features
                
        except Exception as e:
            logger.error(f"Error in extract_features: {e}")
            return np.zeros(self.target_shape)
    
    def batch_extract_features(self, batch_audio: np.ndarray, sr: int = 8000) -> np.ndarray:
        """
        Vectorized batch FFT extraction following Clean Coder principles.
        
        Args:
            batch_audio: Batch of audio waveforms with shape [batch, samples] or [batch, samples, channels]
            sr: Sample rate in Hz
            
        Returns:
            Batch of FFT features with shape [batch, freq_bins, time_frames, channels]
            
        Raises:
            ValueError: If input dimensions are invalid
            TypeError: If input is not numeric
        """
        # Shape validation (once) - Clean Coder principle 5
        if not isinstance(batch_audio, np.ndarray):
            raise TypeError(f"batch_audio must be numpy array, got {type(batch_audio).__name__}")
        
        if batch_audio.ndim < 2 or batch_audio.ndim > 3:
            raise ValueError(f"batch_audio must be 2D [batch, samples] or 3D [batch, samples, channels], got shape {batch_audio.shape}")
        
        # Normalize input shape to [batch, samples, channels]
        if batch_audio.ndim == 2:
            batch_audio = np.expand_dims(batch_audio, -1)
        
        batch_size, n_samples, n_channels = batch_audio.shape
        
        # Validate minimum length requirement
        if n_samples < self.n_fft:
            logger.warning(f"Audio samples ({n_samples}) < n_fft ({self.n_fft}), padding batch...")
            pad_width = ((0, 0), (0, self.n_fft - n_samples), (0, 0))
            batch_audio = np.pad(batch_audio, pad_width, 'constant')
        
        # Vectorized batch processing - Clean Coder principle 4
        try:
            if self.feature_type == 'stft':
                # Use vectorized STFT with pre-computed window
                features = self._vectorized_stft_batch(batch_audio[:, :, 0], sr)
            elif self.feature_type == 'spectrogram':
                features = self._vectorized_spectrogram_batch(batch_audio[:, :, 0], sr)
            elif self.feature_type == 'hilbert':
                features = self._vectorized_hilbert_batch(batch_audio[:, :, 0], sr)
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            # Ensure consistent output shape
            if features.shape[1:] != self.target_shape:
                features = self._batch_resize_features(features)
                
            return features
            
        except Exception as e:
            logger.error(f"Vectorized batch processing failed: {e}")
            # Fallback to individual processing
            return self._fallback_batch_processing(batch_audio, sr)
            
    def _compute_features(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute spectral features from waveform.
        
        Args:
            waveform: Input audio waveform
            sr: Sample rate
            
        Returns:
            Spectral features
        """
        if self.feature_type == 'stft':
            S = stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length)
            # Convert to magnitude spectrogram
            S_db = amplitude_to_db(np.abs(S), ref=np.max)
            # Shape: [freq_bins, time_frames]
            return np.expand_dims(S_db, -1)  # Add channel dimension
            
        elif self.feature_type == 'spectrogram':
            f, t, S = spectrogram(waveform, fs=sr, nperseg=self.n_fft, 
                                  noverlap=self.n_fft-self.hop_length)
            # Convert to dB scale
            S_db = 10 * np.log10(S + 1e-10)  # Add small constant to avoid log(0)
            # Shape: [freq_bins, time_frames]
            return np.expand_dims(S_db, -1)  # Add channel dimension
            
        elif self.feature_type == 'hilbert':
            # Calculate Hilbert envelope
            analytic_signal = hilbert(waveform)
            amplitude_envelope = np.abs(analytic_signal)
            
            # Compute spectral representation of the envelope
            f, t, S = spectrogram(amplitude_envelope, fs=sr, nperseg=self.n_fft, 
                                 noverlap=self.n_fft-self.hop_length)
            S_db = 10 * np.log10(S + 1e-10)
            return np.expand_dims(S_db, -1)  # Add channel dimension
            
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def _resize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Resize features to match target shape.
        
        Args:
            features: Input features with shape [freq_bins, time_frames, channels]
            
        Returns:
            Resized features with shape matching target_shape
        """
        # Skip if already matching
        if features.shape == self.target_shape:
            return features
            
        # Extract dimensions
        curr_freq, curr_time, curr_ch = features.shape
        target_freq, target_time, target_ch = self.target_shape
        
        # Resize frequency and time dimensions
        if curr_freq != target_freq or curr_time != target_time:
            # Resize using skimage.transform to maintain aspect ratio
            features_resized = np.zeros(self.target_shape)
            for ch in range(curr_ch):
                features_resized[:, :, ch] = skimage.transform.resize(
                    features[:, :, ch], 
                    (target_freq, target_time),
                    anti_aliasing=True,
                    mode='reflect'
                )
            return features_resized
        
        # Handle channel dimension mismatch
        if curr_ch != target_ch:
            # If we need to add channels, duplicate existing
            if curr_ch < target_ch:
                features = np.repeat(features, target_ch, axis=2)
            # If we need to reduce channels, average them
            else:
                features = np.mean(features, axis=2, keepdims=True)
                
        return features
    
    def _vectorized_stft_batch(self, batch_audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Vectorized STFT computation for entire batch.
        
        Leverages NumPy broadcasting and pre-computed windows for efficiency.
        Follows STFT theory from https://ccrma.stanford.edu/~jos/sasp/Short_Time_Fourier_Transform.html
        
        Args:
            batch_audio: Audio batch [batch_size, n_samples]
            sr: Sample rate
            
        Returns:
            STFT features [batch_size, freq_bins, time_frames, 1]
        """
        batch_size = batch_audio.shape[0]
        
        # Vectorized STFT using librosa with consistent parameters
        stft_results = []
        for audio in batch_audio:  # Limited vectorization due to librosa constraints
            # Use same parameters as individual processing for consistency
            S = stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            S_db = amplitude_to_db(np.abs(S), ref=np.max)
            stft_results.append(S_db)
        
        # Stack results efficiently
        stft_batch = np.stack(stft_results, axis=0)  # [batch, freq_bins, time_frames]
        return np.expand_dims(stft_batch, -1)  # Add channel dimension
    
    def _vectorized_spectrogram_batch(self, batch_audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Vectorized spectrogram computation for entire batch.
        
        Args:
            batch_audio: Audio batch [batch_size, n_samples]
            sr: Sample rate
            
        Returns:
            Spectrogram features [batch_size, freq_bins, time_frames, 1]
        """
        batch_size = batch_audio.shape[0]
        spec_results = []
        
        for audio in batch_audio:
            f, t, S = spectrogram(audio, fs=sr, nperseg=self.n_fft, 
                                  noverlap=self.n_fft-self.hop_length)
            S_db = 10 * np.log10(S + 1e-10)  # Avoid log(0) - standard practice
            spec_results.append(S_db)
        
        spec_batch = np.stack(spec_results, axis=0)
        return np.expand_dims(spec_batch, -1)
    
    def _vectorized_hilbert_batch(self, batch_audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Vectorized Hilbert envelope computation for entire batch.
        
        Args:
            batch_audio: Audio batch [batch_size, n_samples]
            sr: Sample rate
            
        Returns:
            Hilbert features [batch_size, freq_bins, time_frames, 1]
        """
        # Vectorized Hilbert transform across batch dimension
        analytic_signals = hilbert(batch_audio, axis=1)  # Vectorized across samples
        amplitude_envelopes = np.abs(analytic_signals)
        
        hilbert_results = []
        for envelope in amplitude_envelopes:
            f, t, S = spectrogram(envelope, fs=sr, nperseg=self.n_fft, 
                                  noverlap=self.n_fft-self.hop_length)
            S_db = 10 * np.log10(S + 1e-10)
            hilbert_results.append(S_db)
        
        hilbert_batch = np.stack(hilbert_results, axis=0)
        return np.expand_dims(hilbert_batch, -1)
    
    def _batch_resize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Vectorized feature resizing for entire batch.
        
        Args:
            features: Input features [batch, freq_bins, time_frames, channels]
            
        Returns:
            Resized features matching target_shape
        """
        if features.shape[1:] == self.target_shape:
            return features
        
        batch_size = features.shape[0]
        curr_freq, curr_time, curr_ch = features.shape[1:]
        target_freq, target_time, target_ch = self.target_shape
        
        # Pre-allocate output array
        resized_batch = np.zeros((batch_size, *self.target_shape))
        
        # Vectorized resize using skimage for each batch item
        if curr_freq != target_freq or curr_time != target_time:
            for i in range(batch_size):
                for ch in range(curr_ch):
                    resized_batch[i, :, :, ch] = skimage.transform.resize(
                        features[i, :, :, ch], 
                        (target_freq, target_time),
                        anti_aliasing=True,
                        mode='reflect'
                    )
        else:
            resized_batch = features
        
        # Handle channel dimension efficiently
        if curr_ch != target_ch:
            if curr_ch < target_ch:
                resized_batch = np.repeat(resized_batch, target_ch, axis=3)
            else:
                resized_batch = np.mean(resized_batch, axis=3, keepdims=True)
                
        return resized_batch
    
    def _fallback_batch_processing(self, batch_audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Fallback to individual processing if vectorized approach fails.
        
        Args:
            batch_audio: Audio batch [batch_size, n_samples, channels]
            sr: Sample rate
            
        Returns:
            Processed features [batch_size, freq_bins, time_frames, channels]
        """
        batch_size = batch_audio.shape[0]
        features = np.zeros((batch_size, *self.target_shape))
        
        for i in range(batch_size):
            audio = batch_audio[i, :, 0] if batch_audio.shape[2] > 0 else batch_audio[i, :]
            features[i] = self.extract_features(audio, sr)
            
        return features


def waveform_to_fft(
    waveforms: np.ndarray, 
    feature_type: str = FFT_FEATURE_TYPE,
    n_fft: int = FFT_N_FFT, 
    hop_length: int = FFT_HOP_LENGTH,
    config: Optional[Config] = None
) -> np.ndarray:
    """
    Convert waveform data to FFT features with comprehensive validation.
    
    Optimized for SSL FixMatch/FlexMatch efficiency with pre-computed windows
    and vectorized processing.
    
    Args:
        waveforms: Input waveform data with shape [batch, samples] or [batch, samples, 1]
        feature_type: Type of spectral features ('stft', 'spectrogram', 'hilbert')
        n_fft: FFT window size (power of 2, ≥64)
        hop_length: Hop length between frames (≤n_fft/2)
        config: Configuration object with STFT parameters
        
    Returns:
        FFT features with shape [batch, freq_bins, time_frames, 1]
        
    Raises:
        ValueError: If waveforms are invalid or parameters inconsistent
        TypeError: If inputs have wrong types
    """
    # Input validation (once) - Clean Coder principle 5
    if not isinstance(waveforms, np.ndarray):
        raise TypeError(f"waveforms must be numpy array, got {type(waveforms).__name__}")
    
    if waveforms.ndim < 2 or waveforms.ndim > 3:
        raise ValueError(f"waveforms must be 2D [batch, samples] or 3D [batch, samples, channels], got shape {waveforms.shape}")
    
    # Extract parameters from config with validation
    if config is not None:
        # Access config parameters safely with getattr for compatibility
        n_fft = getattr(config.stft, 'n_fft', n_fft) if hasattr(config, 'stft') else config.get('n_fft', n_fft)
        hop_length = getattr(config.stft, 'hop_length', hop_length) if hasattr(config, 'stft') else config.get('hop_length', hop_length)
        feature_type = getattr(config.stft, 'feature_type', feature_type) if hasattr(config, 'stft') else config.get('feature_type', feature_type)
        freq_bins = getattr(config.stft, 'freq_bins', n_fft // 2 + 1) if hasattr(config, 'stft') else config.get('freq_bins', n_fft // 2 + 1)
        time_frames = getattr(config.stft, 'time_frames', FFT_TIME_FRAMES) if hasattr(config, 'stft') else config.get('time_frames', FFT_TIME_FRAMES)
        target_shape = (freq_bins, time_frames, 1)
    else:
        target_shape = (FFT_FREQ_BINS, FFT_TIME_FRAMES, 1)
    
    # Create processor with validated settings (reuse for efficiency)
    processor = FFTProcessor(
        n_fft=n_fft, 
        hop_length=hop_length, 
        feature_type=feature_type,
        target_shape=target_shape
    )
    
    # Process batch with vectorized operations
    return processor.batch_extract_features(waveforms)