from typing import Tuple, Optional, List, Dict
import numpy as np
import logging
import os
import sys
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, resample
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import config
from src.utils.config import Config
from src.data.normalize import normalize

# Configure logger
logger = logging.getLogger(__name__)

# Default audio processing parameters
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_SEGMENT_SAMPLES = 2400  # 0.3s @ 8kHz
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FILTER_LOWCUT = 350
DEFAULT_FILTER_HIGHCUT = 3800


class AudioProcessor:
    """
    Complete audio processing pipeline for mosquito wingbeat sounds.
    
    Handles loading, filtering, normalization, and segmentation of audio files
    with comprehensive error handling and logging.
    """
    
    def __init__(
        self, 
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
        filter_lowcut: int = DEFAULT_FILTER_LOWCUT,
        filter_highcut: int = DEFAULT_FILTER_HIGHCUT,
        filter_type: str = 'bandpass',
        filter_order: int = 4,
        normalize_method: str = 'rms',
        pad_mode: str = 'reflect'
    ):
        """Initialize the audio processor with specific settings."""
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.overlap_ratio = overlap_ratio
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_type = filter_type
        self.filter_order = filter_order
        self.normalize_method = normalize_method
        self.pad_mode = pad_mode
        
    def process_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (segmented_audio, class_label) where segmented_audio has shape [n_segments, samples, 1]
            Returns (None, None) if processing fails
        """
        # Load and preprocess the audio
        audio, sr = self.load_audio(file_path)
        
        if audio is None:
            logger.warning(f"Failed to load audio from {file_path}")
            return None, None
        
        # Apply filtering
        audio = self.apply_filter(audio)
        
        # Apply normalization
        audio = normalize(audio, normalize_method=self.normalize_method)
        
        # Segment the audio
        segments = self.segment_audio(audio)
        
        # Extract label
        label = self.extract_label(file_path)
        
        return segments, label
    
    def load_audio(self, file_path: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Load and resample an audio file with robust error handling.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_samples, sample_rate) where audio_samples is a numpy array
            Returns (None, sample_rate) if loading fails
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None, self.sample_rate
                
            # Load audio file
            audio, sr = librosa.load(file_path, sr=None)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
                
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None, self.sample_rate
    
    def apply_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to the audio data.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Filtered audio waveform
        """
        try:
            # Get filter coefficients
            sos = self._get_filter_sos()
            
            # Apply filter
            filtered_audio = sosfiltfilt(sos, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return audio
    
    def _get_filter_sos(self):
        """
        Get second-order sections filter coefficients.
        
        Returns:
            SOS filter coefficients
        """
        nyquist = 0.5 * self.sample_rate
        low = self.filter_lowcut / nyquist
        high = self.filter_highcut / nyquist
        
        if self.filter_type == 'bandpass':
            return butter(self.filter_order, [low, high], btype='band', output='sos')
        elif self.filter_type == 'lowpass':
            return butter(self.filter_order, high, btype='low', output='sos')
        elif self.filter_type == 'highpass':
            return butter(self.filter_order, low, btype='high', output='sos')
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def segment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Segment audio into fixed-length windows with overlap.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Array of audio segments with shape [n_segments, segment_samples, 1]
        """
        # Check if audio is shorter than segment length
        if len(audio) < self.segment_samples:
            # Pad audio to segment length
            padded = np.pad(audio, (0, self.segment_samples - len(audio)), mode=self.pad_mode)
            # Return as a single segment with channel dimension
            return np.reshape(padded, (1, self.segment_samples, 1))
            
        # Calculate step size based on overlap ratio
        step = int(self.segment_samples * (1 - self.overlap_ratio))
        
        # Calculate number of segments
        n_segments = 1 + (len(audio) - self.segment_samples) // step
        
        # Initialize output array
        segments = np.zeros((n_segments, self.segment_samples, 1))
        
        # Extract segments
        for i in range(n_segments):
            start = i * step
            end = start + self.segment_samples
            segment = audio[start:end]
            
            # Add channel dimension and store
            segments[i, :, 0] = segment
            
        return segments
    
    def extract_label(self, file_path: str, default_class: int = 10) -> int:
        """
        Extract class label from file path.
        
        Args:
            file_path: Path to the audio file
            default_class: Default class to return if extraction fails (No Mosquito = 10)
            
        Returns:
            Class label (integer)
        """
        try:
            from re import match as re_match
            
            # Define the class dictionary for mosquito species
            species_dict = {
                'Ae.Aegypti_F': 0,
                'Ae.Aegypti_M': 1,
                'Ae.Albopictus_F': 2,
                'Ae.Albopictus_M': 3,
                'An.Dirus_F': 4,
                'An.Dirus_M': 5,
                'An.Minimus_F': 6,
                'An.Minimus_M': 7,
                'Cx.Quin_F': 8,
                'Cx.Quin_M': 9,
                'No.Mos': 10
            }
            
            base_name = os.path.basename(file_path)
            
            # Pattern for standard mosquito species format
            match = re_match(r'.*?([A-Za-z]+\.[A-Za-z]+)_(\d+[FM])', base_name)
            if match:
                species = match.group(1)
                gender = match.group(2)[1:]
                label_key = f"{species}_{gender}"
                return species_dict.get(label_key, default_class)
            
            # Alternative pattern that might contain species name
            for species_key, class_id in species_dict.items():
                if species_key in base_name:
                    return class_id
            
            # Pattern for numeric class format
            alt_match = re_match(r'.*?class_(\d+).*', base_name)
            if alt_match:
                try:
                    return int(alt_match.group(1))
                except ValueError:
                    pass
        
            # If file contains "noise", classify as No Mosquito (10)
            if "noise" in base_name.lower():
                return 10
                
            return default_class
        
        except Exception as e:
            logger.error(f"Error extracting label from {file_path}: {e}")
            return default_class



def process_audio_files(file_paths: List[str], sample_rate=DEFAULT_SAMPLE_RATE, 
                       segment_samples=DEFAULT_SEGMENT_SAMPLES, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process multiple audio files and concatenate the results.
    
    Args:
        file_paths: List of audio file paths
        sample_rate: Target sample rate for audio processing
        segment_samples: Number of samples per audio segment
        **kwargs: Additional arguments to pass to AudioProcessor
        
    Returns:
        Tuple of (all_segments, all_labels, all_file_ids) 
    """
    # Extract normalize and apply_filtering parameters so they're not passed to AudioProcessor constructor
    processor_kwargs = {k: v for k, v in kwargs.items() if k not in ['normalize', 'apply_filtering']}
    
    # Create processor with the filtered kwargs
    processor = AudioProcessor(sample_rate=sample_rate, segment_samples=segment_samples, **processor_kwargs)
    
    all_segments = []
    all_labels = []
    all_file_ids = []  # Track which file each segment came from
    
    for i, file_path in enumerate(file_paths):
        try:
            # Process file
            segments, label = processor.process_file(file_path)
            
            if segments is not None and label is not None:
                # Add segments and labels
                all_segments.append(segments)
                all_labels.extend([label] * len(segments))
                all_file_ids.extend([i] * len(segments))
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Concatenate all segments and labels
    if all_segments:
        all_segments = np.vstack(all_segments)
        all_labels = np.array(all_labels)
        all_file_ids = np.array(all_file_ids)
        
        return all_segments, all_labels, all_file_ids
    
    # Return empty arrays if no valid segments
    return np.array([]), np.array([]), np.array([])