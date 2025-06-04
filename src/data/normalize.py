import numpy as np
import logging

logger = logging.getLogger(__name__)

def peak_normalization(data):
    """
    Normalize the audio data to have a peak amplitude of 1.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    # For our mosquito data, max amplitude is always > 0
    # so we can simplify this
    max_amplitude = np.max(np.abs(data))
    return data / (max_amplitude or 1.0)  # Use 1.0 as fallback if max is 0

def min_max_normalization(data):
    """
    Normalize the audio data to be in the range [0, 1].
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    # Simplified for our specific mosquito data processing
    min_val = np.min(data)
    range_val = np.max(data) - min_val
    return (data - min_val) / (range_val or 1.0)  # Use 1.0 as fallback if range is 0

def z_score_normalization(data):
    """
    Normalize the audio data to have a mean of 0 and standard deviation of 1.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    # Simplified for our specific mosquito audio processing
    mean = np.mean(data)
    std = np.std(data) or 1.0  # Use 1.0 as fallback if std is 0
    return (data - mean) / std

def rms_normalization(data):
    """
    Normalize the audio data using root mean square (RMS) normalization.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    # Simplified for mosquito audio data
    rms = np.sqrt(np.mean(data**2)) or 1.0  # Fallback to 1.0 if rms is 0
    return data / rms

def log_normalization(data):
    """
    Apply logarithmic normalization to the audio data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    # Simplified for our specific task - avoid redundant checks
    offset = 1e-10
    sign = np.sign(data)
    log_data = sign * np.log1p(np.abs(data) + offset)
    # Rescale to [-1, 1] range
    max_val = np.max(np.abs(log_data)) or 1.0  # Fallback to 1.0 if max_val is 0
    return log_data / max_val

def percentile_normalization(data, lower_percentile=1, upper_percentile=99):
    """
    Normalize audio data based on percentiles to handle outliers.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
    lower_percentile : float
        Lower percentile for clipping, default 1.
    upper_percentile : float
        Upper percentile for clipping, default 99.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    
    # Clip the data to remove outliers
    clipped_data = np.clip(data, lower, upper)
    
    # Apply min-max normalization to the clipped data
    min_val = np.min(clipped_data)
    max_val = np.max(clipped_data)
    if max_val - min_val > 0:
        return (clipped_data - min_val) / (max_val - min_val)
    return clipped_data

def dynamic_range_compression(data, threshold=-20, ratio=4, attack=0.01, release=0.1):
    """
    Apply dynamic range compression to the audio data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to compress.
    threshold : float
        Threshold in dB, default -20.
    ratio : float
        Compression ratio, default 4.
    attack : float
        Attack time in seconds, default 0.01.
    release : float
        Release time in seconds, default 0.1.
        
    Returns:
    --------
    numpy.ndarray
        The compressed audio data.
    """
    # Convert to dB
    offset = 1e-10
    data_db = 20 * np.log10(np.abs(data) + offset)
    
    # Apply compression to values above threshold
    mask = data_db > threshold
    reduction = (data_db[mask] - threshold) * (1 - 1/ratio)
    data_db[mask] -= reduction
    
    # Convert back to linear scale
    compressed_data = np.sign(data) * (10 ** (data_db / 20))
    
    # Normalize to match original peak level
    if np.max(np.abs(data)) > 0:
        compressed_data *= (np.max(np.abs(data)) / np.max(np.abs(compressed_data)))
        
    return compressed_data

def energy_normalization(data):
    """
    Normalize the audio data to have unit energy.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    energy = np.sum(data**2)
    if energy > 0:
        return data / np.sqrt(energy)
    return data

def adaptive_normalization(data, window_size=1024, overlap=512):
    """
    Apply adaptive normalization to the audio data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The audio data to normalize.
    window_size : int
        Size of the window for local normalization, default 1024.
    overlap : int
        Overlap between windows, default 512.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    # Check if data is shorter than window size
    if len(data) <= window_size:
        return peak_normalization(data)
    
    # Calculate step size
    step = window_size - overlap
    num_windows = (len(data) - window_size) // step + 1
    
    # Initialize output array
    normalized_data = np.zeros_like(data)
    window_weights = np.zeros_like(data)
    
    # Apply triangular window as weight
    window = np.hanning(window_size)
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        
        # Get window data and normalize
        window_data = data[start:end]
        norm_window = peak_normalization(window_data)
        
        # Apply weighted window
        normalized_data[start:end] += norm_window * window
        window_weights[start:end] += window
    
    # Handle areas with zero weight
    mask = window_weights > 0
    normalized_data[mask] /= window_weights[mask]
    
    # Handle unprocessed ends if any
    if np.sum(window_weights == 0) > 0:
        zero_mask = window_weights == 0
        normalized_data[zero_mask] = data[zero_mask]
    
    return normalized_data

def normalize(audio: np.ndarray, normalize_method='rms') -> np.ndarray:
    """
    Normalize audio data using the specified method.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        The audio data to normalize.
    normalize_method : str
        The normalization method to use: 'peak', 'min_max', 'z_score', 
        'rms', 'log', 'percentile', 'dynamic', 'energy', or 'adaptive'.
        
    Returns:
    --------
    numpy.ndarray
        The normalized audio data.
    """
    try:
        if normalize_method == 'peak':
            return peak_normalization(audio)
        elif normalize_method == 'min_max':
            return min_max_normalization(audio)
        elif normalize_method == 'z_score':
            return z_score_normalization(audio)
        elif normalize_method == 'rms':
            return rms_normalization(audio)
        elif normalize_method == 'log':
            return log_normalization(audio)
        elif normalize_method == 'percentile':
            return percentile_normalization(audio)
        elif normalize_method == 'dynamic':
            return dynamic_range_compression(audio)
        elif normalize_method == 'energy':
            return energy_normalization(audio)
        elif normalize_method == 'adaptive':
            return adaptive_normalization(audio)
        else:
            logger.warning(f"Unknown normalization method: {normalize_method}. Using RMS normalization.")
            return rms_normalization(audio)
    except Exception as e:
        logger.error(f"Error in normalization: {e}. Returning original audio.")
        return audio
