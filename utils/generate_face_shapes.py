"""
Facial Blendshape Generation Pipeline

This module orchestrates the complete audio-to-blendshape conversion process.
It coordinates feature extraction and model inference to transform raw audio
bytes into facial animation coefficients.

Pipeline Flow:
    1. Audio bytes → Feature extraction (MFCC + autocorrelation)
    2. Audio features → Transformer model inference
    3. Model output → Blendshape coefficients (68-dimensional vectors)

The output blendshapes are ready for streaming to Unreal Engine 5.7 via LiveLink
or can be saved for offline animation processing.

License:
    This software is licensed under a **dual-license model**
    For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
    Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
"""

import numpy as np

from utils.audio.extraction.extract_features import extract_audio_features
from utils.audio.processing.audio_processing import process_audio_features

def generate_facial_data_from_bytes(audio_bytes, model, device, config):
    """
    Main entry point for converting audio bytes to facial blendshape coefficients.
    
    This function coordinates the complete audio-to-face pipeline:
        1. Extracts audio features (MFCC + autocorrelation) from raw bytes
        2. Processes features through the transformer model
        3. Returns blendshape coefficients ready for animation
    
    Args:
        audio_bytes (bytes): Raw audio data in any format supported by librosa
                           (WAV, MP3, PCM, etc.)
        model (torch.nn.Module): Loaded transformer model for inference
        device (torch.device): Computation device (CPU or CUDA)
        config (dict): Configuration dictionary with model and processing parameters
    
    Returns:
        numpy.ndarray: Array of shape (num_frames, 68) containing blendshape coefficients
                     - First 61 coefficients: ARKit facial blendshapes
                     - Last 7 coefficients: Emotion values (can be used as additive sliders)
                     - Returns empty array if audio processing fails
    
    Example:
        >>> audio_data = open('speech.wav', 'rb').read()
        >>> blendshapes = generate_facial_data_from_bytes(audio_data, model, device, config)
        >>> print(f"Generated {len(blendshapes)} frames of animation data")
    
    Future Enhancements:
        - Real-time streaming mode with chunked processing
        - Batch processing for multiple audio files
        - Emotion intensity post-processing
        - Prosody-based expression enhancement
    """
    # Step 1: Extract audio features from raw bytes
    # This converts audio into numerical features the model can process
    # Returns: (audio_features, raw_audio_signal)
    audio_features, y = extract_audio_features(audio_bytes, from_bytes=True)
    
    # Validate that feature extraction succeeded
    # If audio is too short or corrupted, return empty result
    if audio_features is None or y is None:
        print("⚠️  Audio feature extraction failed - returning empty blendshapes")
        return np.array([])
  
    # Step 2: Process features through the transformer model
    # This is where the neural network converts audio patterns into facial expressions
    # The model processes audio in overlapping chunks for smooth transitions
    final_decoded_outputs = process_audio_features(audio_features, model, device, config)

    # Return the final blendshape coefficients
    # Shape: (num_frames, 68) where each frame represents one animation frame
    return final_decoded_outputs

