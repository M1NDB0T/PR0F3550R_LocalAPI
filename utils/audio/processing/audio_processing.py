"""
Audio Processing and Model Inference Module

This module handles the neural network inference pipeline that converts audio features
into facial blendshape coefficients. It processes audio in overlapping chunks to ensure
smooth transitions and handles GPU optimization for real-time performance.

Key Functions:
    - decode_audio_chunk: Runs transformer model inference on audio feature chunks
    - process_audio_features: Orchestrates chunked processing with overlap blending
    - pad_audio_chunk: Handles edge cases for short audio segments
    - blend_chunks: Smoothly blends overlapping chunks to prevent discontinuities

The processing uses a sliding window approach with overlap to maintain temporal
continuity in the generated facial animation.

License:
    This software is licensed under a **dual-license model**
    For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
    Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
"""

import numpy as np
import torch
from torch.cuda.amp import autocast

def decode_audio_chunk(audio_chunk, model, device, config):
    """
    Run transformer model inference on a single audio feature chunk.
    
    This function processes one window of audio features through the encoder-decoder
    transformer architecture to generate blendshape coefficients for that time segment.
    
    Args:
        audio_chunk (numpy.ndarray): Audio features of shape (frame_length, num_features)
        model (torch.nn.Module): Loaded transformer model (Seq2Seq architecture)
        device (torch.device): Computation device (CPU or CUDA)
        config (dict): Configuration dictionary containing precision settings
    
    Returns:
        numpy.ndarray: Blendshape coefficients of shape (chunk_length, 68)
                     Each row contains 68 facial animation values
    
    Processing Flow:
        1. Convert audio features to PyTorch tensor
        2. Run through encoder (processes audio patterns)
        3. Run through decoder (generates facial coefficients)
        4. Convert back to numpy array for post-processing
    
    GPU Optimization:
        - Uses half-precision (float16) if enabled and CUDA available
        - Automatic mixed precision (AMP) for faster inference
        - No gradient computation (torch.no_grad) for inference mode
    """
    # Determine precision based on config and device capabilities
    use_half_precision = config.get("use_half_precision", True)
    dtype = torch.float16 if use_half_precision else torch.float32
    
    # Convert numpy array to PyTorch tensor and add batch dimension
    # Shape: (frame_length, num_features) → (1, frame_length, num_features)
    src_tensor = torch.tensor(audio_chunk, dtype=dtype).unsqueeze(0).to(device)

    # Inference mode: no gradients needed
    with torch.no_grad():
        if use_half_precision and device.type == 'cuda':
            # Use automatic mixed precision for faster GPU inference
            # This reduces memory usage and increases throughput
            with autocast(dtype=torch.float16):
                # Encoder: processes audio features into latent representation
                encoder_outputs = model.encoder(src_tensor)
                # Decoder: generates blendshape coefficients from encoder output
                output_sequence = model.decoder(encoder_outputs)
        else:
            # Full precision mode (CPU or when half-precision disabled)
            encoder_outputs = model.encoder(src_tensor)
            output_sequence = model.decoder(encoder_outputs)

        # Remove batch dimension and move to CPU for numpy conversion
        # Shape: (1, chunk_length, 68) → (chunk_length, 68)
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
    
    return decoded_outputs


def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def pad_audio_chunk(audio_chunk, frame_length, num_features, pad_mode='replicate'):
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        
        if pad_mode == 'reflect':
            padding = np.pad(
                audio_chunk,
                pad_width=((0, pad_length), (0, 0)),
                mode='reflect'
            )
            audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
        
        elif pad_mode == 'replicate':
            last_frame = audio_chunk[-1:]  
            replication = np.tile(last_frame, (pad_length, 1)) 
            audio_chunk = np.vstack((audio_chunk, replication))
        
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}. Choose 'reflect' or 'replicate'.")
    
    return audio_chunk


def blend_chunks(chunk1, chunk2, overlap):
    actual_overlap = min(overlap, len(chunk1), len(chunk2))
    if actual_overlap == 0:
        return np.vstack((chunk1, chunk2))
    
    blended_chunk = np.copy(chunk1)
    for i in range(actual_overlap):
        alpha = i / actual_overlap 
        blended_chunk[-actual_overlap + i] = (1 - alpha) * chunk1[-actual_overlap + i] + alpha * chunk2[i]
        
    return np.vstack((blended_chunk, chunk2[actual_overlap:]))

def process_audio_features(audio_features, model, device, config):
    """
    Process complete audio feature sequence through the transformer model.
    
    This function orchestrates chunked processing with overlap blending to ensure
    smooth transitions between processing windows. It handles the complete pipeline
    from raw audio features to final blendshape coefficients.
    
    Args:
        audio_features (numpy.ndarray): Complete audio features of shape (num_frames, num_features)
        model (torch.nn.Module): Loaded transformer model
        device (torch.device): Computation device
        config (dict): Configuration dictionary
    
    Returns:
        numpy.ndarray: Final blendshape coefficients of shape (num_frames, 68)
                     Ready for streaming to Unreal Engine via LiveLink
    
    Processing Strategy:
        1. Sliding window: Process audio in overlapping chunks (128 frames with 32 frame overlap)
        2. Blending: Smoothly blend overlapping regions to prevent discontinuities
        3. Post-processing: Normalize values, apply easing, zero unused blendshapes
        4. Edge handling: Process remaining frames that don't fit into full windows
    
    Post-Processing Steps:
        - Normalize ARKit blendshapes (divide by 100 to convert from percentage)
        - Apply easing at start to prevent sudden expression changes
        - Zero out unused blendshape channels (eye tracking, etc.)
    """
    # Extract processing parameters from config
    frame_length = config['frame_size']  # Window size: 128 frames
    overlap = config.get('overlap', 32)  # Overlap between windows: 32 frames
    num_features = audio_features.shape[1]  # Feature dimension: 256
    num_frames = audio_features.shape[0]  # Total number of frames
    
    all_decoded_outputs = []  # Store processed chunks
    model.eval()  # Set model to evaluation mode (no dropout, batch norm fixes)
    
    # Process audio in sliding windows with overlap
    start_idx = 0
    while start_idx < num_frames:
        # Calculate window boundaries
        end_idx = min(start_idx + frame_length, num_frames)
        audio_chunk = audio_features[start_idx:end_idx]
        
        # Pad chunk if it's shorter than frame_length (edge case)
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        
        # Run model inference on this chunk
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device, config)
        
        # Trim to actual chunk length (removes padding)
        decoded_outputs = decoded_outputs[:end_idx - start_idx]
        
        # Blend with previous chunk if overlap exists
        if all_decoded_outputs:
            last_chunk = all_decoded_outputs.pop()
            # Smoothly blend overlapping regions
            blended_chunk = blend_chunks(last_chunk, decoded_outputs, overlap)
            all_decoded_outputs.append(blended_chunk)
        else:
            # First chunk: no blending needed
            all_decoded_outputs.append(decoded_outputs)
        
        # Move window forward (accounting for overlap)
        start_idx += frame_length - overlap

    # Handle remaining frames that don't fit into full windows
    current_length = sum(len(chunk) for chunk in all_decoded_outputs)
    if current_length < num_frames:
        remaining_frames = num_frames - current_length
        final_chunk_start = num_frames - remaining_frames
        audio_chunk = audio_features[final_chunk_start:num_frames]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device, config)
        all_decoded_outputs.append(decoded_outputs[:remaining_frames])

    # Concatenate all chunks into final output
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)[:num_frames]
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    
    # Post-processing: Normalize ARKit blendshapes (model outputs 0-100, need 0-1)
    final_decoded_outputs[:, :61] /= 100  
    
    # Apply easing at the start to prevent sudden expression changes
    # Eases in over 0.1 seconds (6 frames at 60 FPS)
    ease_duration_frames = min(int(0.1 * 60), final_decoded_outputs.shape[0])
    easing_factors = np.linspace(0, 1, ease_duration_frames)[:, None]
    final_decoded_outputs[:ease_duration_frames] *= easing_factors
    
    # Zero out unused blendshape channels (eye tracking, etc.)
    final_decoded_outputs = zero_columns(final_decoded_outputs)

    return final_decoded_outputs


def zero_columns(data):
    columns_to_zero = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    modified_data = np.copy(data) 
    modified_data[:, columns_to_zero] = 0
    return modified_data
