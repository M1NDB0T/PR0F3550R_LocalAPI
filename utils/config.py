"""
Configuration parameters for the PR0F3550R_LocalAPI neural processing engine.

This configuration file defines the hyperparameters and system settings for the
transformer model that converts audio features into facial blendshape coefficients.

Key Parameters:
    - sr: Sample rate for audio processing (88200 Hz for high-quality feature extraction)
    - frame_rate: Target animation frame rate (60 FPS for smooth real-time animation)
    - hidden_dim: Transformer hidden dimension (1024 for balanced performance/quality)
    - n_layers: Number of transformer encoder/decoder layers (8 layers)
    - num_heads: Multi-head attention heads (16 heads for rich attention patterns)
    - output_dim: Number of blendshape coefficients (68: 61 ARKit + 7 emotion)
    - input_dim: Audio feature dimension (256 after MFCC + autocorrelation combination)
    - frame_size: Processing window size in frames (128 frames per chunk)
    - use_half_precision: Use float16 for GPU memory optimization (requires CUDA)

Future Expansion:
    - Emotion intensity scaling factors
    - Prosody detection thresholds
    - Real-time streaming buffer sizes
    - Multi-speaker support parameters
"""

config = {
    # Audio Processing Configuration
    'sr': 88200,  # Sample rate in Hz - high sample rate for detailed feature extraction
    'frame_rate': 60,  # Target animation frame rate (FPS) - matches Unreal Engine default
    
    # Transformer Architecture Configuration
    'hidden_dim': 1024,  # Hidden dimension of transformer layers - larger = more capacity
    'n_layers': 8,  # Number of encoder/decoder layers - deeper = more complex patterns
    'num_heads': 16,  # Multi-head attention heads - more heads = richer attention
    'dropout': 0.0,  # Dropout rate (0.0 = no dropout, used during inference)
    
    # Model I/O Dimensions
    'output_dim': 68,  # Blendshape output dimension: 61 ARKit + 7 emotion coefficients
                       # Note: If training your own model, this can be set to 61 for ARKit-only
    'input_dim': 256,  # Combined audio feature dimension (MFCC + autocorrelation features)
    
    # Processing Configuration
    'frame_size': 128,  # Processing window size in frames - larger = more context, slower
    'use_half_precision': False,  # Use float16 for GPU (requires CUDA + cuDNN)
                                   # Enables faster inference and lower memory usage on compatible GPUs
}
