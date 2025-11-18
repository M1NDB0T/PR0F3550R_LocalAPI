"""
PR0F3550R_LocalAPI - The Neural Processing Engine for PR0F3550R-M1NDB0T

This module serves as the "brainstem" of the PR0F3550R-M1NDB0T autonomous avatar system.
It transforms incoming audio signals into 68-dimensional ARKit-style facial blendshape
coefficients that drive real-time facial animation in Unreal Engine 5.7 via LiveLink.

Architecture:
    Audio Input (bytes) → Feature Extraction → Transformer Inference → Blendshape Output
    
The system uses an upgraded NeuroSync-based seq2seq transformer model that processes
audio features through an encoder-decoder architecture, outputting facial expression
coefficients optimized for MetaHuman animation.

License:
    This software is licensed under a **dual-license model**
    For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
    Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
    
    Original NeuroSync codebase attribution preserved in LICENSE file.
"""

from flask import request, jsonify
import numpy as np
import torch
import flask

from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model
from utils.config import config

# ============================================================================
# Flask Application Initialization
# ============================================================================
# This Flask app serves as the HTTP/WebSocket endpoint for the facial animation pipeline.
# It receives audio data and returns blendshape coefficients for real-time streaming.
app = flask.Flask(__name__)

# ============================================================================
# Device Configuration
# ============================================================================
# Automatically detect and use GPU (CUDA) if available, otherwise fall back to CPU.
# GPU acceleration significantly improves inference speed for the transformer model.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🧠 PR0F3550R_LocalAPI: Activated device:", device)
print("📡 Initializing neural processing engine...")

# ============================================================================
# Model Loading
# ============================================================================
# Load the pre-trained NeuroSync transformer model from disk.
# This model converts audio features into facial blendshape coefficients.
# The model path should point to the downloaded model.pth file from HuggingFace.
model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)
print("✅ Model loaded successfully. Ready for inference.")

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    """
    Primary API endpoint for audio-to-blendshape conversion.
    
    This endpoint receives raw audio bytes (WAV, PCM, or other formats supported by librosa)
    and processes them through the complete pipeline:
        1. Audio feature extraction (MFCC + autocorrelation)
        2. Transformer model inference
        3. Blendshape coefficient generation
    
    Request:
        - Method: POST
        - Body: Raw audio bytes (any format supported by librosa)
        - Content-Type: application/octet-stream (or audio/*)
    
    Response:
        - JSON object containing:
            {
                "blendshapes": [[float, ...], ...]  # 68-dimensional vectors per frame
            }
        - Each inner array represents one frame of facial animation data
        - 61 coefficients correspond to ARKit blendshapes
        - 7 coefficients represent emotion values (can be used as additive sliders)
    
    Future Enhancements:
        - WebSocket support for real-time streaming
        - Batch processing for multiple audio files
        - Emotion intensity mapping
        - Prosody-based expression enhancement
    """
    try:
        # Extract raw audio bytes from the HTTP request
        audio_bytes = request.data
        
        if not audio_bytes:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Process audio through the complete pipeline:
        # Audio bytes → Feature extraction → Model inference → Blendshape coefficients
        generated_facial_data = generate_facial_data_from_bytes(
            audio_bytes, 
            blendshape_model, 
            device, 
            config
        )
        
        # Convert numpy array to Python list for JSON serialization
        # The output shape is (num_frames, 68) where each frame contains blendshape values
        generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

        return jsonify({
            'blendshapes': generated_facial_data_list,
            'num_frames': len(generated_facial_data_list),
            'frame_rate': config.get('frame_rate', 60)
        })
    
    except Exception as e:
        # Error handling for debugging and production monitoring
        print(f"❌ Error processing audio: {str(e)}")
        return jsonify({
            'error': 'Failed to process audio',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring and system status.
    
    Returns:
        JSON object with system status, device info, and model readiness.
    """
    return jsonify({
        'status': 'operational',
        'device': str(device),
        'model_loaded': blendshape_model is not None,
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint providing API information and documentation links.
    """
    return jsonify({
        'name': 'PR0F3550R_LocalAPI',
        'description': 'Neural processing engine for PR0F3550R-M1NDB0T facial animation',
        'version': '1.0.0',
        'endpoints': {
            '/audio_to_blendshapes': 'POST - Convert audio to blendshape coefficients',
            '/health': 'GET - System health check'
        },
        'documentation': 'See README.md for full documentation'
    })

# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == '__main__':
    """
    Start the Flask development server.
    
    In production, use a production WSGI server like Gunicorn or uWSGI.
    For real-time streaming, consider implementing WebSocket support via Flask-SocketIO.
    
    Default configuration:
        - Host: 127.0.0.1 (localhost only - change to '0.0.0.0' for network access)
        - Port: 5000
        - Debug: False (set to True for development)
    """
    print("🚀 Starting PR0F3550R_LocalAPI server...")
    print("📡 Server will be available at http://127.0.0.1:5000")
    print("💡 Use /health endpoint to check system status")
    app.run(host='127.0.0.1', port=5000, debug=False)
