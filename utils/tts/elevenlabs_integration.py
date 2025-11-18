"""
ElevenLabs Text-to-Speech Integration for PR0F3550R-M1NDB0T

This module provides integration with ElevenLabs' Text-to-Speech API, enabling
PR0F3550R-M1NDB0T to generate natural-sounding speech from text input. The generated
audio can then be fed back into the facial animation pipeline for complete
autonomous avatar behavior.

Features:
    - Text-to-speech conversion using ElevenLabs API
    - Voice customization and emotion control
    - Real-time streaming support (future)
    - Audio format conversion for facial animation pipeline
    - Error handling and retry logic

Usage:
    The TTS module works in conjunction with the LocalAPI:
        1. Text input → ElevenLabs API → Audio bytes
        2. Audio bytes → LocalAPI → Blendshape coefficients
        3. Blendshape coefficients → FaceL1NK → Unreal Engine animation

This creates a complete text-to-expression pipeline for autonomous avatar behavior.
"""

import requests
import os
from typing import Optional, Dict, Any
import io

# ============================================================================
# Configuration
# ============================================================================

# ElevenLabs API endpoint
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Default voice settings optimized for PR0F3550R-M1NDB0T's academic persona
# These can be customized to match the Professor's personality
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,  # Balance between consistency and expressiveness
    "similarity_boost": 0.8,  # How closely to match the original voice
    "style": 0.0,  # Voice style variation (0.0 = neutral)
    "use_speaker_boost": True  # Enhance speaker characteristics
}

# Default model for multilingual support
DEFAULT_MODEL_ID = "eleven_multilingual_v2"


# ============================================================================
# Core TTS Functions
# ============================================================================

def get_api_key() -> Optional[str]:
    """
    Retrieve ElevenLabs API key from environment variable.
    
    Set the API key in your environment:
        export ELEVENLABS_API_KEY="your_api_key_here"
    
    Or create a .env file with:
        ELEVENLABS_API_KEY=your_api_key_here
    
    Returns:
        str: API key if found, None otherwise
    """
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("⚠️  ELEVENLABS_API_KEY not found in environment variables")
        print("💡 Set it with: export ELEVENLABS_API_KEY='your_key'")
    return api_key


def text_to_speech(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Default voice - can be customized
    model_id: str = DEFAULT_MODEL_ID,
    voice_settings: Dict[str, Any] = None,
    api_key: Optional[str] = None
) -> Optional[bytes]:
    """
    Convert text to speech using ElevenLabs API.
    
    This function sends text to the ElevenLabs API and returns the generated
    audio as bytes, ready for processing by the facial animation pipeline.
    
    Args:
        text (str): Text to convert to speech
        voice_id (str): ElevenLabs voice ID (default: Rachel voice)
                       Get voice IDs from: https://elevenlabs.io/app/voices
        model_id (str): Model to use for generation
                       Options: "eleven_multilingual_v2", "eleven_monolingual_v1", etc.
        voice_settings (dict): Voice customization parameters
                              If None, uses DEFAULT_VOICE_SETTINGS
        api_key (str, optional): API key (if not provided, reads from environment)
    
    Returns:
        bytes: Audio data in MP3 format, ready for facial animation processing
              Returns None if generation fails
    
    Example:
        >>> audio_bytes = text_to_speech("Hello, I am PR0F3550R-M1NDB0T")
        >>> if audio_bytes:
        ...     # Feed audio_bytes to LocalAPI for blendshape generation
        ...     blendshapes = generate_facial_data_from_bytes(audio_bytes, model, device, config)
    
    Future Enhancements:
        - Streaming mode for real-time TTS
        - Emotion-based voice modulation
        - Prosody control for expression enhancement
        - Caching for repeated phrases
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = get_api_key()
        if not api_key:
            return None
    
    # Use default voice settings if none provided
    if voice_settings is None:
        voice_settings = DEFAULT_VOICE_SETTINGS.copy()
    
    # Construct API endpoint URL
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
    
    # Prepare request headers
    headers = {
        "Accept": "audio/mpeg",  # Request MP3 format
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    # Prepare request payload
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": voice_settings
    }
    
    try:
        # Send POST request to ElevenLabs API
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        # Check for successful response
        if response.status_code == 200:
            print(f"✅ Generated speech for text: '{text[:50]}...'")
            return response.content  # Return audio bytes
        
        # Handle API errors
        else:
            print(f"❌ ElevenLabs API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error connecting to ElevenLabs API: {str(e)}")
        return None


def text_to_speech_file(
    text: str,
    output_path: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = DEFAULT_MODEL_ID,
    voice_settings: Dict[str, Any] = None,
    api_key: Optional[str] = None
) -> bool:
    """
    Convert text to speech and save to file.
    
    Convenience function that combines text_to_speech() with file saving.
    
    Args:
        text (str): Text to convert to speech
        output_path (str): Path to save the audio file (e.g., "output.mp3")
        voice_id (str): ElevenLabs voice ID
        model_id (str): Model to use for generation
        voice_settings (dict): Voice customization parameters
        api_key (str, optional): API key
    
    Returns:
        bool: True if successful, False otherwise
    """
    audio_bytes = text_to_speech(text, voice_id, model_id, voice_settings, api_key)
    
    if audio_bytes:
        try:
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"💾 Saved audio to: {output_path}")
            return True
        except IOError as e:
            print(f"❌ Error saving audio file: {str(e)}")
            return False
    
    return False


# ============================================================================
# Voice Management
# ============================================================================

def list_voices(api_key: Optional[str] = None) -> Optional[list]:
    """
    Retrieve list of available voices from ElevenLabs.
    
    Useful for finding voice IDs that match PR0F3550R-M1NDB0T's personality.
    
    Args:
        api_key (str, optional): API key (if not provided, reads from environment)
    
    Returns:
        list: List of voice dictionaries with voice_id, name, and metadata
              Returns None if request fails
    """
    if api_key is None:
        api_key = get_api_key()
        if not api_key:
            return None
    
    url = f"{ELEVENLABS_API_BASE}/voices"
    headers = {"xi-api-key": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('voices', [])
        else:
            print(f"❌ Error fetching voices: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return None


# ============================================================================
# Integration Helper Functions
# ============================================================================

def text_to_blendshapes_pipeline(
    text: str,
    model,
    device,
    config,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    api_key: Optional[str] = None
):
    """
    Complete pipeline: Text → Speech → Blendshapes
    
    This function combines TTS and facial animation in one call, making it
    easy to generate facial expressions directly from text input.
    
    Args:
        text (str): Text to convert
        model: Loaded transformer model for blendshape generation
        device: Computation device
        config: Configuration dictionary
        voice_id (str): ElevenLabs voice ID
        api_key (str, optional): ElevenLabs API key
    
    Returns:
        tuple: (audio_bytes, blendshapes) or (None, None) if failed
    
    Example:
        >>> audio, blendshapes = text_to_blendshapes_pipeline(
        ...     "I observe your presence with curiosity.",
        ...     model, device, config
        ... )
    """
    # Step 1: Convert text to speech
    audio_bytes = text_to_speech(text, voice_id=voice_id, api_key=api_key)
    
    if not audio_bytes:
        return None, None
    
    # Step 2: Convert audio to blendshapes
    from utils.generate_face_shapes import generate_facial_data_from_bytes
    blendshapes = generate_facial_data_from_bytes(audio_bytes, model, device, config)
    
    return audio_bytes, blendshapes

