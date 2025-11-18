"""
ElevenLabs TTS Integration Module

This package provides text-to-speech capabilities for PR0F3550R-M1NDB0T.
"""

from .elevenlabs_integration import (
    text_to_speech,
    text_to_speech_file,
    text_to_blendshapes_pipeline,
    list_voices,
    get_api_key
)

__all__ = [
    'text_to_speech',
    'text_to_speech_file',
    'text_to_blendshapes_pipeline',
    'list_voices',
    'get_api_key'
]

