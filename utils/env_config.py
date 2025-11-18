"""
Environment Configuration Management

This module provides centralized configuration management using environment variables.
It allows users to configure API keys, server settings, and model parameters without
modifying code files.

Usage:
    Set environment variables in a .env file or system environment:
        ELEVENLABS_API_KEY=your_key_here
        API_PORT=5000
    
    Or use python-dotenv to load from .env file:
        from dotenv import load_dotenv
        load_dotenv()
"""

import os
from typing import Optional

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env file loading
    pass


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.
    
    Args:
        key: Environment variable name
        default: Default value if not found
    
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_elevenlabs_api_key() -> Optional[str]:
    """Get ElevenLabs API key from environment."""
    return get_env('ELEVENLABS_API_KEY')


def get_api_config() -> dict:
    """
    Get API server configuration from environment.
    
    Returns:
        Dictionary with host, port, and debug settings
    """
    return {
        'host': get_env('API_HOST', '127.0.0.1'),
        'port': int(get_env('API_PORT', '5000')),
        'debug': get_env('API_DEBUG', 'False').lower() == 'true'
    }


def get_model_config() -> dict:
    """
    Get model configuration overrides from environment.
    
    Returns:
        Dictionary with model configuration overrides
    """
    config = {}
    
    use_half_precision = get_env('USE_HALF_PRECISION')
    if use_half_precision:
        config['use_half_precision'] = use_half_precision.lower() == 'true'
    
    cuda_device = get_env('CUDA_DEVICE')
    if cuda_device:
        config['cuda_device'] = int(cuda_device)
    
    return config


def get_logging_config() -> dict:
    """
    Get logging configuration from environment.
    
    Returns:
        Dictionary with logging settings
    """
    return {
        'level': get_env('LOG_LEVEL', 'INFO'),
        'file': get_env('LOG_FILE')
    }

