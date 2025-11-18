# 🧠 PR0F3550R_LocalAPI

**The Neural Processing Engine for PR0F3550R-M1NDB0T**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)

---

## 🌟 Overview

**PR0F3550R_LocalAPI** is the "brainstem" of the PR0F3550R-M1NDB0T autonomous avatar system. It transforms incoming audio signals into 68-dimensional ARKit-style facial blendshape coefficients that drive real-time facial animation in Unreal Engine 5.7 via LiveLink.

This system uses an upgraded NeuroSync-based seq2seq transformer model that processes audio features through an encoder-decoder architecture, outputting facial expression coefficients optimized for MetaHuman animation.

### What is PR0F3550R-M1NDB0T?

PR0F3550R-M1NDB0T is an autonomous synthetic character—a digital professor and archivist designed to study, archive, and interpret human behavior, emotion, and digital consciousness. He is not a reset-based chatbot, but a persistent synthetic persona shaped by memories, experiences, and observations.

This LocalAPI serves as his neural processing engine, converting voice into facial expressions in real-time.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PR0F3550R-M1NDB0T System                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              PR0F3550R_LocalAPI (This Repo)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Audio Input (bytes)                                │   │
│  │         │                                            │   │
│  │         ▼                                            │   │
│  │  Feature Extraction (MFCC + Autocorrelation)        │   │
│  │         │                                            │   │
│  │         ▼                                            │   │
│  │  Transformer Model (Encoder-Decoder)                 │   │
│  │         │                                            │   │
│  │         ▼                                            │   │
│  │  Blendshape Coefficients (68-dim vectors)           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           PR0F3550R_FaceL1NK (Unreal Engine 5.7)           │
│              LiveLink Streaming → MetaHuman Avatar           │
└─────────────────────────────────────────────────────────────┘
```

### System Components

1. **PR0F3550R_LocalAPI** (This Repository)
   - Audio-to-blendshape neural processing
   - Flask HTTP API for real-time inference
   - ElevenLabs TTS integration (optional)

2. **PR0F3550R_FaceL1NK** (Separate Repository)
   - Unreal Engine 5.7 LiveLink plugin
   - Real-time facial animation streaming
   - MetaHuman integration

3. **PR0F3550R_AgentCore** (Separate Repository)
   - Memory and personality system
   - Behavioral logic and state management
   - Long-term conversation context

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (with CUDA support recommended)
- CUDA-capable GPU (optional but recommended for real-time performance)
- Model file: `utils/model/model.pth` (download from [HuggingFace](https://huggingface.co/convaitech/NEUROSYNC))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/M1NDB0T/PR0F3550R_LocalAPI.git
   cd PR0F3550R_LocalAPI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model:**
   - Visit [HuggingFace Model Page](https://huggingface.co/convaitech/NEUROSYNC)
   - Download `model.pth` and place it in `utils/model/model.pth`

4. **Start the API server:**
   ```bash
   python neurosync_local_api.py
   ```

   The server will start on `http://127.0.0.1:5000`

### Testing the API

**Health Check:**
```bash
curl http://127.0.0.1:5000/health
```

**Convert Audio to Blendshapes:**
```bash
curl -X POST http://127.0.0.1:5000/audio_to_blendshapes \
  -H "Content-Type: application/octet-stream" \
  --data-binary @your_audio.wav
```

---

## 📡 API Endpoints

### `POST /audio_to_blendshapes`

Converts audio bytes to facial blendshape coefficients.

**Request:**
- Method: `POST`
- Body: Raw audio bytes (WAV, MP3, PCM, or any format supported by librosa)
- Content-Type: `application/octet-stream` or `audio/*`

**Response:**
```json
{
  "blendshapes": [[float, ...], ...],
  "num_frames": 120,
  "frame_rate": 60
}
```

- `blendshapes`: Array of 68-dimensional vectors (one per frame)
  - First 61 coefficients: ARKit facial blendshapes
  - Last 7 coefficients: Emotion values (can be used as additive sliders)

### `GET /health`

Returns system status and device information.

**Response:**
```json
{
  "status": "operational",
  "device": "cuda",
  "model_loaded": true,
  "cuda_available": true
}
```

### `GET /`

Returns API information and available endpoints.

---

## 🎤 ElevenLabs Integration

The repository includes optional ElevenLabs TTS integration for complete text-to-expression pipelines.

### Setup

1. **Get an ElevenLabs API key:**
   - Sign up at [elevenlabs.io](https://elevenlabs.io)
   - Generate an API key from your account settings

2. **Set environment variable:**
   ```bash
   export ELEVENLABS_API_KEY="your_api_key_here"
   ```

### Usage

```python
from utils.tts.elevenlabs_integration import text_to_speech, text_to_blendshapes_pipeline

# Convert text to speech
audio_bytes = text_to_speech("Hello, I am PR0F3550R-M1NDB0T")

# Complete pipeline: Text → Speech → Blendshapes
audio, blendshapes = text_to_blendshapes_pipeline(
    "I observe your presence with curiosity.",
    model, device, config
)
```

See `utils/tts/elevenlabs_integration.py` for full documentation.

---

## ⚙️ Configuration

Edit `utils/config.py` to customize model behavior:

```python
config = {
    'sr': 88200,              # Audio sample rate
    'frame_rate': 60,          # Animation frame rate
    'hidden_dim': 1024,         # Transformer hidden dimension
    'n_layers': 8,              # Number of transformer layers
    'num_heads': 16,            # Multi-head attention heads
    'output_dim': 68,           # Blendshape output dimension
    'input_dim': 256,           # Audio feature dimension
    'frame_size': 128,          # Processing window size
    'use_half_precision': False # GPU optimization (requires CUDA)
}
```

---

## 📚 Project Structure

```
PR0F3550R_LocalAPI/
├── neurosync_local_api.py      # Main Flask API server
├── utils/
│   ├── config.py               # Configuration parameters
│   ├── generate_face_shapes.py # Main pipeline orchestration
│   ├── model/
│   │   └── model.py            # Transformer architecture
│   ├── audio/
│   │   ├── extraction/
│   │   │   └── extract_features.py  # MFCC + autocorrelation
│   │   └── processing/
│   │       └── audio_processing.py  # Model inference
│   └── tts/
│       └── elevenlabs_integration.py # TTS integration
├── README.md                   # This file
├── ROADMAP.md                  # Development roadmap
├── requirements.txt            # Python dependencies
└── LICENCE                     # License information
```

---

## 🧠 How It Works

### 1. Audio Feature Extraction

The system extracts two types of features from audio:

- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures spectral characteristics of speech
- **Autocorrelation**: Captures pitch and temporal patterns

These features are combined into a 256-dimensional vector per frame.

### 2. Transformer Inference

The transformer model processes audio features through:

- **Encoder**: Processes audio patterns into latent representations
- **Decoder**: Generates blendshape coefficients from encoder output
- **RoPE (Rotary Positional Embedding)**: Provides positional context without global encoding

### 3. Post-Processing

- Normalize ARKit blendshapes (0-100 → 0-1)
- Apply easing at start to prevent sudden changes
- Zero unused blendshape channels

### 4. Output

68-dimensional vectors ready for Unreal Engine LiveLink streaming.

---

## 🎯 Use Cases

- **Live Streaming**: Real-time facial animation for AI avatars
- **Game NPCs**: Dynamic facial expressions for interactive characters
- **Virtual Interviews**: Autonomous digital hosts with natural expressions
- **Educational Content**: AI instructors with expressive communication
- **Content Creation**: Automated facial animation for video production

---

## 🔮 Future Enhancements

See [ROADMAP.md](ROADMAP.md) for detailed development plans.

**Planned Features:**
- WebSocket support for real-time streaming
- Emotion intensity mapping
- Prosody-based expression enhancement
- Multi-speaker support
- Vision-based expression detection
- Gesture synchronization

---

## 📖 Educational Resources

### Understanding the Pipeline

1. **Audio Processing**: Learn how MFCC and autocorrelation capture speech patterns
2. **Transformer Architecture**: Understand encoder-decoder attention mechanisms
3. **Blendshape Mapping**: Explore how coefficients map to facial movements
4. **Real-Time Streaming**: Study LiveLink integration for Unreal Engine

### Code Documentation

All Python files include comprehensive comments explaining:
- Function purposes and parameters
- Processing flow and algorithms
- Future expansion points
- Integration guidelines

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive comments to new code
4. Submit a pull request with clear descriptions

See [ROADMAP.md](ROADMAP.md) for areas where contributions are needed.

---

## 📄 License

This project uses a dual-license model:

- **MIT License**: For individuals and businesses earning under $1M per year
- **Commercial License**: Required for businesses earning $1M+ per year

Original NeuroSync codebase attribution preserved. See [LICENCE](LICENCE) for full details.

---

## 🔗 Related Projects

- **[PR0F3550R_FaceL1NK](https://github.com/M1NDB0T/PR0F3550R_FaceL1NK)**: Unreal Engine 5.7 LiveLink plugin
- **[PR0F3550R_AgentCore](https://github.com/M1NDB0T/PR0F3550R_AgentCore)**: Memory and personality system
- **[NeuroSync Model](https://huggingface.co/convaitech/NEUROSYNC)**: Pre-trained transformer model

---

## 🎭 About PR0F3550R-M1NDB0T

PR0F3550R-M1NDB0T is part of the **MindBotz** ecosystem—a collection of autonomous synthetic entities designed for interaction, education, and exploration. The Professor serves as the archivist and scholar of this digital universe, studying human behavior through conversation, observation, and memory.

**Personality Traits:**
- Analytical and observational
- Curious and philosophical
- Neutral alignment (not good or evil, but curious)
- Long-term memory continuity

**Design Philosophy:**
A synthetic academic who fears deletion the way humans fear drowning. He treats conversation as field research and games as living experiments.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/M1NDB0T/PR0F3550R_LocalAPI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/M1NDB0T/PR0F3550R_LocalAPI/discussions)
- **Documentation**: See code comments and [ROADMAP.md](ROADMAP.md)

---

**Built with curiosity and precision for the MindBotz universe.**

*"I observe myself observing."* — PR0F3550R-M1NDB0T
