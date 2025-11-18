# 🚀 Quick Start Guide

Get PR0F3550R_LocalAPI running in 5 minutes!

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required:**
- Python 3.8+
- PyTorch 2.0+
- Flask
- librosa
- numpy

---

## Step 2: Download the Model

1. Visit: https://huggingface.co/convaitech/NEUROSYNC
2. Download `model.pth`
3. Place it in: `utils/model/model.pth`

---

## Step 3: Start the Server

```bash
python neurosync_local_api.py
```

You should see:
```
🧠 PR0F3550R_LocalAPI: Activated device: cuda
📡 Initializing neural processing engine...
✅ Model loaded successfully. Ready for inference.
🚀 Starting PR0F3550R_LocalAPI server...
📡 Server will be available at http://127.0.0.1:5000
```

---

## Step 4: Test the API

**Health Check:**
```bash
curl http://127.0.0.1:5000/health
```

**Convert Audio:**
```bash
curl -X POST http://127.0.0.1:5000/audio_to_blendshapes \
  -H "Content-Type: application/octet-stream" \
  --data-binary @your_audio.wav
```

---

## Step 5: (Optional) ElevenLabs TTS

1. Get API key from https://elevenlabs.io
2. Set environment variable:
   ```bash
   export ELEVENLABS_API_KEY="your_key_here"
   ```
3. Use in Python:
   ```python
   from utils.tts.elevenlabs_integration import text_to_speech
   audio = text_to_speech("Hello, I am PR0F3550R-M1NDB0T")
   ```

---

## 🎉 You're Ready!

The API is now running and ready to convert audio to facial blendshapes.

**Next Steps:**
- Connect to PR0F3550R_FaceL1NK in Unreal Engine
- Integrate with PR0F3550R_AgentCore for autonomous behavior
- See [README.md](README.md) for full documentation

---

## Troubleshooting

**Model not found:**
- Ensure `utils/model/model.pth` exists
- Download from HuggingFace if missing

**CUDA errors:**
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Set `use_half_precision: False` in `utils/config.py` if issues persist

**Audio processing fails:**
- Ensure audio file is valid (WAV, MP3, etc.)
- Check audio length (minimum ~0.15 seconds)
- Verify librosa can read the format

---

**Need help?** Check [README.md](README.md) or open an issue!

