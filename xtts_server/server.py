"""Standalone XTTS TTS server. Runs in its own Python 3.11 venv.

Loads the Pyrater/TARS model from HuggingFace and serves TTS via HTTP.
The main mops-voice app calls this server instead of F5-TTS.

Usage:
    cd xtts_server
    uv run --python 3.11 server.py

API:
    POST /synthesize  {"text": "Hello Fran"}
    Returns: WAV audio bytes
"""

import io
import json
import os
import sys

import soundfile as sf
import torch
from flask import Flask, request, Response
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = Flask(__name__)
model = None
MODEL_DIR = None


def load_model():
    global model, MODEL_DIR

    # Download Pyrater/TARS model from HuggingFace
    print("Downloading TARS model from HuggingFace...")
    MODEL_DIR = snapshot_download("Pyrater/TARS")
    print(f"Model downloaded to: {MODEL_DIR}")

    # Load config
    config_path = os.path.join(MODEL_DIR, "config.json")
    config = XttsConfig()
    config.load_json(config_path)

    # Load model
    print("Loading XTTS model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR)

    # model.eval() sets the model to inference mode (not Python's eval)
    model.training = False
    for param in model.parameters():
        param.requires_grad = False

    # Use MPS (Apple Silicon) if available
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Using Apple Silicon MPS acceleration")
    else:
        print("Using CPU")

    print("TARS voice model ready!")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return Response("No text provided", status=400)

    # Synthesize speech using the bundled reference audio
    import os
    ref_wav = os.path.join(MODEL_DIR, "reference.wav")
    outputs = model.synthesize(
        text,
        config=model.config,
        speaker_wav=ref_wav,
        language="en",
    )

    # Convert to WAV bytes
    audio = outputs["wav"]
    buf = io.BytesIO()
    sf.write(buf, audio, model.config.audio.sample_rate, format="WAV")
    buf.seek(0)

    return Response(buf.read(), mimetype="audio/wav")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": "Pyrater/TARS"}


if __name__ == "__main__":
    load_model()
    print("\nXTTS server running on http://localhost:8090")
    app.run(host="127.0.0.1", port=8090, debug=False)
