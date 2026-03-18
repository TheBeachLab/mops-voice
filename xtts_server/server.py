"""Standalone XTTS TTS server with fine-tuned TARS voice.

Loads the Pyrater/TARS model (fine-tuned on Interstellar TARS voice).
Pre-computes speaker latents at startup for fast inference.

Usage:
    cd xtts_server
    .venv/bin/python server.py

API:
    POST /synthesize  {"text": "Hello Fran"}  → WAV audio bytes
    GET  /health                               → {"status": "ok"}
"""

import io
import json
import os

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, Response
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = Flask(__name__)
model = None
gpt_cond_latent = None
speaker_embedding = None


def load_model():
    global model, gpt_cond_latent, speaker_embedding

    print("Downloading TARS model from HuggingFace...")
    model_dir = snapshot_download("Pyrater/TARS")

    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))

    print("Loading fine-tuned XTTS model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir)
    model.training = False
    for param in model.parameters():
        param.requires_grad = False

    # Pre-compute speaker latents from reference audio (done once)
    ref_wav = os.path.join(model_dir, "reference.wav")
    print("Computing TARS speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[ref_wav]
    )

    if torch.backends.mps.is_available():
        model = model.to("mps")
        gpt_cond_latent = gpt_cond_latent.to("mps")
        speaker_embedding = speaker_embedding.to("mps")
        print("Using Apple Silicon MPS acceleration")

    print("TARS voice ready!")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return Response("No text provided", status=400)

    try:
        out = model.inference(
            text=text,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )

        wav = out["wav"]
        buf = io.BytesIO()
        sf.write(buf, np.array(wav), 24000, format="WAV")
        buf.seek(0)
        return Response(buf.read(), mimetype="audio/wav")
    except Exception as e:
        return Response(f"Synthesis error: {e}", status=500)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": "Pyrater/TARS", "voice": "TARS (Interstellar)"}


if __name__ == "__main__":
    load_model()
    print("\nXTTS server running on http://localhost:8090")
    app.run(host="127.0.0.1", port=8090, debug=False)
