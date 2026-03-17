"""Fine-tune F5-TTS on a custom voice using the MLX trainer."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from f5_tts_mlx import F5TTS
from f5_tts_mlx.trainer import F5TTSTrainer
from f5_tts_mlx.audio import log_mel_spectrogram
from f5_tts_mlx.data import vocab
from f5_tts_mlx.utils import list_str_to_idx


def load_training_data(data_dir: Path, max_duration: float = 30.0):
    """Load .wav + .normalized.txt pairs, convert to mel spectrograms.

    Yields batches (batch_size=1) with keys: mel_spec, mel_len, transcript.
    Loops forever for training.
    """
    pairs = []
    for wav_file in sorted(data_dir.glob("*.wav")):
        txt_file = wav_file.with_suffix(".normalized.txt")
        if not txt_file.exists():
            continue

        audio, sr = sf.read(str(wav_file))
        if sr != 24000:
            print(f"  Skipping {wav_file.name}: sample rate {sr} != 24000")
            continue

        duration = len(audio) / sr
        if duration > max_duration:
            print(f"  Skipping {wav_file.name}: {duration:.1f}s > {max_duration}s")
            continue

        text = txt_file.read_text().strip()
        transcript = list_str_to_idx(text, vocab)

        # Convert to mel spectrogram
        audio_mx = mx.array(audio)
        mel = log_mel_spectrogram(audio_mx)

        pairs.append({
            "name": wav_file.name,
            "mel_spec": np.array(mel),
            "mel_len": mel.shape[1],
            "transcript": np.array(transcript),
        })
        print(f"  Loaded {wav_file.name}: {duration:.1f}s, mel={mel.shape}")

    if not pairs:
        raise ValueError(f"No valid .wav + .normalized.txt pairs found in {data_dir}")

    print(f"Loaded {len(pairs)} segments")

    # Yield batches forever (trainer loops over this)
    while True:
        for sample in pairs:
            yield {
                "mel_spec": sample["mel_spec"][np.newaxis],  # add batch dim
                "mel_len": np.array([sample["mel_len"]]),
                "transcript": sample["transcript"][np.newaxis],  # add batch dim
            }


def main():
    parser = argparse.ArgumentParser(description="Train F5-TTS on custom voice data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.home() / ".mops-voice" / "training_data"),
        help="Directory with .wav + .normalized.txt pairs",
    )
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--sample-every", type=int, default=500, help="Generate sample every N steps")
    parser.add_argument("--resume", type=int, default=None, help="Resume from checkpoint step (e.g. --resume 1000)")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load pre-trained model
    print("Loading pre-trained F5-TTS model...")
    model = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")

    # Load dataset
    print(f"Loading data from {data_dir}:")
    dataset = load_training_data(data_dir)

    # Create trainer
    trainer = F5TTSTrainer(
        model,
        num_warmup_steps=200,
        log_with_wandb=args.wandb,
    )

    # Use first segment as reference for sample generation
    ref_audio = str(data_dir / "segment_01.wav")
    ref_text = (data_dir / "segment_01.normalized.txt").read_text().strip()

    resume_step = args.resume
    if resume_step:
        print(f"\nResuming from checkpoint step {resume_step}")
    print(f"Training for {args.steps} steps (lr={args.lr})")
    print(f"Saving every {args.save_every} steps, sampling every {args.sample_every} steps")
    print(f"Checkpoints saved to: results/")
    print()

    trainer.train(
        train_dataset=dataset,
        learning_rate=args.lr,
        total_steps=args.steps,
        save_every=args.save_every,
        sample_every=args.sample_every,
        sample_reference_audio=ref_audio,
        sample_reference_text=ref_text,
        sample_generation_text="Hello Fran, I'm MOPS, your digital fabrication assistant.",
        checkpoint=resume_step,
    )

    print("\nTraining complete! Checkpoints saved to results/")
    print("To use the fine-tuned model, update tts.py to load from the checkpoint.")


if __name__ == "__main__":
    main()
