"""
Generate MusicGen-small takes (no loop), with silence-guard + normalization.
Tested on transformers==4.57.1 (no 'generator' kwarg).
Subprocess-ready script that generates multiple music takes.
"""

import argparse
from pathlib import Path
import sys
import logging
import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration, set_seed

MODEL_ID = "facebook/musicgen-small"

# Slightly looser than your old presets (fewer dead-silent outputs)
PRESETS = [
    dict(guidance_scale=2.6, top_k=160, top_p=0.95, temperature=1.00, typical_p=0.95),
    dict(guidance_scale=2.9, top_k=140, top_p=0.93, temperature=0.95, typical_p=0.95),
    dict(guidance_scale=2.4, top_k=125, top_p=0.96, temperature=1.05, typical_p=0.97),
]
TOKENS_PER_SEC = 50  # ~50 tok/s for musicgen-small @32kHz

SILENCE_PEAK_THRESH = 1e-4  # ~-80 dBFS
SILENCE_RMS_DBFS = -55.0  # reject if quieter than this
NORMALIZE_DBFS = -3.0  # peak target

# Set up logger
logger = logging.getLogger("musicgen_processor")


def dbfs(x: np.ndarray) -> float:
    """Calculate dBFS of audio signal."""
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def peak_normalize(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Peak normalize audio to target dBFS."""
    peak = float(np.max(np.abs(y)))
    if peak < 1e-12:
        return y
    target = 10 ** (target_dbfs / 20.0)
    gain = target / peak
    return np.clip(y * gain, -1.0, 1.0)


def to_mono_float(audio_tensor: torch.Tensor) -> np.ndarray:
    """Convert audio tensor to mono float numpy array in range [-1.0, 1.0]."""
    t = audio_tensor.detach().cpu()
    # Accept (B, L) OR (B, C, L)
    if t.ndim == 2:
        y = t[0].float().numpy()
    elif t.ndim == 3:
        y = t[0, 0].float().numpy()  # take first channel
    else:
        raise RuntimeError(f"Unexpected audio shape: {tuple(t.shape)}")
    return np.clip(y, -1.0, 1.0)


def main():
    """Main function for music generation subprocess"""
    ap = argparse.ArgumentParser(description="MusicGen Music Generation")
    ap.add_argument("--prompt", type=str, required=True, help="Text prompt for music generation")
    ap.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    ap.add_argument("--output_dir", type=str, default="output", help="Output directory for generated files")
    ap.add_argument("--seconds", type=float, default=31.0, help="Duration in seconds")

    args = ap.parse_args()

    # Configure logging to output to stdout for better subprocess visibility
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        logger.info("Starting MusicGen processing")
        logger.info("Prompt: %s", args.prompt)
        logger.info("Seed: %s", args.seed)
        logger.info("Duration: %ss", args.seconds)

        output_seconds = args.seconds
        seed = args.seed
        if seed > (2**31):
            raise ValueError("Seed must be between 0 and 2^31 - 1")

        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

        max_new_tokens = int(TOKENS_PER_SEC * output_seconds)
        inputs = proc(text=[args.prompt], padding=True, return_tensors="pt").to(device)

        take_idx = 0

        # Make CUDA less random-ish
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with torch.no_grad():
            for p in PRESETS:
                # transformers 4.57.1 ignores `generator`, so seed globally per take
                set_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))

                logger.info("Generating take %d/%d with preset: %s", take_idx + 1, len(PRESETS), p)

                audio = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=float(p["guidance_scale"]),
                    top_k=int(p["top_k"]),
                    top_p=float(p["top_p"]),
                    typical_p=float(p["typical_p"]),
                    temperature=float(p["temperature"]),
                    max_new_tokens=max_new_tokens,
                )
                sr = model.config.audio_encoder.sampling_rate  # 32000
                y = to_mono_float(audio)

                # Check audio levels before normalization
                peak = float(np.max(np.abs(y)))
                rms_db = dbfs(y)
                logger.info("Take %d - Peak: %.6f, RMS: %.2f dBFS", take_idx + 1, peak, rms_db)

                # Silence guard - check if truly silent (dead silence, not just quiet)
                is_dead_silent = peak < SILENCE_PEAK_THRESH

                if is_dead_silent:
                    logger.warning("Take %d is dead silent (peak < %s), skipping", take_idx + 1, SILENCE_PEAK_THRESH)
                    take_idx += 1
                    continue

                # Always normalize to make audio audible
                y = peak_normalize(y, NORMALIZE_DBFS)

                # Log after normalization
                peak_after = float(np.max(np.abs(y)))
                rms_db_after = dbfs(y)
                logger.info("After normalization - Peak: %.6f, RMS: %.2f dBFS", peak_after, rms_db_after)

                # Convert to torch tensor for torchaudio (expects 2D: [channels, samples])
                y_tensor = torch.from_numpy(y).float().unsqueeze(0)  # Add channel dimension

                fname = f"take_{take_idx:02d}_g{p['guidance_scale']}_k{p['top_k']}_p{p['top_p']}_t{p['temperature']}_seed{seed}.mp3"
                path = outdir / fname

                # Save as MP3 using torchaudio
                torchaudio.save(path.as_posix(), y_tensor, sr, format="mp3")
                logger.info("Saved: %s", fname)

                take_idx += 1

        logger.info("Done. Wrote %d files to %s", take_idx, outdir)
        print(f"SUCCESS: Generated {take_idx} music files in {outdir}")
        sys.exit(0)

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error during music generation: %s", e)
        print(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("Process interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unexpected error: %s", e)
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
