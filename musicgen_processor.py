"""
Generate MusicGen-small takes (no loop), with silence-guard + normalization.
Tested on transformers==4.57.1 (no 'generator' kwarg).
"""

import argparse
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import torch
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


def float_to_int16(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def peak_normalize(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    if peak < 1e-12:
        return y
    target = 10 ** (target_dbfs / 20.0)
    gain = target / peak
    return np.clip(y * gain, -1.0, 1.0)


def to_mono_float(audio_tensor: torch.Tensor) -> np.ndarray:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--seed", type=int, action="append")
    # ap.add_argument("--seconds", type=float, default=28.0)

    args = ap.parse_args()

    output_seconds = 25
    seed = args.seed

    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

            # Silence guard
            peak = float(np.max(np.abs(y)))
            rms_db = dbfs(y)
            is_silent = (peak < SILENCE_PEAK_THRESH) or (rms_db < SILENCE_RMS_DBFS)

            # Normalize if not silent, so you can actually hear the bed
            if not is_silent:
                y = peak_normalize(y, NORMALIZE_DBFS)

            fname = f"take_{take_idx:02d}_g{p['guidance_scale']}_k{p['top_k']}_p{p['top_p']}_t{p['temperature']}_seed{seed}.wav"
            path = outdir / fname
            wav.write(path.as_posix(), sr, float_to_int16(y))

            take_idx += 1

    print(f"Done. Wrote {take_idx} files to {outdir}")


if __name__ == "__main__":
    main()
