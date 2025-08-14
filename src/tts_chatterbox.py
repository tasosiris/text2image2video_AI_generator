import argparse
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import threading

try:
    import torch
except Exception as exc:  # pragma: no cover
    print("This script requires torch. Please install it first: pip install torch", file=sys.stderr)
    raise

try:
    import torchaudio
except Exception as exc:  # pragma: no cover
    print("This script requires torchaudio. Please install it first: pip install torchaudio", file=sys.stderr)
    raise


def slugify_filename(text: str, max_words: int = 8) -> str:
    words = text.strip().split()
    base = "-".join(w.lower() for w in words[:max_words])
    base = "".join(c for c in base if c.isalnum() or c in ("-", "_"))
    if not base:
        base = "tts"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base[:60]}-{timestamp}.mp3"


_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_or_load_model(resolved_device: str):
    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore
    except Exception as exc:
        print(
            "Missing dependency: chatterbox-tts. Install with: pip install chatterbox-tts",
            file=sys.stderr,
        )
        raise

    # Load once per device
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(resolved_device)
        if model is None:
            print("[TTS] Loading Chatterbox model (cached per device)...", flush=True)
            model = ChatterboxTTS.from_pretrained(device=resolved_device)
            _MODEL_CACHE[resolved_device] = model
    return model


def synthesize_with_chatterbox(
    text: str,
    device: Optional[str] = None,
    voice: Optional[str] = None,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    top_p: float = 1.0,
    min_p: float = 0.05,
    repetition_penalty: float = 1.2,
):
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[TTS] Using device: {resolved_device}", flush=True)
    # Load or reuse model (first run may download weights)
    model = _get_or_load_model(resolved_device)
    print("[TTS] Model ready (cached).", flush=True)

    # Generate waveform
    # API may accept voice/name if provided; pass only when set to avoid errors on older versions
    print(
        (
            f"[TTS] Generating speech (chars: {len(text)}"
            + (f", voice: {voice}" if voice else "")
            + (f", prompt: {audio_prompt_path}" if audio_prompt_path else "")
            + f", exaggeration: {exaggeration}, temperature: {temperature}, cfg_weight: {cfg_weight}, top_p: {top_p}, min_p: {min_p}, repetition_penalty: {repetition_penalty})..."
        ),
        flush=True,
    )
    # Note: ChatterboxTTS.generate does not accept a "voice" string. Reference voice is provided via audio_prompt_path.
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
    )
    print("[TTS] Generation complete.", flush=True)

    # Ensure torch tensor on CPU for saving
    if not torch.is_tensor(wav):
        wav = torch.tensor(wav)
    wav = wav.detach().cpu()

    sample_rate = getattr(model, "sr", 22050)
    try:
        shape_str = "x".join(str(d) for d in wav.shape)
    except Exception:
        shape_str = "unknown"
    print(f"[TTS] Waveform shape: {shape_str}, sample rate: {sample_rate}", flush=True)
    return wav, sample_rate


def save_mp3(output_path: str, wav: torch.Tensor, sample_rate: int) -> None:
    # torchaudio.save supports mp3 when FFmpeg is available. If unavailable, raise a helpful error.
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"[TTS] Saving MP3 to: {output_path}", flush=True)
        torchaudio.save(output_path, wav, sample_rate, format="mp3")
        print("[TTS] MP3 saved successfully.", flush=True)
    except Exception as exc:
        # Fallback: save WAV and inform the user
        wav_path = os.path.splitext(output_path)[0] + ".wav"
        print(
            "[TTS] MP3 save failed, attempting WAV fallback... (install FFmpeg for MP3 support)",
            flush=True,
        )
        torchaudio.save(wav_path, wav, sample_rate)
        msg = (
            f"Failed to save MP3 (saved WAV instead at {wav_path}). "
            "Install FFmpeg for MP3 support, or use pydub/ffmpeg-python to convert."
        )
        print(msg, file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate a TTS MP3 using Chatterbox and save to a folder.")
    parser.add_argument("text", help="The text to synthesize into speech.")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "tts"),
        help="Directory to save the MP3 file (default: outputs/tts)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Output filename (e.g., speech.mp3). Defaults to a slugified name based on the text.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional Chatterbox voice name/ID if supported by your installation.",
    )
    parser.add_argument(
        "--audio-prompt",
        default=None,
        help="Path to a short reference WAV to condition the voice (5-10s clean speech recommended).",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.4,
        help="Emotional expressiveness (e.g., 0.25 calm, 0.8 lively, 1.5+ very dramatic).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Variation/speed of delivery (lower = clearer/slower, higher = faster/more varied).",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Adherence to the reference voice/style when using an audio prompt (0.0-1.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (typ. 0.8–1.0).",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Minimum token probability cutoff (raises clarity by avoiding low-probability tokens).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty to reduce repeated artifacts (typ. 1.1–1.3).",
    )
    parser.add_argument(
        "--append-dot",
        action="store_true",
        help="Ensure a trailing period is present at the end of the text before synthesis.",
    )
    parser.add_argument(
        "--append-comma",
        action="store_true",
        help="Append a trailing comma to induce a short pause before the end.",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=300,
        help="Milliseconds of silence to append to the end of the audio (e.g., 300).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", None],
        help="Force device selection. Defaults to auto-detect.",
    )
    args = parser.parse_args()

    # Optionally enforce a trailing period
    input_text = args.text
    stripped = input_text.rstrip()
    # Optionally add a trailing comma for a short pause
    if args.append_comma and not stripped.endswith(","):
        input_text = stripped + ","
        stripped = input_text
        print("[TTS] Appending trailing comma to input text for a short pause.", flush=True)
    # Optionally add a trailing period to finalize the sentence
    if args.append_dot:
        stripped = input_text.rstrip()
        if not stripped.endswith((".", "!", "?")):
            input_text = stripped + "."
            print("[TTS] Appending trailing period to input text.", flush=True)

    filename = args.filename or slugify_filename(input_text)
    if not filename.lower().endswith(".mp3"):
        filename += ".mp3"

    output_path = os.path.join(args.out_dir, filename)

    print(f"[TTS] Output directory: {args.out_dir}", flush=True)
    print(f"[TTS] Target filename: {filename}", flush=True)

    wav, sr = synthesize_with_chatterbox(
        input_text,
        device=args.device,
        voice=args.voice,
        audio_prompt_path=args.audio_prompt,
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        cfg_weight=args.cfg_weight,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Ensure 2D tensor shape (channels, frames) for torchaudio.save
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # Optionally append silence at the end
    if args.end_silence_ms and args.end_silence_ms > 0:
        extra_samples = int(sr * (args.end_silence_ms / 1000.0))
        if extra_samples > 0:
            print(f"[TTS] Appending {args.end_silence_ms} ms ({extra_samples} samples) of silence.", flush=True)
            silence = torch.zeros((wav.shape[0], extra_samples), dtype=wav.dtype)
            wav = torch.cat([wav, silence], dim=1)

    try:
        save_mp3(output_path, wav, sr)
        print(f"Saved MP3 to {output_path}")
    except Exception:
        # save_mp3 already wrote a WAV and emitted an error message
        pass


if __name__ == "__main__":
    main()


