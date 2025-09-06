"""
Reusable narration tool wrapping Chatterbox TTS with configurable settings.

Usage:
    from tools.narration import NarrationSettings, narrate_sentence, narrate_phrases

    settings = NarrationSettings(
        device="cuda:0",                 # or "cpu"
        voice="alloy",                  # optional preset voice name if supported by your backend
        voice_sample_path="path/to/voice_sample.wav",  # optional reference audio for cloning
        exaggeration=0.4,
        temperature=0.5,
        cfg_weight=0.5,
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=1.2,
        end_silence_ms=300,
    )

    narrate_sentence("Hello world", "./outputs/narration/example.mp3", settings)
    narrate_phrases(["One", "Two"], "./outputs", settings=settings, max_workers=3)
"""

from __future__ import annotations

import os
import concurrent.futures
from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Optional, Sequence

import torch

from src.tts_chatterbox import synthesize_with_chatterbox, save_mp3
import re


@dataclass(frozen=True)
class NarrationSettings:
    """Configuration for narration generation.

    - device: e.g., "cuda:0" or "cpu"
    - voice: preset voice name if supported by the TTS backend
    - voice_sample_path: path to a reference audio to guide the voice (audio prompt)
    - end_silence_ms: extra silence appended to the end of each clip
    Other fields map directly to the underlying TTS generation params.
    """

    # device: 'auto' picks GPU if available else CPU; 'cuda' maps to 'cuda:0'
    device: Optional[str] = None
    voice: Optional[str] = None
    voice_sample_path: Optional[str] = None

    exaggeration: float = 0.4
    temperature: float = 0.5
    cfg_weight: float = 0.5
    top_p: float = 1.0
    min_p: float = 0.05
    repetition_penalty: float = 1.2
    end_silence_ms: int = 300


def _append_silence(waveform: torch.Tensor, sample_rate: int, end_silence_ms: int) -> torch.Tensor:
    if end_silence_ms <= 0:
        return waveform
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    extra = int(sample_rate * (end_silence_ms / 1000.0))
    if extra <= 0:
        return waveform
    silence = torch.zeros((waveform.shape[0], extra), dtype=waveform.dtype)
    return torch.cat([waveform, silence], dim=1)


def _normalize_device(device: Optional[str]) -> Optional[str]:
    if device is None or device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def _should_fallback_to_cpu(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "no kernel image is available" in msg
        or "torch not compiled with cuda" in msg
        or "sm_" in msg and "not compatible" in msg
    )


def _synthesize_waveform(text: str, settings: NarrationSettings) -> tuple[torch.Tensor, int]:
    effective_device = _normalize_device(settings.device)
    try:
        wav, sr = synthesize_with_chatterbox(
            text,
            device=effective_device,
            voice=settings.voice,
            audio_prompt_path=settings.voice_sample_path,
            exaggeration=settings.exaggeration,
            temperature=settings.temperature,
            cfg_weight=settings.cfg_weight,
            top_p=settings.top_p,
            min_p=settings.min_p,
            repetition_penalty=settings.repetition_penalty,
        )
        return wav, sr
    except Exception as exc:
        # Graceful CPU fallback for CUDA arch/build issues
        if effective_device and effective_device.startswith("cuda") and _should_fallback_to_cpu(exc):
            print("[Narration] CUDA issue detected; retrying on CPU...", flush=True)
            wav, sr = synthesize_with_chatterbox(
                text,
                device="cpu",
                voice=settings.voice,
                audio_prompt_path=settings.voice_sample_path,
                exaggeration=settings.exaggeration,
                temperature=settings.temperature,
                cfg_weight=settings.cfg_weight,
                top_p=settings.top_p,
                min_p=settings.min_p,
                repetition_penalty=settings.repetition_penalty,
            )
            print("[Narration] Using CPU fallback.", flush=True)
            return wav, sr
        raise


def _synthesize_and_save(
    text: str,
    output_path: str,
    settings: NarrationSettings,
) -> Optional[str]:
    try:
        wav, sr = _synthesize_waveform(text, settings)
        wav = _append_silence(wav, sr, settings.end_silence_ms)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_mp3(output_path, wav, sr)
        return output_path
    except Exception as exc:  # noqa: BLE001 - allow broad to surface errors
        print(f"Narration error: {exc}")
        return None


def _with_overrides(settings: Optional[NarrationSettings], overrides: dict[str, Any]) -> NarrationSettings:
    base = settings or NarrationSettings()
    if not overrides:
        return base
    # Only apply known fields
    valid_fields = set(NarrationSettings.__annotations__.keys())
    filtered = {k: v for k, v in overrides.items() if k in valid_fields}
    return replace(base, **filtered)


def narrate_sentence(
    sentence: str,
    output_path: str,
    settings: Optional[NarrationSettings] = None,
    **settings_overrides: Any,
) -> Optional[str]:
    """Synthesize a single sentence and save to an MP3 file.

    settings_overrides can provide any field from NarrationSettings as keyword args.
    Example: narrate_sentence(text, path, temperature=0.6, voice_sample_path="sample.wav")
    """
    final_settings = _with_overrides(settings, settings_overrides)
    return _synthesize_and_save(sentence, output_path, final_settings)


def narrate_phrases(
    phrases: Sequence[str] | Iterable[str],
    output_dir: str,
    settings: Optional[NarrationSettings] = None,
    *,
    max_workers: int = 3,
    force_regeneration: bool = False,
    filename_prefix: str = "narration_",
    start_index: int = 1,
    create_narration_subdir: bool = True,
    **settings_overrides: Any,
) -> List[Optional[str]]:
    """Synthesize many phrases concurrently and save as MP3 files.

    Returns a list of output file paths (or None for failures) in submission order.
    """
    final_settings = _with_overrides(settings, settings_overrides)

    if create_narration_subdir:
        narration_output_dir = os.path.join(output_dir, "narration")
        os.makedirs(narration_output_dir, exist_ok=True)
    else:
        narration_output_dir = output_dir
        os.makedirs(narration_output_dir, exist_ok=True)

    phrases_list = list(phrases)
    if not phrases_list:
        print("No phrases provided to narrate.")
        return []

    print(f"Narrating {len(phrases_list)} phrases.")

    jobs: list[tuple[int, str, str]] = []
    for idx, phrase in enumerate(phrases_list, start=start_index):
        filename = f"{filename_prefix}{idx:03d}.mp3"
        output_path = os.path.join(narration_output_dir, filename)
        if not force_regeneration and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"  -> Narration for phrase {idx} already exists. Skipping.")
            continue
        jobs.append((idx, phrase, output_path))

    results: List[Optional[str]] = [None] * len(jobs)
    index_map = {job_idx: i for i, (job_idx, _, __) in enumerate(jobs)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="narr") as executor:
        futures: list[concurrent.futures.Future[Optional[str]]] = []
        for job_idx, phrase, output_path in jobs:
            futures.append(
                executor.submit(_synthesize_and_save, phrase, output_path, final_settings)
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                result_path = future.result()
                # Maintain submission order using index_map
                # Map back from output path to the job index by peeking from the future's args is non-trivial;
                # Build a simple mapping by matching completed results in the order of submission.
                # Here we fill the next available slot sequentially.
                # For clarity and determinism, recompute positions from the futures list.
            except Exception as exc:  # noqa: BLE001
                print(f"Task generated an exception: {exc}")

        # Second pass to retrieve results in submission order
        for i, (job_idx, _phrase, output_path) in enumerate(jobs):
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                results[i] = output_path

    print("\nAll narration files have been generated.")
    return results


def split_script_into_phrases(
    script_text: str,
    *,
    max_chars: int = 400,
    keep_paragraphs: bool = True,
) -> List[str]:
    """Split a long script into manageable phrases for TTS.

    - Splits by paragraphs first (double newlines) if keep_paragraphs is True.
    - Within a block, splits by sentence boundaries and re-packs up to max_chars.
    """
    text = script_text.strip()
    if not text:
        return []

    blocks = re.split(r"\n\s*\n+", text) if keep_paragraphs else [text]

    sentence_re = re.compile(r"(?<=[.!?])\s+(?=[\"'“”‘’A-Z0-9])")
    phrases: list[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        sentences = re.split(sentence_re, block)

        current: list[str] = []
        current_len = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if current_len + (len(s) + (1 if current else 0)) <= max_chars:
                current.append(s)
                current_len += len(s) + (1 if current_len > 0 else 0)
            else:
                if current:
                    phrases.append(" ".join(current))
                # If single sentence is longer than max_chars, hard-split it
                if len(s) > max_chars:
                    for i in range(0, len(s), max_chars):
                        chunk = s[i : i + max_chars]
                        phrases.append(chunk)
                    current = []
                    current_len = 0
                else:
                    current = [s]
                    current_len = len(s)
        if current:
            phrases.append(" ".join(current))

    return phrases


def narrate_script(
    script_text: str,
    output_dir: str,
    *,
    settings: Optional[NarrationSettings] = None,
    max_workers: int = 3,
    force_regeneration: bool = False,
    max_chars_per_phrase: int = 400,
    combine_output_path: Optional[str] = None,
    filename_prefix: str = "narration_",
    start_index: int = 1,
    **settings_overrides: Any,
) -> dict[str, Any]:
    """Narrate a full script.

    - Splits the script into phrases (<= max_chars_per_phrase) respecting sentences/paragraphs.
    - If combine_output_path is provided, generates a single combined MP3 as well.

    Returns a dict with keys:
      - "phrases": list of phrases used
      - "segment_paths": list of per-segment paths (may be empty if skipped)
      - "combined_path": path to the combined MP3 if generated, else None
    """
    final_settings = _with_overrides(settings, settings_overrides)
    phrases = split_script_into_phrases(script_text, max_chars=max_chars_per_phrase)

    segment_paths = narrate_phrases(
        phrases,
        output_dir,
        settings=final_settings,
        max_workers=max_workers,
        force_regeneration=force_regeneration,
        filename_prefix=filename_prefix,
        start_index=start_index,
        **settings_overrides,
    )

    combined_path: Optional[str] = None
    if combine_output_path:
        # Fast path: concatenate already generated segment files instead of re-synthesizing
        print("Generating combined narration track (concatenating segments)...")
        try:
            try:
                import torchaudio  # type: ignore
            except Exception as _exc:
                torchaudio = None  # type: ignore

            if torchaudio is None:
                raise RuntimeError("torchaudio not available for combining MP3s")

            loaded_segments: list[torch.Tensor] = []
            sr: Optional[int] = None
            for p in [p for p in segment_paths if p]:
                wav, this_sr = torchaudio.load(p)  # (channels, frames)
                if sr is None:
                    sr = this_sr
                elif this_sr != sr:
                    raise RuntimeError("Sample rate mismatch during combination")
                loaded_segments.append(_append_silence(wav, this_sr, final_settings.end_silence_ms))

            if loaded_segments and sr is not None:
                combined_wav = torch.cat(loaded_segments, dim=1)
                os.makedirs(os.path.dirname(combine_output_path), exist_ok=True)
                save_mp3(combine_output_path, combined_wav, sr)
                combined_path = combine_output_path
        except Exception as exc:
            print(f"[Narration] Fast combine failed ({exc}); falling back to re-synthesis…")
            combined_wav: Optional[torch.Tensor] = None
            sr: Optional[int] = None
            for phrase in phrases:
                wav, this_sr = _synthesize_waveform(phrase, final_settings)
                wav = _append_silence(wav, this_sr, final_settings.end_silence_ms)
                if combined_wav is None:
                    combined_wav = wav if wav.ndim == 2 else wav.unsqueeze(0)
                    sr = this_sr
                else:
                    if this_sr != sr:
                        raise RuntimeError("Sample rate mismatch during combination")
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    combined_wav = torch.cat([combined_wav, wav], dim=1)
            if combined_wav is not None and sr is not None:
                os.makedirs(os.path.dirname(combine_output_path), exist_ok=True)
                save_mp3(combine_output_path, combined_wav, sr)
                combined_path = combine_output_path

    return {
        "phrases": phrases,
        "segment_paths": [p for p in segment_paths if p],
        "combined_path": combined_path,
    }


__all__ = [
    "NarrationSettings",
    "narrate_sentence",
    "narrate_phrases",
    "split_script_into_phrases",
    "narrate_script",
]


