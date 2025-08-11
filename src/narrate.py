"""
Chatterbox-based narration interface matching the old Deepgram behavior.
This preserves the same outputs in `narration/narration_###.mp3`.
"""

import os
import concurrent.futures
from typing import Optional

import torch

from tts_chatterbox import synthesize_with_chatterbox, save_mp3


def narrate_sentence(sentence: str, output_path: str,
                     device: Optional[str] = None,
                     audio_prompt_path: Optional[str] = None,
                     exaggeration: float = 0.4,
                     temperature: float = 0.5,
                     cfg_weight: float = 0.5,
                     top_p: float = 1.0,
                     min_p: float = 0.05,
                     repetition_penalty: float = 1.2,
                     end_silence_ms: int = 300):
    try:
        wav, sr = synthesize_with_chatterbox(
            sentence,
            device=device,
            voice=None,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if end_silence_ms and end_silence_ms > 0:
            extra = int(sr * (end_silence_ms / 1000.0))
            if extra > 0:
                silence = torch.zeros((wav.shape[0], extra), dtype=wav.dtype)
                wav = torch.cat([wav, silence], dim=1)
        save_mp3(output_path, wav, sr)
        print(f"Successfully saved narration to '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error while processing sentence: {e}")
        return None


def narrate_phrases(phrases, output_dir,
                    force_regeneration: bool = False,
                    device: Optional[str] = None,
                    audio_prompt_path: Optional[str] = None,
                    exaggeration: float = 0.4,
                    temperature: float = 0.4,
                    cfg_weight: float = 0.8,
                    top_p: float = 1.0,
                    min_p: float = 0.05,
                    repetition_penalty: float = 1.2,
                    end_silence_ms: int = 300):
    narration_output_dir = os.path.join(output_dir, "narration")
    os.makedirs(narration_output_dir, exist_ok=True)

    if not phrases:
        print("No phrases provided to narrate.")
        return

    print(f"Narrating {len(phrases)} phrases.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, phrase in enumerate(phrases):
            filename = f"narration_{i+1:03d}.mp3"
            output_path = os.path.join(narration_output_dir, filename)

            if not force_regeneration and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"  -> Narration for phrase {i+1} already exists. Skipping.")
                continue

            futures.append(
                executor.submit(
                    narrate_sentence,
                    phrase,
                    output_path,
                    device,
                    audio_prompt_path,
                    exaggeration,
                    temperature,
                    cfg_weight,
                    top_p,
                    min_p,
                    repetition_penalty,
                    end_silence_ms,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")

    print("\nAll narration files have been generated.")
