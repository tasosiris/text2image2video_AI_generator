import argparse
from tools.narration import NarrationSettings, narrate_sentence

def main():
    parser = argparse.ArgumentParser(description="Narrate text using a cloned voice.")
    parser.add_argument("--text", type=str, required=True, help="The text to narrate.")
    parser.add_argument("--voice_sample", type=str, default="templates/female_voice_new.mp3", help="Path to the voice sample for cloning.")
    parser.add_argument("--output_path", type=str, default="outputs/narration/product_promo.mp3", help="Path to save the output MP3 file.")
    
    # Add arguments for narration settings
    parser.add_argument("--exaggeration", type=float, default=0.6, help="Exaggeration factor for the voice.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--cfg_weight", type=float, default=1.0, help="Classifier-Free Guidance weight.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling probability.")
    parser.add_argument("--min_p", type=float, default=0.05, help="Minimum p for sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.5, help="Repetition penalty.")
    parser.add_argument("--end_silence_ms", type=int, default=0, help="Silence at the end of the narration in ms.")
    
    args = parser.parse_args()

    settings = NarrationSettings(
        voice_sample_path=args.voice_sample,
        device="auto",
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        cfg_weight=args.cfg_weight,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        end_silence_ms=args.end_silence_ms
    )

    print(f"Narrating text: '{args.text}'")
    print(f"Using voice sample: '{args.voice_sample}'")
    print(f"Saving to: '{args.output_path}'")

    narrate_sentence(args.text, args.output_path, settings)

    print("Narration complete.")

if __name__ == "__main__":
    main()
