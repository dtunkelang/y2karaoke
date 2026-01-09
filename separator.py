"""Vocal separation using audio-separator (demucs)."""

import os
from audio_separator.separator import Separator


def mix_stems(stem_files: list[str], output_path: str) -> str:
    """Mix multiple audio stems into a single file using pydub."""
    from pydub import AudioSegment

    if not stem_files:
        raise RuntimeError("No stem files to mix")

    # Load and mix all stems
    mixed = None
    for stem_file in stem_files:
        stem = AudioSegment.from_wav(stem_file)
        if mixed is None:
            mixed = stem
        else:
            mixed = mixed.overlay(stem)

    # Export the mixed audio
    mixed.export(output_path, format="wav")
    return output_path


def separate_vocals(audio_path: str, output_dir: str = ".") -> dict:
    """
    Separate vocals from instrumental track.

    Returns:
        dict with keys: vocals_path, instrumental_path
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Separating vocals from instrumental...")

    # Initialize separator with demucs model
    separator = Separator(
        output_dir=output_dir,
        output_format="wav",
    )

    # Load the demucs model (htdemucs is the default high-quality model)
    separator.load_model(model_filename="htdemucs_ft.yaml")

    # Separate the audio
    output_files = separator.separate(audio_path)

    # Convert to full paths
    output_files = [os.path.join(output_dir, f) if not os.path.isabs(f) else f for f in output_files]

    # Find vocals and instrumental in output
    vocals_path = None
    instrumental_path = None

    for f in output_files:
        if "vocals" in f.lower():
            vocals_path = f
        elif "instrumental" in f.lower() or "no_vocals" in f.lower():
            instrumental_path = f

    # If we didn't find an instrumental track, we need to create one
    # by combining non-vocal stems (bass + drums + other)
    if instrumental_path is None and vocals_path:
        non_vocal_files = [f for f in output_files if "vocals" not in f.lower()]
        if non_vocal_files:
            # Mix bass + drums + other into instrumental
            base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
            instrumental_path = os.path.join(output_dir, f"{base_name}_instrumental.wav")
            print(f"Mixing {len(non_vocal_files)} stems into instrumental track...")
            mix_stems(non_vocal_files, instrumental_path)

    if not vocals_path or not instrumental_path:
        raise RuntimeError(f"Failed to separate tracks. Output files: {output_files}")

    print(f"Vocals: {vocals_path}")
    print(f"Instrumental: {instrumental_path}")

    return {
        'vocals_path': vocals_path,
        'instrumental_path': instrumental_path,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python separator.py <audio_file>")
        sys.exit(1)

    result = separate_vocals(sys.argv[1], output_dir="./output")
    print(f"Vocals: {result['vocals_path']}")
    print(f"Instrumental: {result['instrumental_path']}")
