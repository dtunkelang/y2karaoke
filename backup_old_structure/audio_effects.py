"""Audio effects: key shifting and tempo changes."""

import numpy as np
import librosa
import soundfile as sf


def shift_key(audio_path: str, output_path: str, semitones: int) -> str:
    """
    Shift the pitch of an audio file by the specified number of semitones.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save the processed audio
        semitones: Number of semitones to shift (-12 to +12)

    Returns:
        Path to the output file
    """
    if semitones == 0:
        # No change needed, just copy
        import shutil
        shutil.copy(audio_path, output_path)
        return output_path

    print(f"Shifting key by {semitones:+d} semitones...")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Pitch shift
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

    # Save
    sf.write(output_path, y_shifted, sr)

    return output_path


def change_tempo(audio_path: str, output_path: str, tempo_multiplier: float) -> str:
    """
    Change the tempo of an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save the processed audio
        tempo_multiplier: Tempo change factor (1.0 = original, 2.0 = double speed)

    Returns:
        Path to the output file
    """
    if tempo_multiplier == 1.0:
        # No change needed, just copy
        import shutil
        shutil.copy(audio_path, output_path)
        return output_path

    print(f"Changing tempo to {tempo_multiplier:.2f}x...")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Time stretch (rate > 1 = faster, so we use the multiplier directly)
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_multiplier)

    # Save
    sf.write(output_path, y_stretched, sr)

    return output_path


def process_audio(
    audio_path: str,
    output_path: str,
    semitones: int = 0,
    tempo_multiplier: float = 1.0,
) -> str:
    """
    Apply key shift and/or tempo change to an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save the processed audio
        semitones: Number of semitones to shift (-12 to +12)
        tempo_multiplier: Tempo change factor (1.0 = original)

    Returns:
        Path to the output file
    """
    if semitones == 0 and tempo_multiplier == 1.0:
        # No processing needed
        import shutil
        shutil.copy(audio_path, output_path)
        return output_path

    print(f"Processing audio (key: {semitones:+d}, tempo: {tempo_multiplier:.2f}x)...")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Apply pitch shift if needed
    if semitones != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

    # Apply tempo change if needed
    if tempo_multiplier != 1.0:
        y = librosa.effects.time_stretch(y, rate=tempo_multiplier)

    # Save
    sf.write(output_path, y, sr)

    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python audio_effects.py <input> <output> <semitones> [tempo_multiplier]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    semitones = int(sys.argv[3])
    tempo = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    process_audio(input_file, output_file, semitones, tempo)
    print(f"Saved to {output_file}")
