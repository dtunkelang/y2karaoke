"""Audio effects processing with validation."""

from pathlib import Path

import librosa
import soundfile as sf

from ..config import AUDIO_SAMPLE_RATE
from ..exceptions import ValidationError
from ..utils.logging import get_logger
from ..utils.validation import validate_key_shift, validate_tempo

logger = get_logger(__name__)

class AudioProcessor:
    """Audio effects processor with validation and error handling."""
    
    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def process_audio(
        self,
        input_path: str,
        output_path: str,
        semitones: int = 0,
        tempo_multiplier: float = 1.0,
    ) -> str:
        """Apply key shift and/or tempo change to audio file."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise ValidationError(f"Input audio file not found: {input_path}")
        
        # Validate parameters
        semitones = validate_key_shift(semitones)
        tempo_multiplier = validate_tempo(tempo_multiplier)
        
        # No processing needed
        if semitones == 0 and tempo_multiplier == 1.0:
            logger.debug("No audio effects requested, copying file")
            import shutil
            shutil.copy(input_path, output_path)
            return str(output_path)
        
        logger.info(f"Processing audio: key={semitones:+d}, tempo={tempo_multiplier:.2f}x")
        
        try:
            # Load audio
            y, sr = librosa.load(str(input_path), sr=self.sample_rate)
            
            # Apply pitch shift
            if semitones != 0:
                logger.debug(f"Applying pitch shift: {semitones:+d} semitones")
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
            
            # Apply tempo change
            if tempo_multiplier != 1.0:
                logger.debug(f"Applying tempo change: {tempo_multiplier:.2f}x")
                y = librosa.effects.time_stretch(y, rate=tempo_multiplier)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed audio
            sf.write(str(output_path), y, sr)
            
            logger.debug(f"Audio processing completed: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            raise ValidationError(f"Audio processing failed: {e}")
    
    def shift_key(self, input_path: str, output_path: str, semitones: int) -> str:
        """Shift pitch by specified semitones."""
        return self.process_audio(input_path, output_path, semitones=semitones)
    
    def change_tempo(self, input_path: str, output_path: str, tempo_multiplier: float) -> str:
        """Change tempo by specified multiplier."""
        return self.process_audio(input_path, output_path, tempo_multiplier=tempo_multiplier)

# Convenience functions for backward compatibility
def process_audio(
    audio_path: str,
    output_path: str,
    semitones: int = 0,
    tempo_multiplier: float = 1.0,
) -> str:
    """Apply audio effects to file."""
    processor = AudioProcessor()
    return processor.process_audio(audio_path, output_path, semitones, tempo_multiplier)

def shift_key(audio_path: str, output_path: str, semitones: int) -> str:
    """Shift key by semitones."""
    processor = AudioProcessor()
    return processor.shift_key(audio_path, output_path, semitones)

def change_tempo(audio_path: str, output_path: str, tempo_multiplier: float) -> str:
    """Change tempo by multiplier."""
    processor = AudioProcessor()
    return processor.change_tempo(audio_path, output_path, tempo_multiplier)
