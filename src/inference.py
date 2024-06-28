import io
import torch
import torchaudio
from .model import model
from .config import config
import tempfile
import os

def transcribe_audio(audio_data: bytes) -> str:
  # Create a temporary file to store the audio data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    try:
        # Transcribe the audio file by passing the file path

        with torch.no_grad():
            transcription = model.transcribe([temp_audio_path])[0]
            return transcription
    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)
