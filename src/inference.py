import io
import torch
import torchaudio
from .model import model
from .config import config
import tempfile
import os

def transcribe_audio(audio_data: bytes) -> str:
#    audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_data))
#    target_sample_rate = config["model"]["sample_rate"]
#    if sample_rate != target_sample_rate:
#        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
#        audio_tensor = resampler(audio_tensor)
#    
#    # NeMo models expect audio in [B, T] format
#    audio_tensor = audio_tensor.squeeze().unsqueeze(0)
#    
#    with torch.no_grad():
#        transcription = model.transcribe(audio_tensor)[0]
#    
#    return transcription
   # Create a temporary file to store the audio data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    try:
        # Transcribe the audio file by passing the file path
        transcription = model.transcribe([temp_audio_path])[0]
        return transcription
    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)
