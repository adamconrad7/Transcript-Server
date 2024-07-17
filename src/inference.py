import io
import torch
import torchaudio
from typing import List
import logging
from .model import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_audio_chunked(audio_data: bytes, chunk_duration: int = 1800, sample_rate: int = 16000) -> str: #8313MiB 
    model = get_model()
#def process_audio_chunked(audio_data: bytes, chunk_duration: int = 2400, sample_rate: int = 16000) -> str: #10245MiB
#def process_audio_chunked(audio_data: bytes, chunk_duration: int = 240, sample_rate: int = 16000) -> str: #10245MiB
    """Process audio in chunks efficiently from memory."""
    results = []
    
    # Load audio data from bytes
    audio_io = io.BytesIO(audio_data)
    waveform, loaded_sample_rate = torchaudio.load(audio_io)
    logger.info(f"Loaded waveform shape: {waveform.shape}, Sample rate: {loaded_sample_rate}")
   
    logger.info(f"Final waveform shape before chunking: {waveform.shape}")
    
    chunk_size = chunk_duration * sample_rate
    for i in range(0, waveform.shape[1], chunk_size):
        chunk = waveform[:, i:i+chunk_size]
       
        with torch.no_grad():
            chunk_result = model.transcribe([chunk.numpy().squeeze()], batch_size=1) #8313MiB 74.61s 0.00491
            #chunk_result = model.transcribe([chunk.numpy().squeeze()], batch_size=4)
            #chunk_result = model.transcribe([chunk.numpy().squeeze()], batch_size=512, num_workers=2)
        results.append(chunk_result[0])
        
        torch.cuda.empty_cache()  # Clear GPU cache after each chunk
    
    return " ".join(results)

def transcribe_audio(audio_data: bytes) -> str:
    try:
        transcription = process_audio_chunked(audio_data)
        return transcription
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        return f"Transcription failed: {str(e)}"
