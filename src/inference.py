import io
import torch
import torchaudio
from .model import model
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_audio_chunked(audio_data: bytes, chunk_duration: int = 1800, sample_rate: int = 16000) -> str: #8313MiB 
#def process_audio_chunked(audio_data: bytes, chunk_duration: int = 2400, sample_rate: int = 16000) -> str: #10245MiB
#def process_audio_chunked(audio_data: bytes, chunk_duration: int = 240, sample_rate: int = 16000) -> str: #10245MiB
    """Process audio in chunks efficiently from memory."""
    results = []
    
    # Load audio data from bytes
    audio_io = io.BytesIO(audio_data)
    waveform, loaded_sample_rate = torchaudio.load(audio_io)
    logger.info(f"Loaded waveform shape: {waveform.shape}, Sample rate: {loaded_sample_rate}")
    
    # Convert to mono if stereo
#    if waveform.shape[0] > 1:
#        waveform = torch.mean(waveform, dim=0, keepdim=True)
#        logger.info(f"Converted to mono. New shape: {waveform.shape}")
#    
#    # Resample if necessary
#    if loaded_sample_rate != sample_rate:
#        waveform = torchaudio.functional.resample(waveform, loaded_sample_rate, sample_rate)
#        logger.info(f"Resampled. New shape: {waveform.shape}")
#    
#    # Ensure waveform is 2D: (batch_size, time)
#    if waveform.dim() == 1:
#        waveform = waveform.unsqueeze(0)
#    elif waveform.dim() == 3:
#        waveform = waveform.squeeze(0)
    
    logger.info(f"Final waveform shape before chunking: {waveform.shape}")
    
    chunk_size = chunk_duration * sample_rate
#    num_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size != 0 else 0)
    for i in range(0, waveform.shape[1], chunk_size):
        chunk = waveform[:, i:i+chunk_size]
#    for i in range(num_chunks):
#        start = i * chunk_size
#        end = min((i + 1) * chunk_size, len(audio_data))
#        chunk = audio_data[start:end]
#        logger.info(f"Chunk shape: {chunk.shape}")
#        print(f"Chunk shape: {chunk.shape}")
        
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
