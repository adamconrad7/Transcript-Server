from fastapi import FastAPI, File, UploadFile, HTTPException
from src.inference import transcribe_audio
from src.config import config
from src.preprocess import convert
import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        logger.info(f"Received audio file: {file.filename}, Size: {len(audio_data)} bytes")
        
        converted_audio_data, duration = await asyncio.to_thread(convert, audio_data)
        logger.info("Audio conversion completed")
        
        start = time.time()
        transcription = await asyncio.to_thread(transcribe_audio, converted_audio_data)
        end = time.time()
        elapsed = end - start
        
        print(f"Duration: {duration }s")
        print(f'Elapsed: {elapsed}s')
        print(f'RTF: {elapsed/duration}')

        return {"transcription": transcription}
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"], reload=True)
