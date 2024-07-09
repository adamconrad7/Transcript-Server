from fastapi import FastAPI, File, UploadFile
from src.inference import transcribe_audio
from src.config import config
from src.preprocess import convert

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_data = await file.read()
    # TODO: call pre-process
    audio_data = convert(audio_data)

    
    transcription = transcribe_audio(audio_data)
    return {"transcription": transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
