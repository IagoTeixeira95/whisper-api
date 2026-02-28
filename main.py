from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import shutil

app = FastAPI()
model = WhisperModel("base", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open("audio.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, _ = model.transcribe("audio.wav", language="pt")
    text = "".join([segment.text for segment in segments])

    return {"text": text}