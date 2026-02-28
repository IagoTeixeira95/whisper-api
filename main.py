from fastapi import FastAPI, UploadFile, File
import whisper
import shutil

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open("audio.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.transcribe("audio.wav", language="pt")
    return {"text": result["text"]}