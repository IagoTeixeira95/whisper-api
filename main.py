from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import shutil
import subprocess
import uuid
import os

app = FastAPI()
model = WhisperModel("base", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open("audio.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, _ = model.transcribe("audio.wav", language="pt")
    text = "".join([segment.text for segment in segments])

    return {"text": text}



@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):

    input_path = f"/tmp/{uuid.uuid4()}.wav"
    output_path = f"/tmp/{uuid.uuid4()}.wav"

    # 1️⃣ Salvar áudio recebido
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2️⃣ Transcrição
    segments, _ = whisper.transcribe(input_path)
    texto = " ".join([seg.text for seg in segments])

    # 3️⃣ Resposta (LLM simples para economizar RAM)
    resposta = f"Você disse: {texto}"

    # 4️⃣ Gerar áudio com Piper
    subprocess.run(
        ["piper", "--model", "pt_BR-faber-medium.onnx", "--output_file", output_path],
        input=resposta.encode()
    )

    # 5️⃣ Retornar áudio
    return FileResponse(output_path, media_type="audio/wav")