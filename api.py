from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torchaudio
from transformers import pipeline

app = FastAPI()

# Initialize models
asr_pipeline = pipeline("automatic-speech-recognition")
translator_pipeline = pipeline("translation_en_to_fr")  # Example: English to French

class TranslationRequest(BaseModel):
    audio_format: str
    target_language: str

@app.post("/translate")
async def translate_audio(file: UploadFile, request: TranslationRequest):
    # Load audio file
    audio_data = await file.read()
    waveform, sample_rate = torchaudio.load(audio_data)

    # Perform speech recognition
    transcription = asr_pipeline(waveform, sampling_rate=sample_rate)
    text = transcription['text']

    # Perform translation
    translation = translator_pipeline(text, target_lang=request.target_language)
    translated_text = translation[0]['translation_text']

    return {"translated_text": translated_text}
