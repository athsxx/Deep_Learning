from flask import Flask, request, jsonify
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, MarianMTModel, MarianTokenizer
import numpy as np
import sounddevice as sd
import io, torchaudio
import noisereduce as nr


app = Flask(__name__)

# Load models and processors
asr_model_name = "facebook/wav2vec2-large-960h"  # Example model, replace with your ASR model
translation_model_name = "Helsinki-NLP/opus-mt-en-fr"  # Example translation model, replace with your model

processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)

Tmodel = MarianMTModel.from_pretrained(translation_model_name)
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

sampling_rate = 16000  # Set your sampling rate
duration = 5  # seconds

# Function to normalize the audio
def normalize(audio):
    audio = audio / torch.max(torch.abs(audio))
    return audio

# Function to preprocess the audio
def preprocess_audio(audio, sampling_rate):
    # Resample if necessary
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio = resampler(audio)
        sampling_rate = 16000
    
    # Convert to numpy array for noise reduction
    audio_np = audio.numpy()

    # Apply noise reduction
    reduced_noise_array = nr.reduce_noise(y=audio_np, sr=sampling_rate)
    
    # Convert back to tensor
    audio = torch.tensor(reduced_noise_array)
    
    # Normalize the audio
    audio = normalize(audio)
    
    return audio, sampling_rate


@app.route('/translate', methods=['POST'])
def translate_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_data = file.read()
    
    # Load audio data
    audio_array = np.frombuffer(audio_data, dtype=np.float32)
    
    # Convert audio data to tensor
    audio_tensor = torch.tensor(audio_array).unsqueeze(0)
    
    # Preprocess the audio
    audio_tensor, sr = preprocess_audio(audio_tensor, sampling_rate)
    
    # ASR
    input_values = processor(audio_tensor.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    # Translation
    src_text = [transcription]
    translated = Tmodel.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    
    return jsonify({'translation': translated_text})

if __name__ == '__main__':
    app.run(debug=True)


#curl -X POST http://127.0.0.1:5000/translate -F "file=@test_audio.wav"
