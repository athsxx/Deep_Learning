from flask import Flask, request, jsonify
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, MarianMTModel, MarianTokenizer
import numpy as np
import soundfile as sf
import io
import torchaudio
import noisereduce as nr

app = Flask(__name__)

# Load models and processors
asr_model_name = "facebook/wav2vec2-large-960h"
translation_model_name = "Helsinki-NLP/opus-mt-en-hi"

processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
Tmodel = MarianMTModel.from_pretrained(translation_model_name)
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

sampling_rate = 16000

# Function to normalize the audio
def normalize(audio):
    return audio / torch.max(torch.abs(audio))

# Function to preprocess the audio
def preprocess_audio(audio, sr):
    # Apply noise reduction
    audio_np = audio.numpy()
    reduced_noise_array = nr.reduce_noise(y=audio_np, sr=sr)
    
    # Convert back to tensor
    audio = torch.tensor(reduced_noise_array).float()
    
    # Normalize the audio
    audio = normalize(audio)
    
    return audio

@app.route('/translate', methods=['POST'])
def translate_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_data = file.read()
    audio_io = io.BytesIO(audio_data)
    
    try:
        # Load audio file
        audio, sr = sf.read(audio_io, dtype='float32')

        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            audio_tensor = torch.tensor(audio).float().unsqueeze(0)  # Add batch dimension
            audio = resampler(audio_tensor).squeeze().numpy()  # Remove batch dimension
            sr = sampling_rate

        # Convert to tensor
        audio_tensor = torch.tensor(audio).float()
        
        # Preprocess the audio
        audio_tensor = preprocess_audio(audio_tensor, sr)
        
        # ASR
        input_values = processor(audio_tensor.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        
        # Translation
        src_text = [transcription]
        translated = Tmodel.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
        translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
        print(translated_text)
        response = jsonify({'translation': translated_text})
        response.charset = 'utf-8'
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

#curl -X POST http://127.0.0.1:5000/translate -F "file=@test.wav" | jq 