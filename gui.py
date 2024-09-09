import sys
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import wavio
import requests

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        # Create UI elements
        self.start_button = tk.Button(root, text="Start", command=self.start_recording)
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.translate_button = tk.Button(root, text="Translate", command=self.translate_audio, state=tk.DISABLED)

        # Layout
        self.start_button.pack(pady=10)
        self.stop_button.pack(pady=10)
        self.translate_button.pack(pady=10)

        # Variables
        self.recording = False
        self.audio_data = None
        self.sample_rate = 44100
        self.filename = "test.wav"

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.translate_button.config(state=tk.DISABLED)
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.sample_rate)
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.save_audio()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.translate_button.config(state=tk.NORMAL)

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.recording:
            self.audio_data.append(indata.copy())

    def save_audio(self):
        audio_np = np.concatenate(self.audio_data, axis=0)
        wavio.write(self.filename, audio_np, self.sample_rate, sampwidth=2)
        print(f"Audio saved as {self.filename}")

    def translate_audio(self):
        with open(self.filename, 'rb') as f:
            response = requests.post('http://127.0.0.1:5000/translate', files={'file': f})

        if response.ok:
            response_json = response.json()
            translation = response_json.get('translation', 'No translation found')
            messagebox.showinfo("Translation", translation)
        else:
            messagebox.showerror("Error", f"Failed to translate: {response.status_code}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
