import pyaudio
import numpy as np

# Microphone audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # Sampling rate suitable for most ASR models

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, 
    channels=CHANNELS, 
    rate=RATE, 
    input=True, 
    input_device_index=0,
    frames_per_buffer=CHUNK)

print("Microphone initialized. Listening...")
print("Press Ctrl+C to stop.")