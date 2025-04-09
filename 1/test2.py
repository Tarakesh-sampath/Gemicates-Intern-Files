import wave
import pyaudio
import os
from colorama import Fore, Style
import faster_whisper
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")

NEON_GREEN = Fore.GREEN + Style.BRIGHT
RESET_COLOR = Style.RESET_ALL

chunk_length = 3
sample_rate = 16000
def transcribe_chunk(model, chunk_file):
    # Transcribe audio chunk and return text
    result = model.transcribe(chunk_file)
    transcription = result["text"]
    return transcription

def record_chunk(p, stream, file_path, chunk_length=1):
    frames = [] 
    for _ in range(0, int(sample_rate / 1024 * chunk_length)):
        try:
            data = stream.read(1024)
            frames.append(data)
        except Exception as e:
            print(f"Error reading audio stream: {e}")
            return
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    # Choose your model settings
    model_size = "tiny.en"
    model = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    accumulated_transcription = ""  # Initialize an empty string to accumulate transcriptions
    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file,chunk_length)
            if os.path.exists(chunk_file):
                transcription = transcribe_chunk(model, chunk_file)
                print(NEON_GREEN + transcription + RESET_COLOR)
                os.remove(chunk_file)
                # Append the new transcription to the accumulated transcription
                accumulated_transcription += transcription + " "
    except KeyboardInterrupt:
        print("Stopping...")
        # Write the accumulated transcription to the log file
        with open("log.txt", "a") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG: " + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
