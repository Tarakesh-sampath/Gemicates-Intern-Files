import whisper

model = whisper.load_model("base")

transcription = model.transcribe("Recording.mp3")

print(transcription)