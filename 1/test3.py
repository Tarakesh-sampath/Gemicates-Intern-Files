from faster_whisper import WhisperModel

model_size = "medium.en"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
#model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("harvard.wav")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
segments = list(segments)  
print(segments)
transcribe = ""
for segment in segments:
    transcribe+=str("[%.2fs -> %.2fs] %s \n" % (segment.start, segment.end, segment.text))