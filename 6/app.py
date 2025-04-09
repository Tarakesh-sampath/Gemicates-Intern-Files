from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import io
from scipy.signal import resample
from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name="Tarakeshwaran/whisper-small-en" 

# Load Whisper transcription pipeline
transcription_pipeline = pipeline(
    task='automatic-speech-recognition',
    model=model_name,
    #device=0 if device == "cuda" else -1,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

app = FastAPI()

@app.post("/ASR-in/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Endpoint to upload audio as a NumPy array and save it as a WAV file.
    """
    try:
        # Read the file contents into a NumPy array
        file_contents = await file.read()
        audio_array = np.load(io.BytesIO(file_contents))  # Expecting a NumPy array file

        #transcribe
        transcribe = transcription_pipeline(audio_array)        
        return JSONResponse(
            content={"message": "File saved successfully", "text": transcribe},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"message": "Error processing file", "error": str(e)},
            status_code=500
        )
