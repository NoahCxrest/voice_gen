import sys
import os
import requests
import io
import wave
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from piper.voice import PiperVoice

VOICE_NAME = "en_US-ryan-medium"
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/"
MODEL_PATH = f"{VOICE_NAME}.onnx"
CONFIG_PATH = f"{VOICE_NAME}.onnx.json"
MODEL_URL = BASE_URL + MODEL_PATH
CONFIG_URL = BASE_URL + CONFIG_PATH
SAMPLE_RATE = 22050 

def download_file(url, path):
    """Download a file if it doesn't already exist."""
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        resp = requests.get(url)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"Saved {path}")
    else:
        print(f"{path} already exists, skipping download.")

def synthesize_tts(text: str, voice) -> bytes:
    """Generate audio bytes from text using Piper."""
    print("Synthesizing speech...")
    chunks = list(voice.synthesize(text))
    print("Synthesis complete.")
    
    silence_duration = 0.8
    silence_samples = int(voice.config.sample_rate * silence_duration)
    silence_bytes = b'\x00\x00' * silence_samples
    
    audio_bytes = b""
    for i, chunk in enumerate(chunks):
        if i > 0:
            audio_bytes += silence_bytes
        audio_bytes += chunk.audio_int16_bytes
    
    return audio_bytes

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert PCM bytes to WAV bytes."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()

app = FastAPI()

class SynthesizeRequest(BaseModel):
    text: str

print("Loading voice model...")
download_file(MODEL_URL, MODEL_PATH)
download_file(CONFIG_URL, CONFIG_PATH)
voice = PiperVoice.load(MODEL_PATH)
print("Model loaded.")

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        pcm_bytes = synthesize_tts(text, voice)
        
        wav_bytes = pcm_to_wav(pcm_bytes)
        
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
