from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, torch, io
from melo.api import TTS
from typing import Optional
import tempfile
import uuid

app = FastAPI(title="MeloTTS API")

device = 'auto'
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}

default_text_dict = {
    'EN': 'The field of text-to-speech has seen rapid development recently.',
    'ES': 'El campo de la conversión de texto a voz ha experimentado un rápido desarrollo recientemente.',
    'FR': 'Le domaine de la synthèse vocale a connu un développement rapide récemment',
    'ZH': 'text-to-speech 领域近年来发展迅速',
    'JP': 'テキスト読み上げの分野は最近急速な発展を遂げています',
    'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',    
}

class TTSRequest(BaseModel):
    speaker: str
    text: str
    speed: float = 1.0
    language: str

class SpeakersRequest(BaseModel):
    language: str
    text: str

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    try:
        if request.language not in models:
            raise HTTPException(status_code=400, detail="Invalid language")
        
        # UUIDを生成してファイル名に使用
        unique_filename = f"speech_{str(uuid.uuid4())}.wav"
        
        bio = io.BytesIO()
        models[request.language].tts_to_file(
            request.text, 
            models[request.language].hps.data.spk2id[request.speaker], 
            bio, 
            speed=request.speed,
            format='wav'
        )
        bio.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(bio.getvalue())
            temp_file_path = temp_file.name

        async def cleanup_file():
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up file: {e}")
        
        return FileResponse(
            temp_file_path,
            media_type="audio/wav",
            filename=unique_filename,  # UUIDベースのファイル名を使用
            background=cleanup_file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_speakers")
async def load_speakers(request: SpeakersRequest):
    try:
        if request.language not in models:
            raise HTTPException(status_code=400, detail="Invalid language")
        
        available_speakers = list(models[request.language].hps.data.spk2id.keys())
        new_text = default_text_dict[request.language] if request.text in default_text_dict.values() else request.text
        
        return {
            "speaker": available_speakers[0],
            "available_speakers": available_speakers,
            "text": new_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
