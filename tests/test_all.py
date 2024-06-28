import pytest
from unittest.mock import patch
import io
import torch
from src.inference import transcribe_audio
from src.config import load_config
from src.model import model
from src.main import app
from fastapi.testclient import TestClient

def test_transcribe_audio():
    with open("tests/test_audio.wav", "rb") as f:
        audio_data = f.read()
    result = transcribe_audio(audio_data)
    assert isinstance(result, str)
    assert len(result) > 0


def test_load_config():
    config = load_config()
    assert "model" in config
    assert "server" in config

def test_model_init():
    assert model is not None


@patch('src.inference.transcribe_audio')
def test_transcribe_endpoint(mock_transcribe):
    mock_transcribe.return_value = "Test transcription"
    client = TestClient(app)
    
    with open("tests/test_audio.wav", "rb") as f:
        response = client.post("/transcribe/", files={"file": ("test.wav", f, "audio/wav")})
    
    print(f"Mock called: {mock_transcribe.called}")
    print(f"Mock call count: {mock_transcribe.call_count}")
    print(f"Mock call args: {mock_transcribe.call_args}")
    print(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == {"transcription": "Test transcription"}
##def test_transcribe_endpoint():
##    client = TestClient(app)
##    with patch('src.inference.transcribe_audio') as mock_transcribe:
##        mock_transcribe.return_value = "Test transcription"
##        with open("tests/test_audio.wav", "rb") as f:
##            response = client.post("/transcribe/", files={"file": ("test.wav", f, "audio/wav")})
##    assert response.status_code == 200
##    assert response.json() == {"transcription": "Test transcription"}
##
# Add more tests as needed
