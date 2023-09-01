import pytest
from tts_api import generate_speech

def test_generate_speech_no_gpu():
    with pytest.raises(ValueError):
        generate_speech("Hello World", "output.wav", gpu=False)

def test_generate_speech_no_text():
    with pytest.raises(ValueError):
        generate_speech("", "output.wav")

def test_generate_speech_no_output_file():
    with pytest.raises(ValueError):
        generate_speech("Hello World", "")

def test_generate_speech_invalid_model():
    with pytest.raises(ValueError):
        generate_speech("Hello World", "output.wav", model_path="invalid/path")

def test_generate_speech_invalid_voice_dir():
    with pytest.raises(ValueError):
        generate_speech("Hello World", "output.wav", voice_dir="invalid/path")

def test_generate_speech_invalid_speaker():
    with pytest.raises(ValueError):
        generate_speech("Hello World", "output.wav", speaker="invalid_speaker")

def test_generate_speech():
    assert generate_speech("Hello World", "output.wav")
