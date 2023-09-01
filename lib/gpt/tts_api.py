import os
from TTS.api import TTS

def generate_speech(text, output_file, model_path="tts_models/multilingual/multi-dataset/bark",
                    voice_dir="bark_voices/", speaker="ljspeech", gpu=True):
    
    if not gpu:
        raise ValueError("GPU is required for TTS operations.")

    if not text:
        raise ValueError("Text is required.")

    if not output_file:
        raise ValueError("Output file path is required.")

    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    if not os.path.exists(voice_dir):
        raise ValueError(f"Voice directory does not exist: {voice_dir}")

    tts = TTS(model_path, gpu=gpu)

    # If speaker is provided, it's assumed that we're cloning a new speaker.
    if speaker:
        if not os.path.exists(os.path.join(voice_dir, speaker)):
            raise ValueError(f"Speaker does not exist: {speaker}")
        
        tts.tts_to_file(text=text,
                        file_path=output_file,
                        voice_dir=voice_dir,
                        speaker=speaker)
    else:
        tts.tts_to_file(text, file_path=output_file)

    return True
