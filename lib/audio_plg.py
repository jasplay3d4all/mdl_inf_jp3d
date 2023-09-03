import torchaudio
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

class audio_gen:
    def __init__(self, musicgen_mdl='small'):
        # 'small', - 
        # 'melody' - text to music and text+melody to music, 'medium', 
        # 'large' - text to music only
        # 
        self.music_gen = MusicGen.get_pretrained(musicgen_mdl)
        preload_models() # Bark Model for text 2 speech
        return
    
    def gen_music(self, bg_music_prompt, num_sec, music_path):
        melody_path=None; num_sam=1;
        self.music_gen.set_generation_params(duration=num_sec)  # generate 8 seconds.
        if(bg_music_prompt == None):
            wav = self.music_gen.generate_unconditional(num_sam)    # generates 4 unconditional audio samples
        if(bg_music_prompt):
            wav = self.music_gen.generate(bg_music_prompt)  # generates 3 samples.
        if(bg_music_prompt and melody_path):
            melody, sr = torchaudio.load(melody_path)
            # generates using the melody from the given audio and the provided descriptions.
            wav = self.music_gen.generate_with_chroma(bg_music_prompt, melody[None].expand(num_sam, -1, -1), sr)

        os.makedirs(music_path, exist_ok=True)
        # vdo_filepath = os.path.join(vdo_path, "generated.mp4")
        format = "wav" #".wav"
        ado_list = []
        for idx, one_wav in enumerate(wav):
            music_file_apth = os.path.join(music_path, "output"+str(idx).zfill(5))
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(music_file_apth, one_wav.cpu(), self.music_gen.sample_rate, 
                strategy="loudness", format=format)
            # write_wav(music_file_apth, self.music_gen.sample_rate, one_wav.cpu())
            # write_wav(music_file_apth, rate=self.music_gen.sample_rate, data=one_wav.cpu().numpy())
            ado_list.append({'path':music_file_apth+"."+format})

        return ado_list
    
    def gen_speech(self, text, music_path, speaker=None, language=None):
        # ~/.cache/suno/bark_v0/text_2.pt
        # generate audio from text
        # text_prompt = """
        #     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
        #     But I also have other interests such as playing tic tac toe.
        # """
        
        # https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
        # supported speaker voice
        # https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb
        # long form audio
        # https://discord.com/invite/J2B2vsjKuE - discord community
        audio_array = generate_audio(text, history_prompt="v2/en_speaker_9")

        # save audio to disk
        os.makedirs(music_path, exist_ok=True)
        music_file_apth = os.path.join(music_path, "generated.wav")
        write_wav(music_file_apth, SAMPLE_RATE, audio_array)
        return {'path':music_file_apth}


def gen_music(bg_music_prompt, num_sec, music_path, melody_path=None, musicgen_mdl='melody'):
    # 'small', - 
    # 'melody' - text to music and text+melody to music, 'medium', 
    # 'large' - text to music only
    # 

    bg_music_prompt = [bg_music_prompt]
    music_gen = MusicGen.get_pretrained(musicgen_mdl)
    music_gen.set_generation_params(duration=num_sec)
    # if(bg_music_prompt == None):
    #     wav = music_gen.generate_unconditional(num_sam)    # generates 4 unconditional audio samples
    if(bg_music_prompt):
        wav = music_gen.generate(bg_music_prompt)  # generates 3 samples.
    if(bg_music_prompt and melody_path):
        melody, sr = torchaudio.load(melody_path)
        # generates using the melody from the given audio and the provided descriptions.
        wav = music_gen.generate_with_chroma(bg_music_prompt, melody[None].expand(num_sam, -1, -1), sr)

    os.makedirs(music_path, exist_ok=True)
    # vdo_filepath = os.path.join(vdo_path, "generated.mp4")
    format = "wav" #".wav"
    ado_list = []
    for idx, one_wav in enumerate(wav):
        music_file_apth = os.path.join(music_path, "output"+str(idx).zfill(5))
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(music_file_apth, one_wav.cpu(), music_gen.sample_rate, 
            strategy="loudness", format=format)
        # write_wav(music_file_apth, self.music_gen.sample_rate, one_wav.cpu())
        # write_wav(music_file_apth, rate=self.music_gen.sample_rate, data=one_wav.cpu().numpy())
        ado_list.append({'path':music_file_apth+"."+format})

    return ado_list

def gen_speech(text, music_path, history_prompt="v2/en_speaker_9"):
    # ~/.cache/suno/bark_v0/text_2.pt
    # generate audio from text
    # text_prompt = """
    #     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
    #     But I also have other interests such as playing tic tac toe.
    # """
    
    # https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
    # supported speaker voice
    # https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb
    # long form audio
    # https://discord.com/invite/J2B2vsjKuE - discord community
    preload_models() # Bark Model for text 2 speech
    audio_array = generate_audio(text, history_prompt=history_prompt)

    # save audio to disk
    os.makedirs(music_path, exist_ok=True)
    music_file_apth = os.path.join(music_path, "generated.wav")
    write_wav(music_file_apth, SAMPLE_RATE, audio_array)
    return [{'path':music_file_apth}]


if __name__ == "__main__":

    audio_gen_mdl = audio_gen("melody")
    bg_music_prompt = ["Drum beats"]
    # audio_gen_mdl.gen_music(bg_music_prompt, 2, "./output")
    audio_gen_mdl.gen_speech("Mighty jungle in mighty jungle ", "./output")
