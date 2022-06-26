import os
import torch
from tqdm import tqdm
from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

if __name__ == '__main__':
    # exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    exec_device = 'cpu'

    tts_prosa = InferenceFastSpeech2(device=exec_device, model_name='German', noise_reduce=True)
    tts_prosa.set_language('de')
    tts_prosa.set_utterance_embedding("audios/references/karlsson.wav")

    tts_poetry = InferenceFastSpeech2(device=exec_device, model_name='Wunderhorn', noise_reduce=True)
    tts_poetry.set_language('de')
    tts_prosa.set_utterance_embedding("audios/references/karlsson.wav")

    for poem in os.listdir("/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data_2/Test"):
        print(poem)
        out_dir_prose = f"audios/test/Karlsson/prose/{poem}"
        out_dir_poetry = f"audios/test/Karlsson/poetry/{poem}"
        os.makedirs(out_dir_prose, exist_ok=True)
        os.makedirs(out_dir_poetry, exist_ok=True)

        with open(os.path.join("/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data_2/Test", poem, "transcript.txt"), "r") as transcript:
            for i, line in enumerate(transcript):
                text = line.split('\t')[1].rstrip().lstrip()
                tts_prosa.read_to_file([text], file_location=f"{out_dir_prose}/{i}.wav")
                tts_poetry.read_to_file([text], file_location=f"{out_dir_poetry}/{i}.wav")





