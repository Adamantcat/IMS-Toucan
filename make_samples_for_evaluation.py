import os
import torch
from tqdm import tqdm
from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from run_utterance_cloner import UtteranceCloner

def wav2transcript():
    root = "/mount/arbeitsdaten/textklang/synthesis/Interspeech_2022/Evaluation"
    wav2text = dict()
    for dir in os.listdir(os.path.join(root, 'txt')):
        filename = dir.split('.')[0]
        with open(os.path.join(root, 'txt', dir), 'r') as text:
            transcript = text.read().replace('\n', ' ~ ')
        audio = os.path.join(root, 'wavs', filename + '.wav')
        wav2text[audio] = transcript
    return wav2text

if __name__ == '__main__':
    # exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    exec_device = 'cpu'
    out_dir = "audios/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    
    wav2text = wav2transcript()
    
    tts_prosa = InferenceFastSpeech2(device=exec_device, model_name='German', noise_reduce=True)
    tts_prosa.set_language('de')
    tts_prosa.set_utterance_embedding("/mount/arbeitsdaten/textklang/synthesis/Interspeech_2022/test/Heidelberg_Strophen/segment_6.wav")

    tts_poetry = InferenceFastSpeech2(device=exec_device, model_name='Zischler', noise_reduce=True)
    tts_poetry.set_language('de')

    uc_prosa = UtteranceCloner(model_id="German", device=exec_device)
    uc_poetry = UtteranceCloner(model_id="Zischler", device=exec_device)

    for audio, transcript in tqdm(wav2text.items()):
        audio_name = audio.split('/')[-1]
        audio_name = audio_name.split('.')[0]
        print(audio_name)
        tts_prosa.read_to_file([transcript], file_location=os.path.join(out_dir, 'prosa_unconditional', f"{audio_name}_prosa_uncond.wav"))
        tts_poetry.read_to_file([transcript], file_location=os.path.join(out_dir, 'poetry_unconditional', f"{audio_name}_poetic_uncond.wav"))

        uc_prosa.clone_utterance(path_to_reference_audio=audio,
                       reference_transcription=transcript,
                       filename_of_result=os.path.join(out_dir, 'prosa_cloned', f"{audio_name}_prosa_cloned.wav"),
                       clone_speaker_identity=True,
                       lang="de")
        
        uc_poetry.clone_utterance(path_to_reference_audio=audio,
                       reference_transcription=transcript,
                       filename_of_result=os.path.join(out_dir, 'poetry_cloned', f"{audio_name}_poetic_cloned.wav"),
                       clone_speaker_identity=False,
                       lang="de")
        





