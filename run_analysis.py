import torch
import soundfile as sf
import os
import re

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Parselmouth
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend

import warnings
warnings.filterwarnings("ignore")


def get_avg_dur_phones(audio_path, transcript):
    phone_durations = list()
    pause_durations = list()
     # loading modules
    acoustic_model = Aligner()
    # acoustic_model.load_state_dict(torch.load("Models/Aligner/aligner.pt", map_location='cpu')["asr_model"])
    acoustic_model.load_state_dict(torch.load("Models/Aligner/aligner.pt", map_location='cpu')["asr_model"])
    dc = DurationCalculator(reduction_factor=1)
    tf = ArticulatoryCombinedTextFrontend(language="de", use_word_boundaries=False)
    #vad = VoiceActivityDetection(sample_rate=16000, trigger_time=0.0001, trigger_level=3.0, pre_trigger_time=0.2)


    # extract audio
    audio, sr = sf.read(audio_path)
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
    norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=audio)
    melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)

    phones = tf.string_to_tensor(transcript, handle_missing=False, view=False).squeeze()
    phone_string = tf.get_phone_string(transcript)
    alignment_path = acoustic_model.inference(mel=melspec,
                                                tokens=phones,
                                                save_img_for_debug=None,
                                                return_ctc=False)
    durations = dc(torch.LongTensor(alignment_path), vis=None).cpu()

    # total_dur = sum(durations) * 256 / sr
    # print(str(total_dur) + " seconds")
    # ignore pauses at start and end, exclude EOS token
    phones = phones[1:-2]
    durations = durations[1:-2]

    for phone, dur in zip (phones, durations):
        if phone[13] == 1:
            phone_durations.append(dur)
        if phone[14] == 1:
            pause_durations.append(dur)

    return phone_durations, pause_durations


def get_speaking_rate_and_pause_durs(path_to_transcript, hop_length=256, sr=16000):
    phone_durs = list()
    pause_durs = list()

    for audio_file, transcript in path_to_transcript.items():
        # print(audio_file)
        phones, pauses = get_avg_dur_phones(audio_file, transcript)
        # print(pauses)
        phone_durs += phones
        pause_durs += pauses
    print(pause_durs)
    
    # print(len(phone_durs))
    # print(len(pause_durs))

    # speaking rate = number of phonemes / total duration of phones, i.e. speaking rate without pauses
    # first convert frames to seconds
    time = sum(phone_durs) * hop_length / sr
    speaking_rate = len(phone_durs) / time # phones per second

    # avg length of pauses = total duration of pauses / number of pauses
    pause_time = sum(pause_durs) * hop_length / sr
    # print('pause time: ', pause_time)
    # print('num_pauses: ', len(pause_durs))
    avg_pause_length = pause_time / len(pause_durs)

    return speaking_rate, avg_pause_length

def get_pitch_stats(file_list):
    pitch_max = list()
    pitch_min = list()
    pitch_median = list()

    ap = AudioPreprocessor(cut_silence=True, input_sr=48000, output_sr=16000)
    parsel = Parselmouth(fs=16000, use_token_averaged_f0=False, use_log_f0=False, use_continuous_f0=False)

    for file in file_list:
        if not file.endswith(".wav"):
            continue
        # print(file)
        wave, sr = sf.read(file)
        norm_wave = ap.audio_to_wave_tensor(wave, normalize=True)
        pitch_curve = parsel(norm_wave.unsqueeze(0), norm_by_average=True)[0].squeeze()

        pitch_max.append(torch.max(pitch_curve).item())
        pitch_min.append(torch.min(pitch_curve).item())
        pitch_median.append(torch.median(pitch_curve).item())

    return pitch_max, pitch_min, pitch_median


""" Helper methods to sort list of files alphanumerically
    from https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
"""
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def path_to_transcript_dict(poem_dir):
    path_to_transcript = dict()
    
    with open(os.path.join(poem_dir, "transcript.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
        for line in lookup.split("\n"):
            if line.strip() != "":
                norm_transcript = line.split("\t")[1].rstrip().lstrip()
                wav_path = os.path.join(poem_dir, line.split("\t")[0])                  
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript

if __name__ == '__main__':
    import sys
    
    # for poem in os.listdir("/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data_2/Test"):
    # for poem in os.listdir("/Users/kockja/Documents/textklang/PlottingPoetry/audios/test/Karlsson/prose"):
    for poem in os.listdir("/Users/kockja/Documents/textklang/PlottingPoetry/Test"):
        print(poem)
        path_to_transcript = path_to_transcript_dict(f"/Users/kockja/Documents/textklang/PlottingPoetry/Test/{poem}")
        # path_to_transcript = path_to_transcript_dict(f"/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data_2/Test/{poem}")
        sr, pl = get_speaking_rate_and_pause_durs(path_to_transcript)
        print("speaking_rate:")
        print(sr.item())
        print("pause dur")
        print(pl.item())

    sys.exit(0)
    file_list_human = os.listdir("/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data_2/Test/W1_002_Das_Wunderhorn/")
    sort_nicely(file_list_human)
    path_list_human = ["/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data_2/Test/W1_002_Das_Wunderhorn/" + file for file in file_list_human]
    
    p_max_human, p_min_human, p_median_human = get_pitch_stats(path_list_human)

    print('HUMAN')
    print(p_max_human, "\n")
    #print(p_median_human)

    file_list_prose = os.listdir("audios/test/Karlsson/prose/W1_002_Das_Wunderhorn/")
    sort_nicely(file_list_prose)
    path_list_prose = ["audios/test/Karlsson/prose/W1_002_Das_Wunderhorn/" + file for file in file_list_prose]
    
    p_max_prose, p_min_prose, p_median_prose = get_pitch_stats(path_list_prose)

    print('PROSE')
    print(p_max_prose, "\n")
    #print(p_median_prose)

    file_list_poetry = os.listdir("audios/test/Karlsson/poetry/W1_002_Das_Wunderhorn/")
    sort_nicely(file_list_poetry)
    path_list_poetry = ["audios/test/Karlsson/poetry/W1_002_Das_Wunderhorn/" + file for file in file_list_poetry]
    
    p_max_poetry, p_min_poetry, p_median_poetry = get_pitch_stats(path_list_poetry)

    print('Poetry')
    print(p_max_poetry, "\n")
    #print(p_median_poetry)

    

