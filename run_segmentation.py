import os
import sys
import re
import soundfile as sf
from numpy import trim_zeros
from torchaudio.transforms import Vad as VoiceActivityDetection
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator


def split_text(text, n=10):
    stanzas = text.split('\n\n')
    cutted = []
    for stanza in stanzas:
        lines = stanza.split('\n')
        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        if len(lines) <= n:
            cutted.append(stanza)
        else:
            while len(lines) > n:
                first_n = '\n'.join(lines[:n])
                lines = lines[n:]
                cutted.append(first_n)
            if len(lines) > 0:
                cutted.append('\n'.join(lines))
    return list(filter(lambda x: not re.match(r'^\s*$', x), cutted))

def split_audio(audio_path, text_path, out_dir, max_num_lines=10, n=20):
    # loading modules
    acoustic_model = Aligner()
    acoustic_model.load_state_dict(torch.load("Models/Aligner/aligner.pt", map_location='cpu')["asr_model"])
    dc = DurationCalculator(reduction_factor=1)
    tf = ArticulatoryCombinedTextFrontend(language="de")
    vad = VoiceActivityDetection(sample_rate=16000, trigger_time=0.0001, trigger_level=3.0, pre_trigger_time=0.2)

    os.makedirs(out_dir, exist_ok=True)


    # extract audio
    audio, sr = sf.read(audio_path)
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
    norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=audio)
    melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)


    # extract phonemes
    lines = list()
    with open(text_path, "r", encoding="utf8") as text_file:
        text = text_file.read()
        stanzas = split_text(text, max_num_lines)
        stanzas = [s.replace('\n', ' ') for s in stanzas]

        for line in stanzas:
            lines.append(tf.string_to_tensor(line, handle_missing=False).squeeze())
    # postprocess phonemes: [~ sentence ~ #] --> [sentence ~] except for the first one, which is [~ sentence ~]
    processed_lines = list()
    for index, line in enumerate(lines):
        if index == 0:
            processed_lines.append(line[:-1])
        else:
            processed_lines.append(line[1:-1])
    lines = processed_lines
    joined_phonemes = torch.cat(lines, dim=0)

    # get durations of each phone in audio as average of an ensemble
    alignment_paths = list()
    ensemble_of_durations = list()
    for _ in tqdm(range(n)):
        alignment_paths.append(acoustic_model.inference(mel=melspec,
                                                        tokens=joined_phonemes,
                                                        save_img_for_debug=os.path.join(out_dir, "debug_alignment.png"),
                                                        return_ctc=False))
    for alignment_path in alignment_paths:
        ensemble_of_durations.append(dc(torch.LongTensor(alignment_path), vis=None).squeeze())
    durations = list()
    for i, _ in enumerate(ensemble_of_durations[0]):
        duration_of_phone = list()
        for ensemble_member in ensemble_of_durations:
            duration_of_phone.append(ensemble_member.squeeze()[i])
        durations.append(sum(duration_of_phone) / len(duration_of_phone))

    # cut audio according to duration sum of each line in transcript
    line_lens = [len(x) for x in lines]
    index = 0
    segment_durations = list()
    for num_phones in line_lens:
        segment_durations.append(sum(durations[index: index + num_phones]))
        index += num_phones
    spec_to_wave_factor = len(norm_wave) / sum(segment_durations)
    wave_segment_lens = [int(x * spec_to_wave_factor) for x in segment_durations]
    start_index = 0
    wave_segments = list()
    for index, segment_len in enumerate(wave_segment_lens):
        if index == len(wave_segment_lens) - 1:
            wave_segments.append(norm_wave[start_index:])
        else:
            wave_segments.append(norm_wave[start_index: start_index + segment_len])
            start_index += segment_len

    with open(os.path.join(out_dir, "transcript.txt"), 'w') as transcript:
        # write the segments into new files
        assert len(wave_segments) == len(stanzas)
        for f, item in enumerate(zip(wave_segments, stanzas)):
            wave_segment = item[0]
            line = item[1]
            # print(line, " ", f)
            transcript.write(f"{f}.wav\t{line}\n")
            silence = torch.zeros([40000])
            no_silence_front = vad(torch.cat((silence, torch.Tensor(wave_segment), silence), 0))
            reversed_audio = torch.flip(no_silence_front, (0,))
            no_silence_back = vad(torch.Tensor(reversed_audio))
            unreversed_audio = torch.flip(no_silence_back, (0,))
            sf.write(os.path.join(out_dir, f"{f}.wav"), trim_zeros(unreversed_audio.numpy()), 16000)


if __name__ == '__main__':
    txt_root = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/txt/Wunderhorn"
    audio_root = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/wavs/Wunderhorn"

    for wav in os.listdir(audio_root):
        poem = wav.split('.')[0]
        print(poem)
        audio_path = os.path.join(audio_root, wav)
        text_path = os.path.join(txt_root, f"{poem}.txt")
        out_dir = os.path.join("/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data", poem)
        split_audio(audio_path, text_path, out_dir, max_num_lines=4, n=10)

    # audio_path = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/wavs/Wunderhorn/W3_K_027_Der_wunderliche_Kittel.wav"
    # text_path = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/txt/Wunderhorn/W3_K_027_Der_wunderliche_Kittel.txt"
    # out_dir = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/Test"
   
        