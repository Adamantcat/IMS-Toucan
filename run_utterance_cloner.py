import os

import soundfile as sf
import torch
from torch.optim import SGD
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Parselmouth


class UtteranceCloner:

    def __init__(self, model_id, device):
        self.tts = InferenceFastSpeech2(device=device, model_name=model_id)
        self.device = device
        acoustic_checkpoint_path = os.path.join("Models", "Aligner", "aligner.pt")
        self.aligner_weights = torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"]
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
        # careful: assumes 16kHz or 8kHz audio
        self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False,
                                                  verbose=False)
        (self.get_speech_timestamps, _, _, _, _) = utils
        torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
        # this to false globally during model loading rather than using inference mode or no_grad

    def extract_prosody(self, transcript, ref_audio_path, lang="de", on_line_fine_tune=True, input_is_phones=False):
        acoustic_model = Aligner()
        acoustic_model.load_state_dict(self.aligner_weights)
        acoustic_model = acoustic_model.to(self.device)
        parsel = Parselmouth(reduction_factor=1, fs=16000)
        energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        dc = DurationCalculator(reduction_factor=1)
        wave, sr = sf.read(ref_audio_path)
        tf = ArticulatoryCombinedTextFrontend(language=lang)
        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        try:
            norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            print('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        start_silence = speech_timestamps[0]['start']
        end_silence = len(norm_wave) - speech_timestamps[-1]['end']
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=input_is_phones).squeeze(0)
        melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
        melspec_length = torch.LongTensor([len(melspec)]).numpy()

        if on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 3
            tokens = list()  # we need an ID sequence for training rather than a sequence of phonological features
            for vector in text:
                if vector[19] == 0:  # we don't include word boundaries when performing alignment, since they are not always present in audio.
                    for phone in tf.phone_to_vector:
                        if vector.numpy().tolist()[11:] == tf.phone_to_vector[phone][11:]:
                            # the first 10 dimensions are for modifiers, so we ignore those when trying to find the phoneme in the ID lookup
                            tokens.append(tf.phone_to_id[phone])
                            # this is terribly inefficient, but it's fine
                            break
            tokens = torch.LongTensor(tokens).squeeze().to(self.device)
            tokens_len = torch.LongTensor([len(tokens)]).to(self.device)
            mel = melspec.unsqueeze(0).to(self.device)
            mel.requires_grad = True
            mel_len = torch.LongTensor([len(mel[0])]).to(self.device)
            # actual fine-tuning starts here
            optim_asr = SGD(acoustic_model.parameters(), lr=0.1)
            acoustic_model.train()
            for _ in list(range(steps)):
                pred = acoustic_model(mel)
                loss = acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
                optim_asr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(acoustic_model.parameters(), 1.0)
                optim_asr.step()
            acoustic_model.eval()

        # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
        # We note the index of word boundaries and insert durations of 0 afterwards
        text_without_word_boundaries = list()
        indexes_of_word_boundaries = list()
        for phoneme_index, vector in enumerate(text):
            if vector[19] == 0:
                text_without_word_boundaries.append(vector.numpy().tolist())
            else:
                indexes_of_word_boundaries.append(phoneme_index)
        matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

        alignment_path = acoustic_model.inference(mel=melspec.to(self.device),
                                                  tokens=matrix_without_word_boundaries.to(self.device),
                                                  return_ctc=False)

        duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

        for index_of_word_boundary in indexes_of_word_boundaries:
            duration = torch.cat([duration[:index_of_word_boundary],
                                  torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                  duration[index_of_word_boundary:]])

        last_vec = None
        for phoneme_index, vec in enumerate(text):
            if last_vec is not None:
                if last_vec.numpy().tolist() == vec.numpy().tolist():
                    # we found a case of repeating phonemes!
                    # now we must repair their durations by giving the first one 3/5 of their sum and the second one 2/5 (i.e. the rest)
                    dur_1 = duration[phoneme_index - 1]
                    dur_2 = duration[phoneme_index]
                    total_dur = dur_1 + dur_2
                    new_dur_1 = int((total_dur / 5) * 3)
                    new_dur_2 = total_dur - new_dur_1
                    duration[phoneme_index - 1] = new_dur_1
                    duration[phoneme_index] = new_dur_2
            last_vec = vec

        energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                             input_waves_lengths=norm_wave_length,
                             feats_lengths=melspec_length,
                             text=text,
                             durations=duration.unsqueeze(0),
                             durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        pitch = parsel(input_waves=norm_wave.unsqueeze(0),
                       input_waves_lengths=norm_wave_length,
                       feats_lengths=melspec_length,
                       text=text,
                       durations=duration.unsqueeze(0),
                       durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
        return duration, pitch, energy, start_silence, end_silence

    def clone_utterance(self,
                        path_to_reference_audio,
                        reference_transcription,
                        clone_speaker_identity=True,
                        lang="de",
                        input_is_phones=False):
        if clone_speaker_identity:
            prev_speaker_embedding = self.tts.default_utterance_embedding.clone().detach()
            self.tts.set_utterance_embedding(path_to_reference_audio=path_to_reference_audio)
        duration, pitch, energy, silence_frames_start, silence_frames_end = self.extract_prosody(reference_transcription,
                                                                                                 path_to_reference_audio,
                                                                                                 lang=lang,
                                                                                                 input_is_phones=input_is_phones)
        self.tts.set_language(lang)
        start_sil = torch.zeros([silence_frames_start * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        end_sil = torch.zeros([silence_frames_end * 3]).to(self.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
        cloned_speech = self.tts(reference_transcription, view=False, durations=duration, pitch=pitch, energy=energy,
                                 input_is_phones=input_is_phones)
        cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0)
        if clone_speaker_identity:
            self.tts.default_utterance_embedding = prev_speaker_embedding.to(self.device)  # return to normal
        return cloned_utt


if __name__ == '__main__':
    tf = ArticulatoryCombinedTextFrontend(language='de')
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")

    # transcript = "In deinen Tälern wachte mein Herz mir auf Zum Leben, deine Wellen umspielten mich,"
    # transcript = "Wunderbar ist die Gunst der Hocherhabnen und niemand Weiß von wannen und was einem geschiehet von ihr."
    
    # uc.clone_utterance(path_to_reference_audio="/Users/kockja/Documents/textklang/ICPhS/Brod_und_Wein_s02.wav",
    #                    reference_transcription=transcript,
    #                    filename_of_result="/Users/kockja/Documents/textklang/ICPhS/synthese/brod_cloned.wav",
    #                    clone_speaker_identity=True,
    #                    lang="de")

    # phones = tf.get_phone_string(transcript)
    # dur, pitch, en, _, _ = uc.extract_prosody(transcript=transcript, ref_audio_path="/Users/kockja/Documents/textklang/ICPhS/Brod_und_Wein_s02.wav", lang='de', on_line_fine_tune=True)

    # for i, (p, d)  in enumerate(zip(phones, dur)):
    #     print(i, "\t", p, "\t", d)

    # # lengthening niemand
    # dur_len = dur

    # dur_len[46] = dur[46] + 4
    # dur_len[47] = dur[47] + 4
    # dur_len[48] = dur[48] + 4

    # for i, d in enumerate(dur_len):
    #     print(i, "  ", d)
    # tts = uc.tts
    # tts.set_language('de')
    # tts.set_utterance_embedding("/Users/kockja/Documents/textklang/ICPhS/Brod_und_Wein_s02.wav")
    # tts.read_to_file(text_list=[transcript], dur_list=[dur_len], pitch_list=[pitch], energy_list=[en], file_location="/Users/kockja/Documents/textklang/ICPhS/synthese/brod_len+4.wav")


    # transcript = "Aber der Muth von ihnen ist groß, es füllen das Herz ihm Ihre Freuden und kaum weiß er zu brauchen das Gut,"


    # uc.clone_utterance(path_to_reference_audio="/Users/kockja/Documents/textklang/ICPhS/Brod_und_Wein_s05.wav",
    #                    reference_transcription=transcript,
    #                    filename_of_result="/Users/kockja/Documents/textklang/ICPhS/synthese/brod_s05_ihm_cloned.wav",
    #                    clone_speaker_identity=True,
    #                    lang="de")

    # phones = tf.get_phone_string(transcript)
    # dur, pitch, en, _, _ = uc.extract_prosody(transcript=transcript, ref_audio_path="/Users/kockja/Documents/textklang/ICPhS/Brod_und_Wein_s05.wav", lang='de', on_line_fine_tune=True)

    # for i, (p, d)  in enumerate(zip(phones, dur)):
    #     print(i, "\t", p, "\t", d)

    # # lengthening ihm
    # dur_len = dur

    # dur_len[51] = dur[51] - 4
    # dur_len[52] = dur[52] - 4

    # for i, d in enumerate(dur_len):
    #     print(i, "  ", d)
    # tts = uc.tts
    # tts.set_language('de')
    # tts.set_utterance_embedding("/Users/kockja/Documents/textklang/ICPhS/Brod_und_Wein_s05.wav")
    # tts.read_to_file(text_list=[transcript], dur_list=[dur_len], pitch_list=[pitch], energy_list=[en], file_location="/Users/kockja/Documents/textklang/ICPhS/synthese/brod_s05_ihm_len-4.wav")

    transcript = "In deinen Thälern wachte mein Herz mir auf Zum Leben, deine Wellen umspielten mich,"


    wav = uc.clone_utterance(path_to_reference_audio="/Users/kockja/Documents/textklang/ICPhS/original/Neckar_s01_auf.wav",
                       reference_transcription=transcript,
                       clone_speaker_identity=True,
                       input_is_phones=False,
                       lang="de")
    sf.write(file="/Users/kockja/Documents/textklang/ICPhS/synthese_v3.2/Neckar_s01_auf.wav", data=wav.cpu().numpy(), samplerate=48000)

    # phones = tf.get_phone_string(transcript)
    # dur, pitch, en, _, _ = uc.extract_prosody(transcript=transcript, ref_audio_path="/Users/kockja/Documents/textklang/ICPhS/Neckar_s07.wav", lang='de', on_line_fine_tune=True)

    # for i, (p, d)  in enumerate(zip(phones, dur)):
    #     print(i, "\t", p, "\t", d)

    # # lengthening armen
    # dur_len = dur

    # dur_len[33] = dur[33] + 4
    # dur_len[34] = dur[34] + 4

    # for i, d in enumerate(dur_len):
    #     print(i, "  ", d)
    # tts = uc.tts
    # tts.set_language('de')
    # tts.set_utterance_embedding("/Users/kockja/Documents/textklang/ICPhS/Neckar_s07.wav")
    # tts.read_to_file(text_list=[transcript], dur_list=[dur_len], pitch_list=[pitch], energy_list=[en], file_location="/Users/kockja/Documents/textklang/ICPhS/synthese/neckar_s07_armen+4.wav")

