import soundfile as sf
import torch
import os
from run_utterance_cloner import UtteranceCloner
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

uc = UtteranceCloner(model_id="Zischler", device="cuda" if torch.cuda.is_available() else "cpu")
tts = InferenceFastSpeech2(model_name="Zischler", device="cuda" if torch.cuda.is_available() else "cpu")
tts.set_language('de')
tf = ArticulatoryCombinedTextFrontend(language='de', use_word_boundaries=False, use_explicit_eos=False, strip_silence=False)


# Set 1 - Part 1
def create_sample(transcript, path_to_ref_1, path_to_ref_2, filename_of_result, transplant_indices=[], clone_references=False, filename_of_clone_1=None, filename_of_clone_2=None):
    
    if clone_references:
        uc.clone_utterance(path_to_reference_audio=path_to_ref_1,
                            reference_transcription=transcript,
                            filename_of_result=filename_of_clone_1,
                            clone_speaker_identity=False,
                            lang="de")

        uc.clone_utterance(path_to_reference_audio=path_to_ref_2,
                            reference_transcription=transcript,
                            filename_of_result=filename_of_clone_2,
                            clone_speaker_identity=False,
                            lang="de")

    dur_1, pitch_1, en_1, silence_frames_start, silence_frames_end = uc.extract_prosody(transcript, path_to_ref_1, lang="de", on_line_fine_tune=True)
    dur_2, pitch_2, en_2, _, _ = uc.extract_prosody(transcript, path_to_ref_2, lang="de", on_line_fine_tune=True)

    dur = dur_1.detach().clone()
    pitch = pitch_1.detach().clone()
    en = en_1.detach().clone()

    for idx in transplant_indices:
        #print(dur_1[idx], " ", dur_2[idx])
        dur[idx] = dur_2[idx]
        pitch[idx] = pitch_2[idx]
        en[idx] = en_2[idx]

    start_sil = torch.zeros([silence_frames_start * 3]).to(tts.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
    end_sil = torch.zeros([silence_frames_end * 3]).to(tts.device)  # timestamps are from 16kHz, but now we're using 48kHz, so upsampling required
    cloned_speech = tts(transcript, view=False, durations=dur, pitch=pitch, energy=en)
    cloned_utt = torch.cat((start_sil, cloned_speech, end_sil), dim=0)
    sf.write(file=filename_of_result, data=cloned_utt.cpu().numpy(), samplerate=48000)

def set1_part1():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Brueckner
    # ref 2 = Quadflieg
    transcript = "Denn o saget, wo lebt menschliches Leben sonst, Da die knechtische jetzt alles, die Sorge, zwingt?"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set1', 'Hoelderlin_Die-Liebe_Brueckner_Strophe2_1.wav') 
    ref_2_path = os.path.join(root, 'Set1', 'Hoelderlin_Die-Liebe_Quadflieg_Strophe2_1.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p1_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p1_ref2.wav')

    # transplant prosody for "sonst, da die knechtische"
    indices = list(range(31, 49))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p1_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p1_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set1_part2():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Brueckner
    # ref 2 = Quadflieg
    transcript = "Darum wandelt der Gott auch ~ Sorglos über dem Haupt uns längst."

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set1', 'Hoelderlin_Die-Liebe_Brueckner_Strophe2_2.wav') 
    ref_2_path = os.path.join(root, 'Set1', 'Hoelderlin_Die-Liebe_Quadflieg_Strophe2_2.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p2_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p2_ref2.wav')

    # transplant prosody for "auch ~ Sorglos"
    indices = list(range(19, 30))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p2_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set1', 's1_p2_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set2_part1():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Moenckeberg
    # ref 2 = Zischler
    transcript = "Quellen hattest du ihm, hattest dem Flüchtigen ~ kühle Schatten geschenkt,"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set2', '1_2_Mönckeberg-Heidelberg-1978-Strophe5_1.wav') 
    ref_2_path = os.path.join(root, 'Set2', '1_2_Zischler-Heidelberg-2020-Strophe5_1.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p1_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p1_ref2.wav')

    # transplant prosody for "Flüchtigen kühle Schatten"
    indices = list(range(27, 46))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p1_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p1_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set2_part2():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Moenckeberg
    # ref 2 = Zischler
    transcript = "und die Gestade sahn ~ All, ihm nach,"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set2', '1_2_Mönckeberg-Heidelberg-1978-Strophe5_2.wav') 
    ref_2_path = os.path.join(root, 'Set2', '1_2_Zischler-Heidelberg-2020-Strophe5_2.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p2_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p2_ref2.wav')

    # transplant prosody for "sahn all"
    indices = list(range(13, 19))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p2_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p2_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set2_part3():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Moenckeberg
    # ref 2 = Zischler
    transcript = "und es bebte ~ Aus den Wellen ihr lieblich Bild."

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set2', '1_2_Mönckeberg-Heidelberg-1978-Strophe5_3.wav') 
    ref_2_path = os.path.join(root, 'Set2', '1_2_Zischler-Heidelberg-2020-Strophe5_3.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p3_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p3_ref2.wav')

    # transplant prosody for "bebte aus den Wellen"
    indices = list(range(6, 23))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p3_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set2', 's2_p3_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set3_part1():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Quadflieg
    # ref 2 = Mönckeberg
    transcript = "Wohin denn ich? Es leben die Sterblichen ~ Von Lohn und Arbeit;"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set3', 'Hoelderlin_Abendphantasie_Quadflieg_Strophe3_1.wav') 
    ref_2_path = os.path.join(root, 'Set3', 'Hölderlin_Abendphantasie_Mönckeberg_Strophe3_1.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p1_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p1_ref2.wav')

    # transplant prosody for "Sterblichen von Lohn"
    indices = list(range(21, 38))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p1_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p1_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set3_part2():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Quadflieg
    # ref 2 = Mönckeberg
    transcript = "wechselnd in Müh' und Ruh ~ Ist alles freudig;"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set3', 'Hoelderlin_Abendphantasie_Quadflieg_Strophe3_2.wav') 
    ref_2_path = os.path.join(root, 'Set3', 'Hölderlin_Abendphantasie_Mönckeberg_Strophe3_2.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p2_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p2_ref2.wav')

    # transplant prosody for "Ruh ist alles"
    indices = list(range(16, 26))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p2_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p2_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set3_part3():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Quadflieg
    # ref 2 = Mönckeberg
    transcript = "warum schläft denn ~ Nimmer nur mir in der Brust der Stachel?"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set3', 'Hoelderlin_Abendphantasie_Quadflieg_Strophe3_3.wav') 
    ref_2_path = os.path.join(root, 'Set3', 'Hölderlin_Abendphantasie_Mönckeberg_Strophe3_3.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p3_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p3_ref2.wav')

    # transplant prosody for "denn nimmer"
    indices = list(range(11, 19))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p3_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set3', 's3_p3_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set4_part1():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Mönckeberg
    # ref 2 = Quadflieg
    transcript = "Willkommen dann, o Stille der Schattenwelt! ~ Zufrieden bin ich, wenn auch mein Saitenspiel ~ Mich nicht hinab geleitet;"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set4', 'Hoelderlin_An-die-Parzen_Moenckeberg_Strophe3_1.wav') 
    ref_2_path = os.path.join(root, 'Set4', 'Hoelderlin_An-die-Parzen_Quadflieg_Strophe3_1.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p1_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p1_ref2.wav')

    # transplant prosody for "Saitenspiel mich"
    indices = list(range(58, 72))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p1_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p1_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set4_part2():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Mönckeberg
    # ref 2 = Quadflieg
    transcript = "Einmal ~ Lebt ich, wie Götter, und mehr bedarfs nicht."

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set4', 'Hoelderlin_An-die-Parzen_Moenckeberg_Strophe3_2.wav') 
    ref_2_path = os.path.join(root, 'Set4', 'Hoelderlin_An-die-Parzen_Quadflieg_Strophe3_2.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p2_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p2_ref2.wav')

    # transplant prosody for "Einmal lebt"
    indices = list(range(0, 12))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p2_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set4', 's4_p2_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set5_part1():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Mönckeberg
    # ref 2 = Quadflieg
    transcript = "Laß endlich, Vater! offenen Aug's mich dir ~ Begegnen!"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set5', 'Hoelderlin_Der-Zeitgeist_Moenckeberg_Strophe3_1.wav') 
    ref_2_path = os.path.join(root, 'Set5', 'Hoelderlin_Der-Zeitgeist_Quadflieg_Strophe3_1.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p1_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p1_ref2.wav')

    # transplant prosody for "dir begegnen"
    indices = list(range(29, 41))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p1_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p1_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set5_part2():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Mönckeberg
    # ref 2 = Quadflieg
    transcript = "hast denn du nicht zuerst den Geist ~ Mit deinem Strahl aus mir geweckt?"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set5', 'Hoelderlin_Der-Zeitgeist_Moenckeberg_Strophe3_2.wav') 
    ref_2_path = os.path.join(root, 'Set5', 'Hoelderlin_Der-Zeitgeist_Quadflieg_Strophe3_2.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p2_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p2_ref2.wav')

    # transplant prosody for "Geist mit deinem Strahl"
    indices = list(range(24, 44))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p2_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p2_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)

def set5_part3():
    root = "/projekte/textklang/TTS-Interspeech/Recordings-for-experiment/User_Study"

    # Set1 Part1
    # ref 1 = Mönckeberg
    # ref 2 = Quadflieg
    transcript = "mich ~ Herrlich ans Leben gebracht, o Vater!"

    # phones = tf.get_phone_string(transcript)
    # for i, phone in enumerate(phones):
    #     print(phone, ": ", i)

    ref_1_path = os.path.join(root, 'Set5', 'Hoelderlin_Der-Zeitgeist_Moenckeberg_Strophe3_3.wav') 
    ref_2_path = os.path.join(root, 'Set5', 'Hoelderlin_Der-Zeitgeist_Quadflieg_Strophe3_3.wav')

    clone_path_1 = os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p3_ref1.wav')
    clone_path_2 = os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p3_ref2.wav')

    # transplant prosody for "Geist mit deinem Strahl"
    indices = list(range(0, 11))
    create_sample(transcript, ref_1_path, ref_2_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p3_base1_pros_2.wav'), transplant_indices=indices, clone_references=True, filename_of_clone_1=clone_path_1, filename_of_clone_2=clone_path_2)
    create_sample(transcript, ref_2_path, ref_1_path, filename_of_result=os.path.join('audios', 'PoetryStudy', 'Set5', 's5_p3_base2_pros_1.wav'), transplant_indices=indices, clone_references=False)



if __name__ == '__main__':
    set1_part1()
    set1_part2()
    set2_part1()
    set2_part2()
    set2_part3()
    set3_part1()
    set3_part2()
    set3_part3()
    set4_part1()
    set4_part2()
    set5_part1()
    set5_part2()
    set5_part3()