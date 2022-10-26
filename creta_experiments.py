import torch
import numpy
import os
from run_utterance_cloner import UtteranceCloner
from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
import sys

def clone_all(root, wavs, texts):
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")

    for wav, text in zip(wavs, texts):
        out_file = wav.replace('original', 'AA')

        uc.clone_utterance(path_to_reference_audio=os.path.join(root, wav),
                        reference_transcription=text,
                        filename_of_result=os.path.join(root, out_file),
                        clone_speaker_identity=True,
                        lang="de")

# Vers 1: O weh! nicht weiter sag!
def stündlein_s03_v01(root):
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")
    tf = ArticulatoryCombinedTextFrontend(language='de')
    # Ein Stündlein wohl vor Tag
    # root = "/Users/kockja/Documents/textklang/Creta_2022/tts-ayd-0081-m01-s01-t03-v01"
    audio_A = "Ein_Stündlein_s03_v01_original.wav"
    audio_B = "Ein_Stündlein_s03_v04_original.wav"

    #Vers 1: O weh! nicht weiter sag!
    text_A = "O weh! nicht weiter sag!"
    text_B = "Ach, Lieb und Treu ist wie ein Traum"

    dur_A, pitch_A, en_A, _, _ = uc.extract_prosody(text_A, os.path.join(root, audio_A), lang="de", on_line_fine_tune=True)
    dur_B, pitch_B, en_B, _, _ = uc.extract_prosody(text_B, os.path.join(root, audio_B), lang='de', on_line_fine_tune=True)

    phones_A = tf.get_phone_string(text_A)
    phones_B = tf.get_phone_string(text_B)

    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_A, pitch_A, en_A)):
    #     print(f"{i}  {phone}  {dur} {pitch} {en}")
    # print()
    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_B, dur_B, pitch_B, en_B)):
    #     print(f"{i}  {phone}  {dur} {pitch} {en}")
    # print()

    text_AB = "Ach weh! nicht weiter sag!"

    phones_AB = tf.get_phone_string(text_AB)

    # BB
    dur_BB = torch.cat([dur_B[:3], dur_A[2:]], 0)
    pitch_BB = torch.cat([pitch_B[:3], pitch_A[2:]], 0)
    en_BB = torch.cat([en_B[:3], en_A[2:]], 0)

    assert(len(phones_AB) == len(dur_BB) == len(pitch_BB) == len(en_BB))

    #for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BB, pitch_BB, en_BB)):
    #    print(f"{i} {phone} {dur}   {pitch} {en}")

    # AB
    # o weh nicht weiter sag
    dur_o = torch.add(dur_B[1], dur_B[2].item()) # dur vom 'o' = dur 'a' + dur 'x'
    #print(dur_o)
    dur_AB = torch.cat([dur_B[:1], dur_o.reshape(1), dur_A[2:]], 0) # dur vom 'o' = dur 'a' + dur 'x'
    # dur_AB = torch.cat([dur_B[:2], dur_A[2:]], 0) # nur dur vom 'a'
    pitch_AB = torch.cat([pitch_B[:2], pitch_A[2:]], 0) # nur das a vom 'ach', das x hat keinen pitch
    en_AB = torch.cat([en_B[:2], en_A[2:]], 0) # nur das a vom 'ach', das x hat keine energy

    assert(len(phones_A) == len(dur_AB) == len(pitch_AB) == len(en_AB))

    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_AB, pitch_AB, en_AB)):
    #     print(f"{i}  {phone}  {dur} {pitch} {en}")

    # BA
    # Ach weh nicht weiter sag
    dur_BA = torch.cat([dur_A[:2], dur_B[2:3], dur_A[2:]], 0) #dur o, dur x (vom ach), ...
    pitch_BA = torch.cat([pitch_A[:2], pitch_B[2:3], pitch_A[2:]], 0)
    en_BA = torch.cat([en_A[:2], en_B[2:3], en_A[2:]], 0)

    assert(len(phones_AB) == len(dur_BA) == len(pitch_BA) == len(en_BA))
    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BA, pitch_BA, en_BA)):
    #    print(f"{i}  {phone}  {dur} {pitch} {en}")

    tts = uc.tts
    tts.set_language('de')
    tts.set_utterance_embedding(os.path.join(root, audio_A))
    # AA
    tts.read_to_file(text_list=[text_A], dur_list=[dur_A], pitch_list=[pitch_A], energy_list=[en_A], file_location=os.path.join(root, "Ein_Stündlein_s03_v01_AA.wav"))
    # BB
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BB], pitch_list=[pitch_BB], energy_list=[en_BB], file_location=os.path.join(root, "Ein_Stündlein_s03_v01_BB.wav"))
    # AB
    tts.read_to_file(text_list=[text_A], dur_list=[dur_AB], pitch_list=[pitch_AB], energy_list=[en_AB], file_location=os.path.join(root, "Ein_Stündlein_s03_v01_AB.wav"))
    # BA
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BA], pitch_list=[pitch_BA], energy_list=[en_BA], file_location=os.path.join(root, "Ein_Stündlein_s03_v01_BA.wav"))

#Vers 2 "O still! nichts hören mag!"
def stündlein_s03_v02(root):
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")
    tf = ArticulatoryCombinedTextFrontend(language='de')
    # Ein Stündlein wohl vor Tag
   #  root = "/Users/kockja/Documents/textklang/Creta_2022/tts-ayd-0081-m01-s01-t03-v01"
    audio_A = "Ein_Stündlein_s03_v02_original.wav"
    audio_B = "Ein_Stündlein_s03_v04_original.wav"

    text_A = "O still! nichts hören mag!"
    text_B = "Ach, Lieb und Treu ist wie ein Traum"

    dur_A, pitch_A, en_A, _, _ = uc.extract_prosody(text_A, os.path.join(root, audio_A), lang="de", on_line_fine_tune=True)
    dur_B, pitch_B, en_B, _, _ = uc.extract_prosody(text_B, os.path.join(root, audio_B), lang='de', on_line_fine_tune=True)

    phones_A = tf.get_phone_string(text_A)
    phones_B = tf.get_phone_string(text_B)

    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_A, pitch_A, en_A)):
    #     print(f"{i}  {phone}  {dur} {pitch} {en}")
    # print()
    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_B, dur_B, pitch_B, en_B)):
    #     print(f"{i}  {phone}  {dur} {pitch} {en}")
    # print()

    text_AB = "Ach still! nichts hören mag!"
    phones_AB = tf.get_phone_string(text_AB)

    # BB
    dur_BB = torch.cat([dur_B[:3], dur_A[2:]], 0)
    pitch_BB = torch.cat([pitch_B[:3], pitch_A[2:]], 0)
    en_BB = torch.cat([en_B[:3], en_A[2:]], 0)

    assert(len(phones_AB) == len(dur_BB) == len(pitch_BB) == len(en_BB))

    #for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BB, pitch_BB, en_BB)):
    #    print(f"{i} {phone} {dur}   {pitch} {en}")

    # AB
    dur_o = torch.add(dur_B[1], dur_B[2].item()) # dur vom 'o' = dur 'a' + dur 'x'
    #print(dur_o)
    dur_AB = torch.cat([dur_B[:1], dur_o.reshape(1), dur_A[2:]], 0) # dur vom 'o' = dur 'a' + dur 'x'
    # dur_AB = torch.cat([dur_B[:2], dur_A[2:]], 0) # nur dur vom 'a'
    pitch_AB = torch.cat([pitch_B[:2], pitch_A[2:]], 0) # nur das a vom 'ach', das x hat keinen pitch
    en_AB = torch.cat([en_B[:2], en_A[2:]], 0) # nur das a vom 'ach', das x hat keine energy

    assert(len(phones_A) == len(dur_AB) == len(pitch_AB) == len(en_AB))

    # for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_AB, pitch_AB, en_AB)):
    #     print(f"{i}  {phone}  {dur} {pitch} {en}")

    # BA
    # Ach weh nicht weiter sag
    dur_BA = torch.cat([dur_A[:2], dur_B[2:3], dur_A[2:]], 0) #dur o, dur x (vom ach), ...
    pitch_BA = torch.cat([pitch_A[:2], pitch_B[2:3], pitch_A[2:]], 0)
    en_BA = torch.cat([en_A[:2], en_B[2:3], en_A[2:]], 0)

    assert(len(phones_AB) == len(dur_BA) == len(pitch_BA) == len(en_BA))
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BA, pitch_BA, en_BA)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")

    tts = uc.tts
    tts.set_language('de')
    tts.set_utterance_embedding(os.path.join(root, audio_A))
    # AA
    tts.read_to_file(text_list=[text_A], dur_list=[dur_A], pitch_list=[pitch_A], energy_list=[en_A], file_location=os.path.join(root, "Ein_Stündlein_s03_v02_AA.wav"))
    # BB
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BB], pitch_list=[pitch_BB], energy_list=[en_BB], file_location=os.path.join(root, "Ein_Stündlein_s03_v02_BB.wav"))
    # AB
    tts.read_to_file(text_list=[text_A], dur_list=[dur_AB], pitch_list=[pitch_AB], energy_list=[en_AB], file_location=os.path.join(root, "Ein_Stündlein_s03_v02_AB.wav"))
    # BA
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BA], pitch_list=[pitch_BA], energy_list=[en_BA], file_location=os.path.join(root, "Ein_Stündlein_s03_v02_BA.wav"))

#  Vers 4: Ach, Lieb und Treu ist wie ein Traum
def stündlein_s03_v04(root):
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")
    tf = ArticulatoryCombinedTextFrontend(language='de')
    # Ein Stündlein wohl vor Tag
   # root = "/Users/kockja/Documents/textklang/Creta_2022/tts-ayd-0081-m01-s01-t03-v01"
    audio_A = "Ein_Stündlein_s03_v04_original.wav"
    audio_B = "Ein_Stündlein_s03_v01_original.wav"

    text_A = "Ach, Lieb und Treu~ist wie ein Traum"
    text_B = "O weh! nicht weiter sag!"

    dur_A, pitch_A, en_A, _, _ = uc.extract_prosody(text_A, os.path.join(root, audio_A), lang="de", on_line_fine_tune=True)
    dur_B, pitch_B, en_B, _, _ = uc.extract_prosody(text_B, os.path.join(root, audio_B), lang='de', on_line_fine_tune=True)

    phones_A = tf.get_phone_string(text_A)
    phones_B = tf.get_phone_string(text_B)

    for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_A, pitch_A, en_A)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")
    print()
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_B, dur_B, pitch_B, en_B)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")
    print()

    text_AB = "O, Lieb und Treu~ist wie ein Traum"
    phones_AB = tf.get_phone_string(text_AB)

    # BB
    dur_BB = torch.cat([dur_B[:2], dur_A[3:]], 0)
    pitch_BB = torch.cat([pitch_B[:2], pitch_A[3:]], 0)
    en_BB = torch.cat([en_B[:2], en_A[3:]], 0)

    assert(len(phones_AB) == len(dur_BB) == len(pitch_BB) == len(en_BB))
    print('BB')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BB, pitch_BB, en_BB)):
        print(f"{i} {phone} {dur}   {pitch} {en}")

    # AB  - Ach lieb und treu, prosodie vom O
    dur_AB = torch.cat([dur_B[:2], dur_A[2:]], 0) #dur o, dur x (vom ach), ...
    pitch_AB = torch.cat([pitch_B[:2], pitch_A[2:]], 0)
    en_AB = torch.cat([en_B[:2], en_A[2:]], 0)

    assert(len(phones_A) == len(dur_AB) == len(pitch_AB) == len(en_AB))
    print('AB')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_AB, pitch_AB, en_AB)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")

    # BA - O lieb und treu, Prosodie vom Ach
    dur_o = torch.add(dur_A[1], dur_A[2].item()) # dur vom 'o' = dur 'a' + dur 'x'
    #print(dur_o)
    dur_BA = torch.cat([dur_A[:1], dur_o.reshape(1), dur_A[3:]], 0) # dur vom 'o' = dur 'a' + dur 'x'
    # dur_AB = torch.cat([dur_B[:2], dur_A[2:]], 0) # nur dur vom 'a'
    pitch_BA = torch.cat([pitch_A[:1], pitch_A[1:2], pitch_A[3:]], 0) 
    en_BA = torch.cat([en_A[:1], en_A[1:2], en_A[3:]], 0) 

    assert(len(phones_AB) == len(dur_BA) == len(pitch_BA) == len(en_BA))
    print('BA')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BA, pitch_BA, en_BA)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")

    tts = uc.tts
    tts.set_language('de')
    tts.set_utterance_embedding(os.path.join(root, audio_A))
    # AA
    tts.read_to_file(text_list=[text_A], dur_list=[dur_A], pitch_list=[pitch_A], energy_list=[en_A], file_location=os.path.join(root, "Ein_Stündlein_s03_v04_AA.wav"))
    # BB
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BB], pitch_list=[pitch_BB], energy_list=[en_BB], file_location=os.path.join(root, "Ein_Stündlein_s03_v04_BB.wav"))
    # AB
    tts.read_to_file(text_list=[text_A], dur_list=[dur_AB], pitch_list=[pitch_AB], energy_list=[en_AB], file_location=os.path.join(root, "Ein_Stündlein_s03_v04_AB.wav"))
    # BA
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BA], pitch_list=[pitch_BA], energy_list=[en_BA], file_location=os.path.join(root, "Ein_Stündlein_s03_v04_BA.wav"))

def sehnsucht_s01_v04(root):
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")
    tf = ArticulatoryCombinedTextFrontend(language='de')
    # Ein Stündlein wohl vor Tag
    # root = "/Users/kockja/Documents/textklang/Creta_2022/tts-ayd-0081-m01-s01-t03-v01"
    audio_A = "Sehnsucht_s01_v04_original.wav"
    audio_B = "Sehnsucht_s03_v04_original.wav"

    text_A = "Ach wie fühlt ich mich beglückt!"
    text_B = "O~wie labend muß sie sein!"

    dur_A, pitch_A, en_A, _, _ = uc.extract_prosody(text_A, os.path.join(root, audio_A), lang="de", on_line_fine_tune=True)
    dur_B, pitch_B, en_B, _, _ = uc.extract_prosody(text_B, os.path.join(root, audio_B), lang='de', on_line_fine_tune=True)

    phones_A = tf.get_phone_string(text_A)
    phones_B = tf.get_phone_string(text_B)

    for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_A, pitch_A, en_A)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")
    print()
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_B, dur_B, pitch_B, en_B)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")
    print()

    text_AB = "O wie fühlt ich mich beglückt!"

    phones_AB = tf.get_phone_string(text_AB)

    # BB
    dur_o = torch.add(dur_B[1], dur_B[2].item()) # o + hilfspause
    dur_BB = torch.cat([dur_B[:2], dur_A[3:]], 0)
    dur_BB[1] = dur_o
    pitch_BB = torch.cat([pitch_B[:2], pitch_A[3:]], 0)
    en_BB = torch.cat([en_B[:2], en_A[3:]], 0)

    assert(len(phones_AB) == len(dur_BB) == len(pitch_BB) == len(en_BB))

    print('BB')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BB, pitch_BB, en_BB)):
       print(f"{i} {phone} {dur}   {pitch} {en}")

    # AB - Ach wie fühl ich mich beglückt!; Prosody vom O
    dur_o = torch.add(dur_B[1], dur_B[2].item()) # o + hilfspause
    dur_AB = torch.cat([dur_B[:2], dur_A[2:]], 0) #dur o, dur x (vom ach), ...
    dur_AB[1] = dur_o
    pitch_AB = torch.cat([pitch_B[:2], pitch_A[2:]], 0)
    en_AB = torch.cat([en_B[:2], en_A[2:]], 0)
    
    assert(len(phones_A) == len(dur_AB) == len(pitch_AB) == len(en_AB))
    print('AB')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_AB, pitch_AB, en_AB)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")

    # BA - O wie fühlt ich mich beglückt!
    dur_o = torch.add(dur_A[1], dur_A[2].item()) # dur vom 'o' = dur 'a' + dur 'x'
    print(dur_o)
    dur_BA = torch.cat([dur_A[:1], dur_o.reshape(1), dur_A[3:]], 0) # dur vom 'o' = dur 'a' + dur 'x'
    # dur_BA = torch.cat([dur_B[:2], dur_A[2:]], 0) # nur dur vom 'a'
    pitch_BA = torch.cat([pitch_A[:2], pitch_A[3:]], 0) ###### Nochmal prüfen
    en_BA = torch.cat([en_A[:2], en_A[3:]], 0)


    assert(len(phones_AB) == len(dur_BA) == len(pitch_BA) == len(en_BA))
    print('BA')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BA, pitch_BA, en_BA)):
       print(f"{i}  {phone}  {dur} {pitch} {en}")

    tts = uc.tts
    tts.set_language('de')
    tts.set_utterance_embedding(os.path.join(root, audio_A))
    # AA
    tts.read_to_file(text_list=[text_A], dur_list=[dur_A], pitch_list=[pitch_A], energy_list=[en_A], file_location=os.path.join(root, "Sehnsucht_s01_v04_AA.wav"))
    # BB
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BB], pitch_list=[pitch_BB], energy_list=[en_BB], file_location=os.path.join(root, "Sehnsucht_s01_v04_BB.wav"))
    # AB
    tts.read_to_file(text_list=[text_A], dur_list=[dur_AB], pitch_list=[pitch_AB], energy_list=[en_AB], file_location=os.path.join(root, "Sehnsucht_s01_v04_AB.wav"))
    # BA
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BA], pitch_list=[pitch_BA], energy_list=[en_BA], file_location=os.path.join(root, "Sehnsucht_s01_v04_BA.wav"))
    
def sehnsucht_s03_v04(root):
    uc = UtteranceCloner(model_id="Meta", device="cuda" if torch.cuda.is_available() else "cpu")
    tf = ArticulatoryCombinedTextFrontend(language='de')
    
    audio_A = "Sehnsucht_s03_v04_original.wav"
    audio_B = "Sehnsucht_s01_v04_original.wav"

    text_A = "O~wie labend muß sie sein!"
    text_B = "Ach wie fühlt ich mich beglückt!"
    

    dur_A, pitch_A, en_A, _, _ = uc.extract_prosody(text_A, os.path.join(root, audio_A), lang="de", on_line_fine_tune=True)
    dur_B, pitch_B, en_B, _, _ = uc.extract_prosody(text_B, os.path.join(root, audio_B), lang='de', on_line_fine_tune=True)

    phones_A = tf.get_phone_string(text_A)
    phones_B = tf.get_phone_string(text_B)

    for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_A, pitch_A, en_A)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")
    print()
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_B, dur_B, pitch_B, en_B)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")
    print()

    text_AB = "Ach wie labend muß sie sein!"

    phones_AB = tf.get_phone_string(text_AB)

    # BB
    dur_BB = torch.cat([dur_B[:4], dur_A[3:]], 0)
    pitch_BB = torch.cat([pitch_B[:4], pitch_A[3:]], 0)
    en_BB = torch.cat([en_B[:4], en_A[3:]], 0)

    assert(len(phones_AB) == len(dur_BB) == len(pitch_BB) == len(en_BB))

    print('BB')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BB, pitch_BB, en_BB)):
       print(f"{i} {phone} {dur}   {pitch} {en}")

    # AB - O wie labend muss sie sein!; Prosody vom Ach
    dur_o = torch.add(dur_B[1], dur_B[2].item()) # dur vom 'o' = dur 'a' + dur 'x'
    print(dur_o)
    dur_AB = torch.cat([dur_B[:1], dur_o.reshape(1), dur_B[3].reshape(1), dur_A[3:]], 0) # dur vom 'o' = dur 'a' + dur 'x'
    pitch_AB = torch.cat([pitch_B[:2], pitch_B[3:4], pitch_A[3:]], 0) 
    en_AB = torch.cat([en_B[:2], en_B[3:4], en_A[3:]], 0)
    
    
    assert(len(phones_A) == len(dur_AB) == len(pitch_AB) == len(en_AB))
    print('AB')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_A, dur_AB, pitch_AB, en_AB)):
        print(f"{i}  {phone}  {dur} {pitch} {en}")

    # BA - Ach wie labend muss sie sein!; Prosodie vom O
    dur_o = torch.add(dur_A[1], dur_A[2].item()) # o + hilfspause
    dur_BA = torch.cat([dur_A[:1], dur_o.reshape(1), dur_B[2:4], dur_A[3:]], 0) 
    #dur_BA[1] = dur_o
    pitch_BA = torch.cat([pitch_A[:2], pitch_B[2:4], pitch_A[3:]], 0)
    en_BA = torch.cat([en_A[:2], en_B[2:4], en_A[3:]], 0)


    assert(len(phones_AB) == len(dur_BA) == len(pitch_BA) == len(en_BA))
    print('BA')
    for i, (phone, dur, pitch, en) in enumerate(zip(phones_AB, dur_BA, pitch_BA, en_BA)):
       print(f"{i}  {phone}  {dur} {pitch} {en}")

    tts = uc.tts
    tts.set_language('de')
    tts.set_utterance_embedding(os.path.join(root, audio_A))
    # AA
    tts.read_to_file(text_list=[text_A], dur_list=[dur_A], pitch_list=[pitch_A], energy_list=[en_A], file_location=os.path.join(root, "Sehnsucht_s03_v04_AA.wav"))
    # BB
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BB], pitch_list=[pitch_BB], energy_list=[en_BB], file_location=os.path.join(root, "Sehnsucht_s03_v04_BB.wav"))
    # AB
    tts.read_to_file(text_list=[text_A], dur_list=[dur_AB], pitch_list=[pitch_AB], energy_list=[en_AB], file_location=os.path.join(root, "Sehnsucht_s03_v04_AB.wav"))
    # BA
    tts.read_to_file(text_list=[text_AB], dur_list=[dur_BA], pitch_list=[pitch_BA], energy_list=[en_BA], file_location=os.path.join(root, "Sehnsucht_s03_v04_BA.wav"))

if __name__ == '__main__':
    root = "/Users/kockja/Documents/textklang/Creta_2022/Stavenhagen_Stündlein"
    # root = "/Users/kockja/Documents/textklang/Creta_2022/Sehnsucht"
    # wavs = ["Ein_Stündlein_s03_v01_original.wav", "Ein_Stündlein_s03_v02_original.wav", "Ein_Stündlein_s03_v03_original.wav", "Ein_Stündlein_s03_v04_original.wav", "Ein_Stündlein_s03_v05_original.wav"]
    # texts = ["O weh! nicht weiter sag!", "O still! nichts hören mag!", "Flieg ab, flieg ab von meinem Baum!", "Ach, Lieb und Treu ist wie ein Traum", "Ein Stündlein wohl vor Tag."]
    # wavs = ["Sehnsucht_s01_v01_original.wav", "Sehnsucht_s01_v02_original.wav", "Sehnsucht_s01_v03_original.wav", "Sehnsucht_s01_v04_original.wav",
    # "Sehnsucht_s01_v05_original.wav", "Sehnsucht_s01_v06_original.wav", "Sehnsucht_s01_v07_original.wav", "Sehnsucht_s01_v08_original.wav"]
    # texts = ["Ach, aus dieses Tales Gründen", "Die der kalte Nebel drückt,", "Könnt ich doch den Ausgang finden,", "Ach wie fühlt ich mich beglückt!",
    # "Dort erblick ich schöne Hügel,", "Ewig jung und ewig grün!", "Hätt ich Schwingen, hätt ich Flügel,", "Nach den Hügeln zög ich hin."]

    # wavs = ["Sehnsucht_s03_v01_original.wav", "Sehnsucht_s03_v02_original.wav", "Sehnsucht_s03_v03_original.wav", "Sehnsucht_s03_v04_original.wav",
    # "Sehnsucht_s03_v05_original.wav", "Sehnsucht_s03_v06_original.wav", "Sehnsucht_s03_v07_original.wav", "Sehnsucht_s03_v08_original.wav"]
    # texts = ["Ach wie schön muß sichs ergehen", "Dort im ewgen Sonnenschein,", "Und die Luft auf jenen Höhen,", "O wie labend muß sie sein!",
    # "Doch mir wehrt des Stromes Toben,", "Der ergrimmt dazwischen braust,", "Seine Wellen sind gehoben,", "Daß die Seele mir ergraust."]

    #clone_all(root, wavs, texts)
    #stündlein_s03_v01(root)
    #stündlein_s03_v02(root)
    stündlein_s03_v04(root)
    #sehnsucht_s01_v04(root)
    #sehnsucht_s03_v04(root)
    sys.exit(0)