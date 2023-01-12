import librosa
import os
import torch
import copy
from run_utterance_cloner import UtteranceCloner
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
import librosa.display as lbd
import matplotlib.pyplot as plt

class Enjambment:
    def __init__(self, model_id, device):
        self.lang = "de"
        self.tf = ArticulatoryCombinedTextFrontend(language=self.lang)
        self.uc = UtteranceCloner(model_id=model_id, device=device)
        self.baseline_durs = {"a": 0.1165449, "n": 0.1382676, "t": 0.08158591, "ə": 0.09228161, "e": 0.1396795, "ɪ": 0.07832647,
            "ŋ": 0.1664985, "s": 0.1611922, "m": 0.1055175, "o": 0.2254623, "ʊ": 0.09851111, "ɑ": 0.22241028, "ɔ": 0.10085145, "i": 0.1203422, "ɜ": 0.1497454 
            #,"ə": 0.12186987 #finales Schwa für [Heidelberg - bebte], [An die Hoffnung - liebliche], [Lebensalter - Grenze], [Hälfte des Lebens - Winde]
            }
        self.len_1_durs = {"a": 0.1318755, "n": 0.2001694, "t": 0.1113052, "ə": 0.101913, "e": 0.1766673, "ɪ": 0.1040012,
            "ŋ": 0.2001694, "s": 0.2146674, "m": 0.1414272, "o": 0.3319946, "ʊ": 0.1070239, "ɑ": 0.2909994, "ɔ": 0.135002, "i": 0.1700067, "ɜ": 0.2706714
            #,"ə": 0.1773717 #nur für [Heidelberg - bebte] und [An die Hoffnung - liebliche], [Lebensalter - Grenze], [Hälfte des Lebens - Winde]
            } 
        self.len_2_durs = {"a": 0.1425118, "n": 0.265015, "t": 0.130001, "ə": 0.122501, "e": 0.225, "ɪ": 0.1325035,
            "ŋ": 0.265015, "s": 0.29, "m": 0.1900025, "o": 0.3674925, "ʊ": 0.110001, "ɑ": 0.3324973, "ɔ": 0.1500185, "i": 0.360001, "ɜ": 0.340002
            #,"ə": 0.250004 #nur für [Heidelberg - bebte] und [An die Hoffnung - liebliche], [Lebensalter - Grenze], [Hälfte des Lebens - Winde]
            } 
        self.len_max_durs = {"a": 0.200012, "n": 0.529999, "t": 0.35, "ə": 0.33, "e": 0.27, "ɪ": 0.170002,
            "ŋ": 0.529999, "s": None, "m": None, "o": 0.2254623, "ʊ": 0.09851111, "ɑ": 0.22241028, "ɔ": None, "ɜ": None
            #,"ə": 0.33 #nur für [Heidelberg - bebte] und [An die Hoffnung - liebliche], [Lebensalter - Grenze], [Hälfte des Lebens - Winde]
            } 

    def find_enjambment(self, vers1, vers2, reference_audio, save_dir, file_naming, view=True):
        vers1_phones = self.tf.get_phone_string(vers1)
        vers2_phones = self.tf.get_phone_string(vers2)

        text_combined = " ".join([vers1, vers2])
        phones_combined = self.tf.get_phone_string(text_combined)

        enj_idx = len(vers1_phones) - 3 # -1 for len=/=idx, -1 for EOS token, -1 for final pause

        print("vers1 ", vers1_phones, " len1 ", len(vers1_phones))
        print("vers2 ", vers2_phones, " len2 ", len(vers2_phones))
        print("total ", phones_combined, " len_total ", len(phones_combined))
        print("enjambment found at: ", enj_idx, " ", phones_combined[enj_idx])

        final_rhyme_idx = self.find_enj_rhyme(text_combined, enj_idx)
        print("final rhyme starts at position: ", final_rhyme_idx, " ")
        print(phones_combined[final_rhyme_idx:enj_idx+1])

        # extract prosody from reference
        # dur, pitch, en, _, _ = self.uc.extract_prosody(phones_combined, reference_audio, lang=self.lang, on_line_fine_tune=True, input_is_phones=True)
        dur, pitch, en, _, _ = self.uc.extract_prosody(phones_combined, reference_audio, lang=self.lang, on_line_fine_tune=True, input_is_phones=True)
        tts = self.uc.tts
        tts.set_language(self.lang)
        #tts.set_utterance_embedding(reference_audio)
        tts.set_utterance_embedding("/Users/kockja/Documents/textklang/ICPhS/original/Brod_und_Wein_s02_niemand.wav")
        
        for i, (p, d)  in enumerate(zip(phones_combined, dur)):
            print(i, "\t", p, "\t", d)

        new_dur = copy.deepcopy(dur)

        # replace verbunden with verwoben in Winter2
        if text_combined == "Der Ruhe Geist ist aber in den Stunden Der prächtigen Natur mit Tiefigkeit verbunden.":
            text_combined = "Der Ruhe Geist ist aber in den Stunden Der prächtigen Natur mit Tiefigkeit verwoben."
            phones_combined = self.tf.get_phone_string("Der Ruhe Geist ist aber in den Stunden Der prächtigen Natur mit Tiefigkeit verwoben.")
            n = 77 # index of n in verbuNden
            new_dur = torch.cat([new_dur[0:n], new_dur[n+1:]]) # remove duration of the n in verbunden to match verwoben
            pitch = torch.cat([pitch[0:n], pitch[n+1:]])
            en = torch.cat([en[0:n], en[n+1:]])
            print("verwoben")
            print(len(phones_combined), " ", new_dur.size(), " ", pitch.size(), " ", en.size())
            for i, (p, d)  in enumerate(zip(phones_combined, new_dur)):
                print(i, "\t", p, "\t", d)
        # fix pronunciation of Genien and segnend in Vulkan
        elif text_combined == "Und immer wohnt der freundlichen Genjen Noch Einer gerne segnend mit ihm, und wenn":
            phones_combined = ''.join([phones_combined[:31], "e", phones_combined[32], "i", phones_combined[34:]]) # Genien
            phones_combined = ''.join([phones_combined[:53], "e", phones_combined[54:]]) # segnend
            print(phones_combined)
        # fix pronunciation of Busen in An die Hoffnung
        elif text_combined == "Bin ich schon hier; und schon gesanglos Schlummert das schaudernde Herz im Busn":
            phones_combined = ''.join([phones_combined[:65], "u", "z", phones_combined[67:]])
            print(phones_combined)
        # fix pronunciation of Thälern in Neckar
        elif text_combined == "In deinen Thälern wachte mein Herz mir auf Zum Leben, deine Wellen umspielten mich,":
            phones_combined = ''.join([phones_combined[:14], "ɐ", phones_combined[15:]])
            print(phones_combined)
        # fix pronunciation of Obstbaum in Himmel
        elif text_combined == "Herunter, wo der Obstbaum blühend darüber steht Und Duft an wilden Hecken weilet,":
            phones_combined = ''.join([phones_combined[:17], "o", phones_combined[18:]])
            print(phones_combined)
        
        # lengthening in 3 or 4 steps from data
        for l in range(3): # 4 steps for lenghtening
            print("lengthening + ", l)
            out_file = f"{file_naming}_{l}.wav"
            if l == 0: 
                len_dict = self.baseline_durs
                title = "No lengthening"
            elif l == 1:
                len_dict = self.len_1_durs
                title = "1st step"
            elif l == 2:
                len_dict = self.len_2_durs
                title = "2nd step"
            # elif l == 3:
            #     len_dict = self.len_max_durs
            for i in range(final_rhyme_idx, enj_idx + 1):
                phone = phones_combined[i]
                dur_in_sec = len_dict[phone]
                dur_in_frames = librosa.time_to_frames(dur_in_sec, sr=16000, hop_length=256)
                new_dur[i] = dur_in_frames
                print(f"{phone}\t{dur_in_sec} in frames: {dur_in_frames}")
            save_file_path = os.path.join(save_dir, out_file)
            print(len(phones_combined), new_dur.size(), pitch.size(), en.size())

            if view:
                from Preprocessing.TextFrontend import get_language_id
                from Utility.utils import cumsum_durations
                phones_tensor = self.tf.string_to_tensor(phones_combined, input_phonemes=True).to(torch.device(tts.device))
                mel, _, _, _ = tts.phone2mel(phones_tensor,
                                            return_duration_pitch_energy=True,
                                            utterance_embedding=tts.default_utterance_embedding,
                                            durations=new_dur,
                                            pitch=pitch,
                                            energy=en,
                                            lang_id=get_language_id(self.lang),
                                            duration_scaling_factor=1.0,
                                            pitch_variance_scale=1.0,
                                            energy_variance_scale=1.0)
                mel = mel.transpose(0, 1)
                phones_for_plot = phones_combined.replace(" ", "|")
                # fig, ax = plt.subplots(nrows=2, ncols=1)
                fig, ax = plt.subplots(figsize=(6.4, 2.4))
                #ax[0].plot(wave.cpu().numpy())
                lbd.specshow(mel.cpu().numpy(),
                            ax=ax,
                            sr=16000,
                            cmap='GnBu',
                            y_axis='mel',
                            x_axis=None,
                            hop_length=256)
                #ax[0].yaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                duration_splits, label_positions = cumsum_durations(new_dur.cpu().numpy())
                ax.set_xticks(duration_splits, minor=True)
                ax.xaxis.grid(True, which='minor', color='darkgreen')
                ax.xaxis.remove_overlapping_locs = False
                ax.set_xticks(label_positions, minor=False)
                ax.set_xticklabels(phones_for_plot)
                ax.set_title(title)
                plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
                plt.show()

            tts.read_to_file(text_list=[phones_combined], dur_list=[new_dur], pitch_list=[pitch], energy_list=[en], file_location=save_file_path, input_is_phones=True)

        # only for cloning (Neckar - In deinen Tälern wachte mein Herz mir auf...)
        # save_file_path = os.path.join(save_dir, f"{file_naming}_cloned.wav")
        # print(len(phones_combined), new_dur.size(), pitch.size(), en.size())
        # tts.read_to_file(text_list=[phones_combined], dur_list=[new_dur], pitch_list=[pitch], energy_list=[en], file_location=save_file_path, input_is_phones=True)

        

    def find_enj_rhyme(self, text, enj_idx):
        start_idx = enj_idx
        phones_as_tensors = self.tf.string_to_tensor(text)

        # go backwards from enj_idx until we find a vowel, i.e. the vowel of the last syllable rhyme
        while phones_as_tensors[start_idx][12] == 0: #idx 12 corresponds to 'vowel' feature
            start_idx = start_idx -1
        
        return start_idx

if __name__ == '__main__':
    enj = Enjambment(model_id="Meta", device="cpu")
    vers1 = "und niemand"
    vers2 = "Weiß von wannen"
    ref_file = "Brod_und_Wein_niemand_demo"
    path_to_reference = f"/Users/kockja/Documents/textklang/ICPhS/original/{ref_file}.wav"
    #lengthening = 4 # lengthening in 4 steps from data
    enj.find_enjambment(vers1=vers1, vers2=vers2, reference_audio=path_to_reference, save_dir=f"/Users/kockja/Documents/textklang/ICPhS/synthese_v3.2/", file_naming=ref_file)

