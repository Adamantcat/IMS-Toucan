import os

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0, arousal=None, rhythm=None):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    if type(arousal) == float:
        arousal = [arousal]
    if type(rhythm) == float:
        rhythm = [rhythm]
    print("arousal: ", arousal)
    print("rhythm: ", rhythm)
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor, arousal_list=arousal, rhythm_list=rhythm)
    del tts


def the_raven(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Once upon a midnight dreary, while I pondered, weak, and weary,',
                         'Over many a quaint, and curious volume, of forgotten lore,',
                         'While I nodded, nearly napping, suddenly, there came a tapping,',
                         'As of someone gently rapping, rapping at my chamber door.',
                         'Ah, distinctly, I remember, it was in the bleak December,',
                         'And each separate dying ember, wrought its ghost upon the floor.',
                         'Eagerly, I wished the morrow, vainly, I had sought to borrow',
                         'From my books surcease of sorrow, sorrow, for the lost Lenore,',
                         'And the silken, sad, uncertain, rustling of each purple curtain',
                         'Thrilled me, filled me, with fantastic terrors, never felt before.'],
               filename=f"audios/{version}_the_raven.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


def sound_of_silence_single_utt(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""In restless dreams I walked alone,
Narrow streets of cobblestone.
Beneath the halo of a streetlamp,
I turned my collar to the cold and damp,
When my eyes were stabbed, by the flash of a neon light,
That split the night.
And touched the sound, of silence."""],
               filename=f"audios/{version}_sound_of_silence_as_single_utterance.wav",
               device=exec_device,
               language="eng",
               speaker_reference=speaker_reference)


def die_glocke(version, model_id="Meta", exec_device="cpu", speaker_reference=None, arousal=None, rhythm=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Fest gemauert in der Erden ^
                         Steht die Form, aus Lehm gebrannt. ^
                         Heute muss die Glocke werden! ^
                         Frisch, Gesellen, seid zur Hand!"""],
               filename=f"audios/{version}_die_glocke.wav",
               device=exec_device,
               language="deu",
               speaker_reference=speaker_reference,
               arousal=arousal,
               rhythm=rhythm)

def vergissmeinnicht(version, model_id="Meta", exec_device="cpu", speaker_reference=None, arousal=None, rhythm=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Es blüht ein schönes Blümchen ^
                        Auf unsrer grünen Au. ^
                        Sein Aug' ist wie der Himmel ^
                        So heiter und so blau."""],
               filename=f"audios/{version}_vergissmeinnicht.wav",
               device=exec_device,
               language="deu",
               speaker_reference=speaker_reference,
               arousal=arousal,
               rhythm=rhythm)


def viet_poem(version, model_id="Meta", exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""Thân phận,
                            ở một nơi luôn phải nhắc mình,
                            im miệng,
                            thân phận,
                            là khi nói về quá khứ,
                            ngó trước nhìn sau,
                            là phải biết nhắm mắt bịt tai làm lơ,
                            thờ ơ,
                            với tất cả những điều gai chướng,
                            thân phận chúng tôi ở đó,
                            những quyển sách chuyền tay nhau như ăn cắp,
                            ngôn luận ư?
                            không có đất cho nghĩa tự do."""],
               filename=f"audios/{version}_viet_poem.wav",
               device=exec_device,
               language="vie",
               speaker_reference=speaker_reference,
               duration_scaling_factor=1.2)


if __name__ == '__main__':
    import sys
    # exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    exec_device = "cpu"
    print(f"running on {exec_device}")

   # merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]

    # sound_of_silence_single_utt(version="integration_test_en",
    #                             model_id="IntegrationTest",
    #                             exec_device=exec_device,
    #                             #speaker_reference=merged_speaker_references
    #                             )

    # die_glocke(version="style_embedding_NoStyle_no_style",
    #            model_id="Poetry_NoStyle",
    #            exec_device=exec_device,
    #            arousal=None,
    #            rhythm=None
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # vergissmeinnicht(version="style_embedding_NoStyle_no_style",
    #            model_id="Poetry_NoStyle",
    #            exec_device=exec_device,
    #             arousal=None,
    #             rhythm=None
    #            #speaker_reference=merged_speaker_references
    #            )

    # die_glocke(version="style_embedding_NoStyle_with_uttembed",
    #            model_id="Poetry_NoStyle",
    #            exec_device=exec_device,
    #            arousal=None,
    #            rhythm=None,
    #            speaker_reference="/mount/arbeitsdaten/textklang/synthesis/Multispeaker_PoeticTTS_Data/Sprechweisen/tts-ayd-0450-m01-s02-t02-v01/tts-ayd-0450-m01-s02-t02-v01_2.wav"
    #            )
  
    # arousal only, high
    # die_glocke(version="style_embedding_beat100_a0.8_rNone",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #            arousal=0.8,
    #            rhythm=None
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # vergissmeinnicht(version="style_embedding_beat100_a0.8_rNone",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #             arousal=0.8,
    #             rhythm=None
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # arousal only, low
    # die_glocke(version="style_embedding_beat100_a0.1_rNone",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #            arousal=0.1,
    #            rhythm=None
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # vergissmeinnicht(version="style_embedding_beat100_a0.1_rNone",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #             arousal=0.1,
    #             rhythm=None
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # rhythm only, high
    # die_glocke(version="style_embedding_beat100_aNone_r4.0",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #            arousal=None,
    #            rhythm=4.0
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # vergissmeinnicht(version="style_embedding_beat100_aNone_r4.0",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #             arousal=None,
    #             rhythm=4.0
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # rhythm only, low
    # die_glocke(version="style_embedding_beat100_aNone_r0.2",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #            arousal=None,
    #            rhythm=0.2
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # vergissmeinnicht(version="style_embedding_beat100_aNone_r0.2",
    #            model_id="Poetry_StyleEmbedding_beat100",
    #            exec_device=exec_device,
    #             arousal=None,
    #             rhythm=0.2
    #            #speaker_reference=merged_speaker_references
    #            )
    
    # sys.exit(0)
    
    # arousal high, npvi high (not rhythmic)
    die_glocke(version="style_embedding_nPVI_a0.8_r20.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
               arousal=0.8,
               rhythm=20.0
               #speaker_reference=merged_speaker_references
               )
    
    vergissmeinnicht(version="style_embedding_nPVI_a0.8_r20.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
                arousal=0.8,
                rhythm=20.0
               #speaker_reference=merged_speaker_references
               )
    
    # arousal low, npvi low (very rhythmic)
    die_glocke(version="style_embedding_nPVI_a0.2_r10.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
               arousal=0.2,
               rhythm=10.0
               #speaker_reference=merged_speaker_references
               )
    
    vergissmeinnicht(version="style_embedding_nPVI_a0.2_r10.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
                arousal=0.2,
                rhythm=10.0
               #speaker_reference=merged_speaker_references
               )
    
    # arousal high, npvi low
    die_glocke(version="style_embedding_nPVI_a0.8_r10.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
               arousal=0.8,
               rhythm=10.0
               #speaker_reference=merged_speaker_references
               )
    
    vergissmeinnicht(version="style_embedding_nPVI_a0.8_r10.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
                arousal=0.8,
                rhythm=10.0
               #speaker_reference=merged_speaker_references
               )
    
    # arousal low, nPVI high
    die_glocke(version="style_embedding_nPVI_a0.2_r20.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
               arousal=0.2,
               rhythm=20.0
               #speaker_reference=merged_speaker_references
               )
    
    vergissmeinnicht(version="style_embedding_nPVI_a0.2_r20.0",
               model_id="StyleEmbedding_nPVI",
               exec_device=exec_device,
                arousal=0.2,
                rhythm=20.0
               #speaker_reference=merged_speaker_references
               )





    # viet_poem(version="new_voc",
    #           model_id="Meta",
    #           exec_device=exec_device,
    #           #speaker_reference=merged_speaker_references
    #           )
