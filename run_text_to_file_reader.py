import os
import sys

import torch

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(model_id, sentence, filename, device="cpu", language="en", speaker_reference=None, faster_vocoder=False):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=faster_vocoder)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def the_raven(version, model_id="Meta", exec_device="cpu", speed_over_quality=True, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=['Once upon a midnight dreary, while I pondered, weak, and weary,',
                         'Over many a quaint, and curious volume of forgotten lore,',
                         'While I nodded, nearly napping, suddenly, there came a tapping,',
                         'As of someone gently rapping, rapping at my chamber door.',
                         'Tis some visitor, I muttered, tapping at my chamber door,',
                         'Only this, and nothing more.',
                         'Ah, distinctly, I remember, it was in the bleak December,',
                         'And each separate dying ember, wrought its ghost upon the floor.',
                         'Eagerly, I wished the morrow, vainly, I had sought to borrow',
                         'From my books surcease of sorrow, sorrow, for the lost Lenore,',
                         'For the rare and radiant maiden, whom the angels name Lenore,',
                         'Nameless here, for evermore.',
                         'And the silken, sad, uncertain, rustling of each purple curtain',
                         'Thrilled me, filled me, with fantastic terrors, never felt before.'],
               filename=f"audios/the_raven_{version}.wav",
               device=exec_device,
               language="en",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
    
def vergissmeinnicht(version, model_id="Poetry", exec_device="cpu", speed_over_quality=True, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["Es blüht ein schönes Blümchen ~ Auf unsrer grünen Au. Sein Aug' ist wie der Himmel ~ So heiter und so blau.",
                         "Es weiß nicht viel zu reden ~ Und alles, was es spricht, Ist immer nur dasselbe, Ist nur: Vergissmeinnicht.",
                         "Wenn ich zwei Äuglein sehe ~ So heiter und so blau, So denk' ich an mein Blümchen ~ Auf unsrer grünen Au.",
                         "Da kann ich auch nicht reden ~ Und nur mein Herze spricht, So bange nur, so leise, Und nur: Vergissmeinnicht."],
               filename=f"audios/TAI/vergissmeinnicht_{version}.wav",
               device=exec_device,
               language="de",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
    
def an_die_hoffnung(version, model_id="Poetry", exec_device="cpu", speed_over_quality=False, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)
    read_texts(model_id=model_id,
               sentence=["An die Hoffnung",
                         "O Hoffnung! holde! gütiggeschäftige! ~ Die du das Haus der Trauernden nicht verschmähst, Und gerne dienend, Edle! zwischen ~ Sterblichen waltest und Himmelsmächten,",
                         "Wo bist du? wenig lebt ich; doch atmet kalt ~ Mein Abend schon. Und stille, den Schatten gleich, Bin ich schon hier; und schon gesanglos ~ Schlummert das schaudernde Herz im Busen.",
                         "Im grünen Tale, dort, wo der frische Quell ~ Vom Berge täglich rauscht, und die liebliche ~ Zeitlose mir am Herbsttag aufblüht, Dort, in der Stille, du Holde, will ich",
                         "Dich suchen, oder wenn in der Mitternacht ~ Das unsichtbare Leben im Haine wallt, Und über mir die immerfrohen ~ Blumen, die blühenden Sterne, glänzen,","O du des Aethers Tochter! erscheine dann ~ Aus deines Vaters Gärten, und darfst du nicht, Ein Geist der Erde, kommen, schröck, o ~ Schröcke mit anderem nur das Herz mir."],
               filename=f"audios/TAI/an_die_hoffnung_{version}.wav",
               device=exec_device,
               language="de",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
    
def der_sommer(version, model_id="Poetry", exec_device="cpu", speed_over_quality=False, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)
    read_texts(model_id=model_id,
               sentence=["Der Sommer",
                        "Die Tage gehn vorbei mit sanfter Lüfte Rauschen, Wenn mit der Wolke sie der Felder Pracht vertauschen, Des Tales Ende trifft der Berge Dämmerungen ~  Dort, wo des Stromes Wellen sich hinabgeschlungen.",
                        "Der Wälder Schatten sieht umhergebreitet, Wo auch der Bach entfernt hinuntergleitet, Und sichtbar ist der Ferne Bild in Stunden, Wenn sich der Mensch zu diesem Sinn gefunden."
                         ],
                         filename=f"audios/TAI/der_sommer_{version}.wav",
               device=exec_device,
               language="de",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
    
def haelfte_des_lebens(version, model_id="Poetry", exec_device="cpu", speed_over_quality=False, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)
    read_texts(model_id=model_id,
               sentence=["Hälfte des Lebens",
                        "Mit gelben Birnen hänget ~ Und voll mit wilden Rosen ~ Das Land in den See, Ihr holden Schwäne, Und trunken von Küssen ~ Tunkt ihr das Haupt ~ Ins heilignüchterne Wasser.",
                        "Weh mir, wo nehm ich, wenn ~ Es Winter ist, die Blumen, und wo ~ Den Sonnenschein, Und Schatten der Erde? ~ Die Mauern stehn ~  Sprachlos und kalt, im Winde ~ Klirren die Fahnen."
                         ],
                         filename=f"audios/TAI/hälfte_des_lebens_{version}.wav",
               device=exec_device,
               language="de",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)

def die_erwartung(version, model_id="Poetry", exec_device="cpu", speed_over_quality=False, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)
    read_texts(model_id=model_id,
               sentence=["Die Erwartung",
                         "Hör ich das Pförtchen nicht gehen? Hat nicht der Riegel geklirrt? Nein, es war des Windes Wehen, Der durch diese Pappeln schwirrt.",
                         "O schmücke dich, du grün belaubtes Dach, Du sollst die Anmutstrahlende empfangen, Ihr Zweige, baut ein schattendes Gemach, Mit holder Nacht sie heimlich zu umfangen, Und all ihr Schmeichellüfte, werdet wach ~ Und scherzt und spielt um ihre Rosenwangen, Wenn seine schöne Bürde, leicht bewegt, Der zarte Fuß zum Sitz der Liebe trägt.",
                         "Stille, was schlüpft durch die Hecken ~ Raschelnd mit eilendem Lauf? Nein, es scheuchte nur der Schrecken ~ Aus dem Busch den Vogel auf.",
                         "O! lösche deine Fackel, Tag! Hervor, Du geistge Nacht, mit deinem holden Schweigen, Breit um uns her den purpurroten Flor, Umspinn uns mit geheimnisvollen Zweigen, Der Liebe Wonne flieht des Lauschers Ohr, Sie flieht des Strahles unbescheidnen Zeugen! Nur Hesper, der verschwiegene, allein ~ Darf still herblickend ihr Vertrauter sein",
                         "Rief es von ferne nicht leise, Flüsternden Stimmen gleich? ~ Nein, der Schwan ists, der die Kreise ~ Ziehet durch den Silberteich.",
                         "Mein Ohr umtönt ein Harmonienfluß, Der Springquell fällt mit angenehmem Rauschen, Die Blume neigt sich bei des Westes Kuß, Und alle Wesen seh ich Wonne tauschen, Die Traube winkt, die Pfirsche zum Genuß, Die üppig schwellend hinter Blättern lauschen, Die Luft, getaucht in der Gewürze Flut, Trinkt von der heißen Wange mir die Glut.",
                         "Hör ich nicht Tritte erschallen? ~ Rauschts nicht den Laubgang daher? ~ Nein, die Frucht ist dort gefallen, Von der eignen Fülle schwer.",
                         "Des Tages Flammenauge selber bricht ~ In süßem Tod und seine Farben blassen, Kühn öffnen sich im holden Dämmerlicht ~ Die Kelche schon, die seine Gluten hassen, Still hebt der Mond sein strahlend Angesicht, Die Welt zerschmilzt in ruhig große Massen, Der Gürtel ist von jedem Reiz gelöst, Und alles Schöne zeigt sich mir entblößt.",
                         "Seh ich nichts Weißes dort schimmern? ~ Glänzts nicht wie seidnes Gewand? ~ Nein, es ist der Säule Flimmern ~ An der dunkeln Taxuswand.",
                         "O! sehnend Herz, ergötze dich nicht mehr, Mit süßen Bildern wesenlos zu spielen, Der Arm, der sie umfassen will, ist leer, Kein Schattenglück kann diesen Busen kühlen; O! führe mir die Lebende daher, Laß ihre Hand, die zärtliche, mich fühlen, Den Schatten nur von ihres Mantels Saum, Und in das Leben tritt der holde Traum.",
                         "Und leis, wie aus himmlischen Höhen ~ Die Stunde des Glückes erscheint, So war sie genaht, ungesehen, Und weckte mit Küssen den Freund."
                        ],
                         filename=f"audios/TAI/die_erwartung_{version}.wav",
               device=exec_device,
               language="de",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)
    
def die_teilung_der_erde(version, model_id="Poetry", exec_device="cpu", speed_over_quality=False, speaker_reference=None):
    os.makedirs("audios", exist_ok=True)
    read_texts(model_id=model_id,
               sentence=["Die Teilung der Erde",
                         "»Nehmt hin die Welt!« rief Zeus von seinen Höhen ~ Den Menschen zu. »Nehmt, sie soll euer sein! ~ Euch schenk ich sie zum Erb und ewgen Lehen, Doch teilt euch brüderlich darein.«",
                         "Da eilt, was Hände hat, sich einzurichten, Es regte sich geschäftig jung und alt. Der Ackermann griff nach des Feldes Früchten, Der Junker birschte durch den Wald.",
                         "Der Kaufmann nimmt, was seine Speicher fassen, Der Abt wählt sich den edeln Firnewein, Der König sperrt die Brücken und die Straßen ~ Und sprach: »Der Zehente ist mein.«",
                         "Ganz spät, nachdem die Teilung längst geschehen, Naht der Poet, er kam aus weiter Fern; Ach! da war überall nichts mehr zu sehen, Und alles hatte seinen Herrn!",
                         "»Weh mir! so soll ich denn allein von allen ~ Vergessen sein, ich, dein getreuster Sohn?« So ließ er laut der Klage Ruf erschallen ~ Und warf sich hin vor Jovis Thron.",
                         "»Wenn du im Land der Träume dich verweilet«, Versetzt der Gott, »so hadre nicht mit mir. Wo warst du denn, als man die Welt geteilet?«- »Ich war«, sprach der Poet, »bei dir.",
                         "Mein Auge hing an deinem Angesichte, An deines Himmels Harmonie mein Ohr- Verzeih dem Geiste, der, von deinem Lichte ~ Berauscht, das Irdische verlor!«",
                         "»Was tun?« spricht Zeus. »Die Welt ist weggegeben, Der Herbst, die Jagd, der Markt ist nicht mehr mein. Willst du in meinem Himmel mit mir leben: So oft du kommst, er soll dir offen sein.«"
                        ],
                         filename=f"audios/TAI/die_teilung_der_erde_{version}.wav",
               device=exec_device,
               language="de",
               speaker_reference=speaker_reference,
               faster_vocoder=speed_over_quality)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    schiller_reference = "/mount/arbeitsdaten/textklang/synthesis/Multispeaker_PoeticTTS_Data/TAI/Schiller/tts-dd-3681-m03-s01-t03-v01/tts-dd-3681-m03-s01-t03-v01_2.wav"

    # Schiller
    vergissmeinnicht(version="Poetry_TAI_Schiller",
                     model_id="Poetry_TAI_Schiller",
                     speaker_reference=schiller_reference,
                     exec_device=exec_device,
                     speed_over_quality=False)

    der_sommer(version="Poetry_TAI_Schiller",
                    model_id="Poetry_TAI_Schiller",
                    speaker_reference=schiller_reference,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    an_die_hoffnung(version="Poetry_TAI_Schiller",
                    model_id="Poetry_TAI_Schiller",
                    speaker_reference=schiller_reference,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    haelfte_des_lebens(version="Poetry_TAI_Schiller",
                    model_id="Poetry_TAI_Schiller",
                    speaker_reference=schiller_reference,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    die_erwartung(version="Poetry_TAI_Schiller",
                    model_id="Poetry_TAI_Schiller",
                    speaker_reference=schiller_reference,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    die_teilung_der_erde(version="Poetry_TAI_Schiller",
                    model_id="Poetry_TAI_Schiller",
                    speaker_reference=schiller_reference,
                    exec_device=exec_device,
                    speed_over_quality=False)
   
    # Hölderlin
    vergissmeinnicht(version="Poetry_TAI_Hölderlin",
                     model_id="Poetry_TAI_Hölderlin",
                     speaker_reference=None,
                     exec_device=exec_device,
                     speed_over_quality=False)

    der_sommer(version="Poetry_TAI_Hölderlin",
                    model_id="Poetry_TAI_Hölderlin",
                    speaker_reference=None,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    an_die_hoffnung(version="Poetry_TAI_Hölderlin",
                    model_id="Poetry_TAI_Hölderlin",
                    speaker_reference=None,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    haelfte_des_lebens(version="Poetry_TAI_Hölderlin",
                    model_id="Poetry_TAI_Hölderlin",
                    speaker_reference=None,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    die_erwartung(version="Poetry_TAI_Hölderlin",
                    model_id="Poetry_TAI_Hölderlin",
                    speaker_reference=None,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    die_teilung_der_erde(version="Poetry_TAI_Hölderlin",
                    model_id="Poetry_TAI_Hölderlin",
                    speaker_reference=None,
                    exec_device=exec_device,
                    speed_over_quality=False)
    
    #the_raven(version="MetaBaseline",
    #          model_id="Meta",
    #          exec_device=exec_device,
    #          speed_over_quality=exec_device != "cuda")
