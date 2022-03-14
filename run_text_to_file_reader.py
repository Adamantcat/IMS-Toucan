import os

import torch

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2


def read_texts(model_id, sentence, filename, device="cpu", language="en"):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language(language)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def read_texts_as_ensemble(model_id, sentence, filename, device="cpu", language="en", amount=10):
    """
    for this function, the filename should NOT contain the .wav ending, it's added automatically
    """
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language(language)
    if type(sentence) == str:
        sentence = [sentence]
    for index in range(amount):
        tts.default_utterance_embedding = torch.zeros(704).float().random_(-40, 40).to(device)
        tts.read_to_file(text_list=sentence, file_location=filename + f"_{index}" + ".wav")


def read_harvard_sentences(model_id, device):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


def read_contrastive_focus_sentences(model_id, device):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)

    with open("Utility/contrastive_focus_test_sentences.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/focus_{}".format(model_id)
    os.makedirs(output_dir, exist_ok=True)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("audios", exist_ok=True)

    prosa = ["Wenn auch der Hunger nicht zu Stillen war, so ist doch sein Drängen eine Zeit lang unterdrückt.",
        "Wenn man ihn höret, so sollte man glauben, dass er schon Hunger sterbe, und wenn man ihn sieht, scheint er allein von den allgemeinen Qualen verschont zu sein.",
        "Die Hitze ist immer bedeutend, und sogar unerträglich, wenn die Brise sie nicht mäßigt.",
        #"Doch steht diese Tatsache unzweifelhaft fest, und Gott wolle es verhüten, dass wir auch das noch auskosten sollen.",
        #"Sollte das Geschick endlich satt sein, uns zu prüfen?",
        "Alle jene Zufälle, welche uns jetzt so fern zu liegen scheinen.",
        "Ich saß, da ich so in mir sprach, wie ein nachdenkliches Mädchen in einer gedankenlosen Romanze am Bach, sah den fliehenden Wellen nach.",
        #"Ich erinnerte mich, und ich sah uns, wie gelinder Schlaf die Umarmten mitten in der Umarmung umfing.",
        "Mein ganzes Wesen verstummt und lauscht, wenn die zarte Welle der Luft mir um die Brust spielt.",
        "Daß der Mensch in seiner Jugend das Ziel so nahe glaubt! Es ist die schönste aller Täuschungen, womit die Natur der Schwachheit unsers Wesens aufhilft.",
        "Ich war aufgewachsen, wie eine Rebe ohne Stab, und die wilden Ranken breiteten richtungslos über dem Boden sich aus.",
        #"Ein sanfter warmer Hauch glitt über mein Gesicht, ich erwachte wie aus dem Todesschlaf, die Mutter hatte sich über mich hingebeugt."
        ]

    poetry = ["Alles prüfe der Mensch, sagen die Himmlischen, daß er, kräftig genährt, danken für Alles lern', und verstehe die Freiheit, aufzubrechen, wohin er will.",
        "Da ich ein Knabe war, rettet' ein Gott mich oft vom Geschrei und der Rute der Menschen, da spielt ich sicher und gut mit den Blumen des Hains, und die Lüftchen des Himmels spielten mit mir.",
        "Zwar damals ruft ich noch nicht euch mit Namen, auch ihr nanntet mich nie, wie die Menschen sich nennen, als kennten sie sich.",
        "Mich erzog der Wohllaut des säuselnden Hains und lieben lernt ich unter den Blumen.",
        "und wüßten sie noch in kommenden Jahren von uns beiden, wenn einst wieder der Genius gilt, sprächen sie: es schufen sich einst die Einsamen liebend nur von Göttern gekannt ihre geheimere Welt.",
        "Zu lang, zu lang schon treten die Sterblichen sich gern aufs Haupt, und zanken um Herrschaft sich, den Nachbarn fürchtend, und es hat auf eigenem Boden der Mann nicht Segen.",
        "Dann hör ich oft die Stimme des Donnerers am Mittag, wenn der eherne nahe kommt, wenn ihm das Haus bebt und der Boden unter ihm dröhnt und der Berg es nachhallt.",
        "Vor seiner Hütte ruhig im Schatten sitzt der Pflüger, dem Genügsamen raucht sein Herd. Gastfreundlich tönt dem Wanderer im Friedlichen Dorfe die Abendglocke."]

    tts_prosa = InferenceFastSpeech2(device='cpu', model_name='German', noise_reduce=True)
    tts_prosa.set_language('de')
    tts_prosa.set_utterance_embedding("/mount/arbeitsdaten/textklang/synthesis/Interspeech_2022/test/Heidelberg_Strophen/segment_6.wav")

    for i in range(len(prosa)):
        read_texts(model_id="Zischler", sentence=prosa[i], filename=f"audios/Poetry_or_Prose/Prosa/prosa{i}_poetic.wav", device="cpu", language="de")
        tts_prosa.read_to_file([prosa[i]], file_location=f"audios/Poetry_or_Prose/Prosa/prosa{i}_prosaic.wav")

    for i in range(len(poetry)):
        read_texts(model_id="Zischler", sentence=poetry[i], filename=f"audios/Poetry_or_Prose/Poetry/poetry{i}_poetic.wav", device="cpu", language="de")
        tts_prosa.read_to_file([poetry[i]], file_location=f"audios/Poetry_or_Prose/Poetry/poetry{i}_prosaic.wav")



