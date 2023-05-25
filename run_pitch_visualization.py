from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Utility.EvaluationScripts.audio_vs_audio import get_pitch_curves_abc

tf = ArticulatoryCombinedTextFrontend(language='en')
# path_1 = "/Users/kockja/Documents/textklang/PlottingPoetry/Test/W1_005_Großmutter_Schlangenkoechin/2.wav"
# path_2 = "/Users/kockja/Documents/textklang/PlottingPoetry/audios/test/Karlsson/prose/W1_005_Großmutter_Schlangenkoechin/2.wav"
# path_3 = "/Users/kockja/Documents/textklang/PlottingPoetry/audios/test/Karlsson/poetry/W1_005_Großmutter_Schlangenkoechin/2.wav"

path_1 = "/Users/kockja/Documents/textklang/PlottingPoetry/Test/W3_K_026_Kinder-Konzert/11.wav"
path_2 = "/Users/kockja/Documents/textklang/PlottingPoetry/audios/test/Karlsson/prose/W3_K_026_Kinder-Konzert/11.wav"
path_3 = "/Users/kockja/Documents/textklang/PlottingPoetry/audios/test/Karlsson/poetry/W3_K_026_Kinder-Konzert/11.wav"


#text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)

get_pitch_curves_abc(path_1, path_2, path_3)

