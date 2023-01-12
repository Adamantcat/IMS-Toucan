from certifi import where
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Utility.EvaluationScripts.audio_vs_audio import get_pitch_curves_abc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

tf = ArticulatoryCombinedTextFrontend(language='de')
path_1 = "/Users/kockja/Documents/textklang/ICPhS/synthese_v3.2/Haelfte_des_Lebens_s02_stehn_0.wav"
path_2 = "/Users/kockja/Documents/textklang/ICPhS/synthese_v3.2/Haelfte_des_Lebens_s02_stehn_1.wav"
path_3 = "/Users/kockja/Documents/textklang/ICPhS/synthese_v3.2/Haelfte_des_Lebens_s02_stehn_2.wav"

transcript = "Die Mauern stehn sprachlos und kalt, im Winde "

text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)

# get_pitch_curves(path_1, path_2, plot_curves=True)
get_pitch_curves_abc(path_1, path_2, path_3)
# get_pitch_curve_diff_extractors(path, text)

sys.exit(0)
results = dict()
i = 0
with open("/Users/kockja/Documents/PhD/Voiceprivacy/offsets_ICASSP/offsets_results_pitch_corr.txt", "r") as f:
    for line in f.readlines():
        line = line.replace("{", "").replace("}", "").replace("'", "")
        if line in ['\n', '\r\n']:
            continue
        items = line.split(",")
        curr = dict()
        for item in items:      
            key, val = item.split(":")
            curr[key.lstrip().rstrip()] = val.lstrip().rstrip()
        results[i] = curr
        i += 1

out_file = "/Users/kockja/Documents/PhD/Voiceprivacy/offsets_ICASSP/offset_results.csv"
df = pd.DataFrame(results).T
df.to_csv(out_file, sep=";")

df_f = df[df['dataset'] == 'libri_dev_trials_f']
df_m = df[df['dataset'] == 'libri_dev_trials_m']

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df['random_offset_lower'], df['EER'], df['pitch_corr_mean'])
threedee.set_xlabel('offset')
threedee.set_ylabel('EER')
threedee.set_zlabel('pitch')
plt.show()

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');