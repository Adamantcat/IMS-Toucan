# https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from tqdm import tqdm
# from poem_classifier.Classifier_Dataset import Classifier_Dataset

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class VADClassifier:

    def __init__(self, device='cpu'):
        self.device = device
        self.model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = EmotionModel.from_pretrained(self.model_name).to(self.device)
        self.resample = Resample(orig_freq=24000, new_freq=16000).to(self.device)

    def predict_emotions(self, wavs):
        y_list = []
        print('Predict emotions')
        for wav in tqdm(wavs):
            y = self.resample(torch.tensor(wav, dtype=torch.float).to(self.device)) # .cpu().numpy()
            y = self.processor(y, sampling_rate=16000)
            y = y['input_values'][0]
            y = torch.from_numpy(y).unsqueeze(0).to(self.device)

            # run through model
            with torch.no_grad():
                y = self.model(y)[1]

            # convert to numpy
            y = y.detach().squeeze().cpu().numpy()
            y_list.append(y)

        return np.array(y_list)




if __name__ == '__main__':
    vad_classifier = VADClassifier()

    amused, _ = sf.read("/mount/resources/speech/corpora/ThorstenDatasets/thorsten-emotional_v02/amused/3da24b61836ab54f31166159df72cfae.wav")
    angry, _ = sf.read("/mount/resources/speech/corpora/ThorstenDatasets/thorsten-emotional_v02/angry/2cc90eb146f6d574cb75553bb7956a42.wav")
    neutral, _ = sf.read("/mount/resources/speech/corpora/ThorstenDatasets/thorsten-emotional_v02/neutral/0cf91be5ce94ff566520aa47a5b573c8.wav")
    poetry, _ = sf.read("/mount/arbeitsdaten/textklang/synthesis/Multispeaker_PoeticTTS_Data/Sprechweisen/tts-dd-2703-m01-s01-t28-v01/tts-dd-2703-m01-s01-t28-v01_4.wav")

    wavs = [amused, angry, neutral, poetry]

    vad = vad_classifier.predict_emotions(wavs)
    print(vad) # arousal, dominance, valence



