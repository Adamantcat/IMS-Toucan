import torch
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.TrainingPipelines.FastSpeech2_MetaCheckpoint import find_and_remove_faulty_samples
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *

net = net=FastSpeech2(lang_embs=100)
datasets = list()
datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_toni_wunderhorn(),
                                            corpus_dir=os.path.join("Corpora", "Wunderhorn"),
                                            lang="de"))

find_and_remove_faulty_samples(net, datasets, device=torch.device("cuda"), path_to_checkpoint="Models/FastSpeech2_Meta/best.pt")