import random
import tqdm

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, remove_faulty_samples=False):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Wunderhorn")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_toni_wunderhorn(),
                                            ctc_selection=False,
                                            corpus_dir=os.path.join("Corpora", "Wunderhorn"),
                                            lang="de"))

    train_set = ConcatDataset(datasets)

    model = FastSpeech2(lang_embs=100)

    if remove_faulty_samples:
        find_and_remove_faulty_samples(net=FastSpeech2(lang_embs=100),
                                       datasets=datasets,
                                       device=torch.device("cuda"),
                                       path_to_checkpoint=resume_checkpoint)
    
    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=5000,
               batch_size=32,
               lang="de",
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=500,
               path_to_checkpoint="Models/FastSpeech2_German/best.pt",
               fine_tune=True,
               resume=resume)

@torch.inference_mode()
def find_and_remove_faulty_samples(net,
                                   datasets,
                                   device,
                                   path_to_checkpoint):
    net = net.to(device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
    net.load_state_dict(check_dict["model"])
    for dataset_index in range(len(datasets)):
        nan_ids = list()
        for datapoint_index in tqdm(range(len(datasets[dataset_index]))):
            loss = net(text_tensors=datasets[dataset_index][datapoint_index][0].unsqueeze(0).to(device),
                       text_lengths=datasets[dataset_index][datapoint_index][1].to(device),
                       gold_speech=datasets[dataset_index][datapoint_index][2].unsqueeze(0).to(device),
                       speech_lengths=datasets[dataset_index][datapoint_index][3].to(device),
                       gold_durations=datasets[dataset_index][datapoint_index][4].unsqueeze(0).to(device),
                       gold_pitch=datasets[dataset_index][datapoint_index][6].unsqueeze(0).to(device),  # mind the switched order
                       gold_energy=datasets[dataset_index][datapoint_index][5].unsqueeze(0).to(device),  # mind the switched order
                       utterance_embedding=datasets[dataset_index][datapoint_index][7].unsqueeze(0).to(device),
                       lang_ids=datasets[dataset_index][datapoint_index][8].unsqueeze(0).to(device),
                       return_mels=False).squeeze()
            if torch.isnan(loss):
                print(f"NAN DETECTED: {dataset_index}, {datapoint_index}")
                nan_ids.append(datapoint_index)
        datasets[dataset_index].remove_samples(nan_ids)
