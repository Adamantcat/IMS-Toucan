import time

import torch
import wandb
from torch.utils.data import ConcatDataset

from TTSTrainingInterfaces.ToucanTTS.ToucanTTS import ToucanTTS
from TTSTrainingInterfaces.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.path_to_transcript_dicts import *
from Utility.storage_config import MODELS_DIR
from Utility.storage_config import PREPROCESSING_DIR


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_MLS_English")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()

    if gpu_count > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0

    chunk_count = 50
    chunks = split_dictionary_into_chunks(build_path_to_transcript_dict_mls_english(), split_n=chunk_count)
    for index in range(chunk_count):
        datasets.append(prepare_tts_corpus(transcript_dict=chunks[index],
                                           corpus_dir=os.path.join(PREPROCESSING_DIR, f"mls_english_chunk_{index}"),
                                           lang="en",
                                           gpu_count=gpu_count,
                                           rank=rank))

    train_set = ConcatDataset(datasets)

    model = ToucanTTS(use_conditional_layernorm_embedding_integration=True)  # if we set this to true, a different embedding integration method will be used to give us a better embedding function

    if gpu_count > 1:
        model.to(rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        torch.distributed.barrier()
    train_sampler = torch.utils.data.RandomSampler(train_set)

    if use_wandb:
        if rank == 0:
            wandb.init(
                name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
                id=wandb_resume_id,  # this is None if not specified in the command line arguments.
                resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               datasets=[train_set],
               device=device,
               save_directory=save_dir,
               eval_lang="en",
               path_to_checkpoint=resume_checkpoint,
               path_to_embed_model=None,  # if we set this to None, we train the embedding function jointly from scratch
               fine_tune=finetune,
               steps=2000000,
               resume=resume,
               use_wandb=use_wandb,
               train_embed=True,
               gpu_count=gpu_count,
               train_samplers=[train_sampler],
               steps_per_checkpoint=2000)  # we want to train the embedding function
    if use_wandb:
        wandb.finish()
