from copy import deepcopy
from tqdm import tqdm, trange
import numpy as np
from Settings import Settings
from dataset import Dataset
from train import train, create_dataloaders, init_models
from test import load_models
from build_train_data import build_data_sets
from networks import ClassifierNet, EmbeddingNet

import torch
from collections import OrderedDict


if __name__ == '__main__':
    build_data_sets()

    args = Settings().parse_arguments()
    embedding_model, classifier_model = init_models(args)
    dataloader, dataloader_val = create_dataloaders(args)
    number_triplets, batchsize = args.number_triplets, args.batch_size

    stages = ["freeze_class", "freeze_embedding", "fine_tuning"]
    for stage in stages:
        train(embedding_model, classifier_model, dataloader, dataloader_val, args, stage)


    args.dataroot = "small_test_mnist.npz"
    args.stage = "Test"
    dataloader = Dataset(args).create_dataset(do_transform=False, type_dataloader="Single")
    model_emb, model_class = load_models(args)
    n_batches = len(dataloader.dataset.data[0]) // batchsize
    accuracy = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
            embedding = model_emb.predict(data["images"])
            model_class.predict(embedding)
            model_class.set_ground_truth(data["labels"])
            accuracy += model_class.compute_accuracy() / (n_batches * batchsize)
    print("final accuracy: {}".format(accuracy))
