from copy import deepcopy
from tqdm import tqdm, trange
import numpy as np

from Settings import Settings
from dataset import Dataset
import torch
from networks import ClassifierNet, EmbeddingNet


def validation(embedding_model, classifier_model, dataloader, batchsize, stage):
    embedding_model.eval()
    classifier_model.eval()
    val_loss = 0.0
    n_batches = len(dataloader.dataset) // batchsize
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
            embedding_model.set_input(data)
            embedding_model.forward()
            classifier_model.set_input(embedding_model.output)
            classifier_model.forward()
            val_loss += classifier_model.compute_loss().detach().cpu().item() + \
                        embedding_model.compute_loss().detach().cpu().item()
    embedding_model.train()
    classifier_model.train()
    return val_loss / (n_batches * batchsize)


def set_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def freeze_models(stage, embedding_model, classifier_model):
    if stage == "freeze_embedding":
        set_grad(embedding_model, False)
        set_grad(classifier_model, True)
    elif stage == "freeze_class":
        set_grad(embedding_model, True)
        set_grad(classifier_model, False)
    else:
        set_grad(embedding_model, True)
        set_grad(classifier_model, True)


def train(model_emb, model_class, dataloader, dataloader_val, args, stage):
    n_samples = args.number_triplets
    batchsize = args.batch_size
    epochs = args.epochs
    n_batches = n_samples // batchsize
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    freeze_models(stage, model_emb, model_class)
    if stage == "fine_tuning":
        for param_group in model_emb.optimizer.param_groups:
            param_group['lr'] = args.lr_fine
        for param_group in model_class.optimizer.param_groups:
            param_group['lr'] = args.lr_fine
    for epoch in trange(epochs, desc='Epoch: '):
        for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
            model_emb.set_input(data)
            model_emb.forward()
            if stage == "freeze_class":
                model_emb.backward_net()
            else:
                model_class.set_input(model_emb.output)
                model_class.forward()
                model_class.backward_net()

        train_loss[epoch] = (model_class.cum_loss + model_emb.cum_loss) / (batchsize * n_batches)
        model_class.cum_loss = 0.0
        model_emb.cum_loss = 0.0

        val_loss[epoch] = validation(model_emb, model_class, dataloader_val, batchsize, stage)

        model_class.lr_decay(epoch)
        model_emb.lr_decay(epoch)

    save_model(model_class, args, "Classifier")
    save_model(model_emb, args, "Embedding")
    return train_loss, val_loss


def create_dataloaders(args):
    dataloader = Dataset(args).create_dataset()
    args_val = deepcopy(args)
    args_val.dataroot = "small_val_mnist.npz"
    args_val.stage = "Test"
    dataloader_test = Dataset(args_val).create_dataset(do_transform=False)
    return dataloader, dataloader_test


def init_models(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    EmbeddingModel = EmbeddingNet(args).to(device)
    ClassifierModel= ClassifierNet(args).to(device)
    return EmbeddingModel, ClassifierModel


def save_model(model, args, comment):
    state = {
            'comment': comment,
            'state_dict': model.state_dict(),
            'state_dict_optm': model.optimizer.state_dict(),
    }
    torch.save(state, args.checkpointroot_save + "OneShot_Model_" + comment + ".pth")


if __name__ == '__main__':
    args = Settings().parse_arguments()
    embedding_model, classifier_model = init_models(args)
    dataloader, dataloader_val = create_dataloaders(args)
    number_triplets, batchsize = args.number_triplets, args.batch_size

    stages = ["freeze_class", "freeze_embedding", "fine_tuning"]
    for stage in stages:
        train(embedding_model, classifier_model, dataloader, dataloader_val, args, stage)
