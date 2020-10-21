from collections import OrderedDict
import torch
from tqdm import tqdm

from Settings import Settings
from dataset import Dataset
from networks import EmbeddingNet, ClassifierNet


def load_models(settings):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_class = ClassifierNet(settings).to(device)
    model_emb = EmbeddingNet(settings).to(device)
    model_emb, model_class = load_parameters_model(settings, model_emb, model_class)
    model_emb.eval()
    model_class.eval()
    return model_emb, model_class


def load_parameters_model(settings, model_emb, model_class):
    if torch.cuda.is_available():
        model_emb.load_state_dict(
            torch.load(settings.checkpointroot_load_emb)["state_dict"])
        model_emb.load_state_dict(
            torch.load(settings.checkpointroot_load_class)["state_dict"])
    else:
        model_emb, model_class = load_from_cpu(settings, model_emb, model_class)
    return model_emb, model_class


def get_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        parts = k.split(".")
        name = ""
        for p in parts:
            if p != "module":
                name += p + "."
        name = name[0:-1]
        new_state_dict[name] = v
    return new_state_dict


def load_from_cpu(settings, model_emb, model_class):
    state_dict_emb = torch.load(settings.checkpointroot_load_emb, map_location=torch.device('cpu'))["state_dict"]
    state_dict_class = torch.load(settings.checkpointroot_load_class, map_location=torch.device('cpu'))["state_dict"]
    params_emb = get_state_dict(state_dict_emb)
    params_class = get_state_dict(state_dict_class)
    model_emb.load_state_dict(params_emb)
    model_class.load_state_dict(params_class)
    return model_emb, model_class


if __name__ == '__main__':
    settings = Settings().parse_arguments()
    settings.dataroot = "small_test_mnist.npz"
    settings.stage = "Test"
    dataloader = Dataset(settings).create_dataset(do_transform=False, type_dataloader="Single")
    model_emb, model_class = load_models(settings)
    batchsize = settings.batch_size
    n_batches = len(dataloader.dataset.data[0]) // batchsize
    accuracy = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
            embedding = model_emb.predict(data["images"])
            model_class.predict(embedding)
            model_class.set_ground_truth(data["labels"])
            accuracy += model_class.compute_accuracy() / (n_batches * batchsize)
    print("final accuracy: {}".format(accuracy))

