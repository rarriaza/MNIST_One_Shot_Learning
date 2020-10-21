import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
from skimage.morphology import dilation, erosion


class Dataset:
    def __init__(self, settings):
        self.settings = settings
        self.images_path = settings.dataroot
        self.batch_size = settings.batch_size

    def create_dataset(self, do_transform=True, type_dataloader="Triplet"):
        if type_dataloader == "Triplet":
            dataloader = TripletDataLoader(self.settings, do_transform=do_transform)
        else:
            dataloader = SingleDataLoader(self.settings)
        dataset = dataloader.load_data()
        return dataset


class SingleDataset(data.Dataset):
    def __init__(self, settings):
        self.settings = settings
        self.dataroot = settings.dataroot
        self.load_traindata()

    def load_traindata(self):
        self.data = np.load(self.dataroot, allow_pickle=True)["arr_0"]

    def __getitem__(self, index):
        image, label = self.data[0][index], self.data[1][index]
        image_tensor = transforms.ToTensor()(image)
        return {"images": image_tensor, "labels": label}

    def __len__(self):
        return len(self.data[0])


class TripletDataset(data.Dataset):
    def __init__(self, settings, do_transform=True):
        self.settings = settings
        self.dataroot = settings.dataroot
        self.do_transform = do_transform
        self.set_random_augmentation()
        self.load_traindata()

    def load_traindata(self):
        self.data = np.load(self.dataroot, allow_pickle=True)["arr_0"]

    def __getitem__(self, index):
        a, n = random.sample(range(10), 2)
        A_img, A_label = self.data[0][a], self.data[1][a]
        P_img, P_label = self.data[0][a], self.data[1][a]
        N_img, N_label = self.data[0][n], self.data[1][n]
        if self.do_transform:
            A = self.transforms(A_img)
            P = self.transforms(P_img)
            N = self.transforms(N_img)
        else:
            A = transforms.ToTensor()(A_img)
            P = transforms.ToTensor()(P_img)
            N = transforms.ToTensor()(N_img)
        return {"A": A, "P": P, "N": N,
                "A_label": A_label, "P_label": P_label, "N_label": N_label}

    def __len__(self):
        if self.do_transform:
            return self.settings.number_triplets
        return len(self.data[0])

    def set_random_augmentation(self):
        self.transforms = transforms.Compose([
            RandomMorphology(),
            Image.fromarray,
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=(-10, 10, -10, 10)),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])


class RandomMorphology:
    def __call__(self, image):
        transform = random.choice(["dilation", "erosion", "None"])
        if transform == "dilation":
            return dilation(image)
        elif transform == "erosion":
            return erosion(image)
        return image


class TripletDataLoader():
    def __init__(self, settings, do_transform=True):
        self.settings = settings
        self.images_path = settings.dataroot
        self.batch_size = settings.batch_size
        self.do_transform = do_transform
        self.dataset = TripletDataset(settings, do_transform=do_transform)
        self.number_triplets = settings.number_triplets
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0, drop_last=True)

    def load_data(self):
        return self

    def __iter__(self):
        if self.do_transform:
            for i, data in enumerate(self.dataloader):
                if i * self.batch_size >= self.number_triplets:
                    break
                yield data
        else:
            for i, data in enumerate(self.dataloader):
                yield data


class SingleDataLoader():
    def __init__(self, settings):
        self.settings = settings
        self.images_path = settings.dataroot
        self.batch_size = settings.batch_size
        self.dataset = SingleDataset(settings)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0, drop_last=True)

    def load_data(self):
        return self

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
