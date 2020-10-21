import pickle
import torchvision
import torch
import numpy as np


def build_data_sets():
    data = torchvision.datasets.MNIST(".", train=True, transform=None, target_transform=None, download=True)
    data, target = torch.load("MNIST/processed/training.pt")
    data_test, target_test = torch.load("MNIST/processed/test.pt")
    train_images = []
    train_label = []
    val_images = []
    val_label = []
    test_images = []
    test_label = []
    track_train = []
    for i in range(data.shape[0]):
        if target[i] not in track_train:
            train_images += [data[i, :, :].numpy()]
            train_label += [target[i]]
            track_train += [target[i]]
        if len(track_train) == 10:
            break
    track_val = np.zeros(10)
    for i in range(data_test.shape[0]):
        val_images += [data_test[i, :, :].numpy()]
        val_label += [target_test[i]]
        track_val[target_test[i]] += 1
        if np.sum(track_val) == 100:
            break
    track_test = np.zeros(10)
    for i in range(data_test.shape[0]):
        if track_test[target_test[i]] < 100:
            test_images += [data_test[i, :, :].numpy()]
            test_label += [target_test[i]]
            track_test[target_test[i]] += 1
        if np.sum(track_test) == 1000:
            break
    np.savez("small_train_mnist.npz", (train_images, train_label))
    np.savez("small_val_mnist.npz", (val_images, val_label))
    np.savez("small_test_mnist.npz", (test_images, test_label))
