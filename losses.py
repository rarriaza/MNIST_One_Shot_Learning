import torch.nn as nn
import torch
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def triplet_loss(self, inputs):
        A, P, N = inputs["A"], inputs["P"], inputs["N"]
        p_dist = torch.sum(torch.pow(A - P, 2), axis=-1)
        n_dist = torch.sum(torch.pow(A - N, 2), axis=-1)
        return torch.sum(torch.relu(p_dist - n_dist + self.margin), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        return loss

