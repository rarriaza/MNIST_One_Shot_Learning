import torch
import torch.nn as nn
from torch.nn import init
import itertools
from losses import TripletLoss
from torch.optim import lr_scheduler


class EmbeddingNet(nn.Module):
    def __init__(self, settings, input_shape=(1, 28, 28), embedding_size=10, margin_loss=0.2):
        super(EmbeddingNet, self).__init__()
        in_channels = input_shape[0]
        sequence = [nn.Conv2d(in_channels, 128, kernel_size=7, padding=1),  # (28, 28)
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),
                 nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (11, 11)
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),
                 nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (5, 5)
                 nn.ReLU(inplace=True),
                 Flatten(),  # (4, 4)
                 nn.Linear(6 * 6 * 256, 4096),
                 nn.ReLU(inplace=True),
                 nn.Linear(4096, embedding_size),
                 nn.LayerNorm(embedding_size, elementwise_affine=False)
                 ]
        self.model = nn.Sequential(*sequence)
        self.model.apply(weight_init)
        self.loss_fun = TripletLoss(margin_loss).triplet_loss
        self.cum_loss = 0.0
        self.scheduler, self.optimizer = self.init_optimizer(settings)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def init_optimizer(self, settings):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=settings.lr_emb, betas=(settings.beta1, 0.999))
        if settings.stage == "Train":
            scheduler = self.set_scheduler(optimizer, settings)
            return scheduler, optimizer
        return None, optimizer

    def set_scheduler(self, optimizer, settings):
        def lambda_rule(epoch):
            init_epoch_decay = settings.lr_decay_epoch
            n_epochs = settings.epochs
            lr_l = 1.0 if epoch < init_epoch_decay else 1 - (epoch - init_epoch_decay) / (n_epochs - init_epoch_decay)
            return lr_l
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    def set_input(self, data):
        self.anchor = data["A"].to(self.device)
        self.positive = data["P"].to(self.device)
        self.negative = data["N"].to(self.device)
        self.anchor_label = data["A_label"].to(self.device)
        self.positive_label = data["P_label"].to(self.device)
        self.negative_label = data["N_label"].to(self.device)

    def forward(self):
        self.output = {"A": self.model(self.anchor),
                       "P": self.model(self.positive),
                       "N": self.model(self.negative),
                       "A_label": self.anchor_label,
                       "P_label": self.positive_label,
                       "N_label": self.negative_label}

    def predict(self, anchor):
        return self.model(anchor)

    def compute_loss(self):
        loss = self.loss_fun(self.output)
        return loss

    def backward_net(self):
        loss = self.compute_loss()
        loss.backward()
        self.cum_loss += loss.detach().cpu().item()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def lr_decay(self, epoch):
        self.scheduler.step(epoch)


class ClassifierNet(nn.Module):
    def __init__(self, settings, n_classes=10, embedding_size=10):
        super(ClassifierNet, self).__init__()
        sequence = [
            nn.Linear(embedding_size, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, n_classes),
            nn.Softmax(dim=1)
        ]
        self.model = nn.Sequential(*sequence)
        self.model.apply(weight_init)
        self.scheduler, self.optimizer = self.init_optimizer(settings)
        self.loss_fun = nn.CrossEntropyLoss()
        self.cum_loss = 0.0
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_input(self, data):
        self.anchor = data["A"].to(self.device)
        self.positive = data["P"].to(self.device)
        self.negative = data["N"].to(self.device)
        self.anchor_label = data["A_label"].to(self.device)
        self.positive_label = data["P_label"].to(self.device)
        self.negative_label = data["N_label"].to(self.device)

    def forward(self):
        self.output = {"A": self.model(self.anchor),
                       "P": self.model(self.positive),
                       "N": self.model(self.negative)}

    def set_ground_truth(self, labels):
        self.labels = labels

    def predict(self, embedding):
        self.output = self.model(embedding)

    def init_optimizer(self, settings):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=settings.lr_class, betas=(settings.beta1, 0.999))
        if settings.stage == "Train":
            scheduler = self.set_scheduler(optimizer, settings)
            return scheduler, optimizer
        return None, optimizer

    def set_scheduler(self, optimizer, settings):
        def lambda_rule(epoch):
            init_epoch_decay = settings.lr_decay_epoch
            n_epochs = settings.epochs
            lr_l = 1.0 if epoch < init_epoch_decay else 1 - (epoch - init_epoch_decay) / (n_epochs - init_epoch_decay)
            return lr_l
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    def compute_loss(self):
        loss = self.loss_fun(self.output["A"], self.anchor_label)
        loss += self.loss_fun(self.output["P"], self.positive_label)
        loss += self.loss_fun(self.output["N"], self.negative_label)
        return loss

    def backward_net(self):
        loss = self.compute_loss()
        loss.backward()
        self.cum_loss += loss.detach().cpu().item()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def lr_decay(self, epoch):
        self.scheduler.step(epoch)

    def compute_accuracy(self):
        prediction_anchor = torch.argmax(self.output, dim=1)
        accuracy = (prediction_anchor.eq(self.labels)).sum()
        return accuracy.detach().cpu().item()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LambdaLayer(nn.Module):
    def __init__(self, l):
        super(LambdaLayer, self).__init__()
        self.fun = l

    def forward(self, x):
        return self.fun(x)


def weight_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.kaiming_normal_(m.weight)
        init.zeros_(m.bias)





