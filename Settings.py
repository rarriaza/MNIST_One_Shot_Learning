import argparse


class Settings:
    def __init__(self):
        self.initialized = False

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='One-Shot Learning running script')
        parser.add_argument('--dataroot', type=str, default="small_train_mnist.npz",
                            help='path to dataset')
        parser.add_argument('--lr_emb', type=float, default=0.01,
                            help='learning rate')
        parser.add_argument('--lr_class', type=float, default=0.0003,
                            help='learning rate')
        parser.add_argument('--lr_fine', type=float, default=0.0003,
                            help='learning rate')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='learning rate')
        parser.add_argument('--number_triplets', type=int, default=1000,
                            help='learning rate')
        parser.add_argument('--margin_loss', type=int, default=0.2,
                            help='margin for triplet loss')
        parser.add_argument('--epochs', type=int, default=2,
                            help='number of epochs')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='first beta for Adam optimizer')
        parser.add_argument('--stage', type=str, default="Train")
        parser.add_argument('--lr_decay_epoch', type=int, default=150,
                            help='epoch lr starts decaying')
        parser.add_argument('--checkpointroot_save', type=str, default="./checkpoints/",
                            help='epoch lr starts decaying')
        parser.add_argument('--checkpointroot_load_emb', type=str, default="./checkpoints/OneShot_Model_Embedding.pth")
        parser.add_argument('--checkpointroot_load_class', type=str, default="./checkpoints/OneShot_Model_Classifier.pth")
        parser.add_argument('--random', type=bool, default=True)
        parser.add_argument('--augmentation', type=bool, default=True)
        settings = parser.parse_args()
        self.initialized = True
        return settings
