import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class Plotting:
    def __init__(self):
        # Writer will output to ./runs/ directory by default
        self.writer = SummaryWriter()

    def add_loss(self, loss, epoch, tag):
        self.writer.add_scalar(f'Loss/{tag}', loss, epoch)
    
    def add_accuracies(self, train_accuracy, source_test_accuracy, target_test_accuracy, epoch):
        self.writer.add_scalar(f'Accuracy/train', train_accuracy, epoch)
        self.writer.add_scalar(f'Accuracy/SVHN_test', source_test_accuracy, epoch)
        self.writer.add_scalar(f'Accuracy/MNIST_test', target_test_accuracy, epoch)

    def add_feature_maps(self, feature_map, tag):
        self.add_embedding(feature_map, tag=tag)

    def close(self):
        self.writer.close()
