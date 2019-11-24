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
    
    def add_accuracies(self, train_accuracy, source_test_accuracy, target_test_accuracy, epoch, tag):
        self.writer.add_scalar(f'Accuracy/{tag}_train', train_accuracy, epoch)
        self.writer.add_scalar(f'Accuracy/{tag}_SVHN_test', source_test_accuracy, epoch)
        self.writer.add_scalar(f'Accuracy/{tag}_MNIST_test', target_test_accuracy, epoch)

    def add_feature_maps(self, feature_map, labels, tag):
        if isinstance(feature_map, list): 
            feature_map = torch.cat(feature_map, 0)
            labels = [label for temp in labels for label in temp ]

        feature_map = feature_map.reshape((-1, feature_map.size(1))) 
        # self.writer.add_embedding(feature_map, label_img=images, tag=tag, global_step=1)
        self.writer.add_embedding(feature_map, metadata=labels, tag=tag, global_step=1)

    def close(self):
        self.writer.close()
