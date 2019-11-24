from pathlib import Path

import scipy.io
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import Net
from loss import DiscrLoss
from plots import Plotting


device = torch.device("cuda")

mnist_transformations = transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.Grayscale(num_output_channels=3),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])

svhn_transformations = transforms.Compose([ 
                        #    transforms.Grayscale(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                        ])

def load_datasets(download=False):
    mnist_train = datasets.MNIST('./data/mnist', train=True, download=download,
                                transform=mnist_transformations)
   
    mnist_test = datasets.MNIST('./data/mnist', train=False, download=download,
                                transform=mnist_transformations)
    svhn_train = datasets.SVHN('./data/svhn', split='train', download=download,
                               transform=svhn_transformations)

    svhn_test = datasets.SVHN('./data/svhn', split='test', download=download,
                              transform=svhn_transformations)    
    
    return svhn_train, svhn_test, mnist_train, mnist_test

def test(test_loader, model):
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, feature_maps = model(data)               
            pred = torch.argmax(logits, dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100 * correct/len(test_loader.dataset)
    
    return accuracy


def visualise_feature_maps(model, test_loader, tag, plotting):
    feature_maps_all = []
    target_all = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, feature_maps = model(data)               
            feature_maps_all.append(feature_maps[-1])
            target_all.append(target.cpu().numpy())

    plotting.add_feature_maps(feature_maps_all, target_all, tag)


def pretrain(model, epochs, pretrain_step_lr_epochs, loaders, plotting):
    source_train_loader, source_test_loader, target_train_loader, target_test_loader = loaders
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, pretrain_step_lr_epochs, gamma=0.1)
    clf_criterion = nn.CrossEntropyLoss()

    visualise_feature_maps(model, source_test_loader, "svhn_before_training", plotting)
    visualise_feature_maps(model, target_test_loader, "mnist_before_training", plotting)

    for epoch in range(epochs):
        epoch_loss = 0
        train_iter = tqdm(source_train_loader, leave=False)
        for step, (data, target) in enumerate(train_iter):
            data, target = data.to(device), target.to(device)
            logits, feature_maps = model(data)
            optim.zero_grad()
            loss = clf_criterion(logits, target)   
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            train_iter.set_description(f'Loss: {epoch_loss/(step+1):.4f}')

        train_accuracy = test(source_train_loader, model)
        source_test_accuracy = test(source_test_loader, model)
        target_test_accuracy = test(target_test_loader, model)
        plotting.add_loss(epoch_loss, epoch + 1, "pretraining")
        plotting.add_accuracies(train_accuracy, source_test_accuracy, target_test_accuracy, epoch+1, "pretraining")

        print(f"Epoch {epoch+1}: Training loss: {epoch_loss/len(source_train_loader)}")
        print(f"Epoch {epoch+1}: Test accuracy: {target_test_accuracy}")
        
        scheduler.step()

    visualise_feature_maps(model, source_test_loader, "svhn_before_DA", plotting)
    visualise_feature_maps(model, target_test_loader, "mnist_before_DA", plotting)


def main():
    pretrain_epochs = 1
    epochs = 10
    batch_size = 128
    pretrain_step_lr_epochs = 3 
    discr_weight = torch.tensor(0.5).to(device)
    feature_maps_weights = [0.3, 0.5, 1]

    svhn_train, svhn_test, mnist_train, mnist_test = load_datasets()

    source_train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=1)
    source_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=True, num_workers=1)
    target_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    target_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=1)
    loaders = (source_train_loader, source_test_loader, target_train_loader, target_test_loader)

    plotting = Plotting()

    model = Net().to(device)

    pretrain(model, pretrain_epochs, pretrain_step_lr_epochs, loaders, plotting)

    clf_criterion = nn.CrossEntropyLoss()
    discr_criterion = DiscrLoss(weights=feature_maps_weights)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003) 
    scheduler = torch.optim.lr_scheduler.StepLR(optim, pretrain_step_lr_epochs, gamma=0.1)
    t_iter = iter(target_train_loader)
    
    train_accuracies = []
    source_test_accuracies = []
    target_test_accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_clf_loss = 0
        epoch_discr_loss = 0
        train_iter = tqdm(source_train_loader, leave=False)
        for step, (s_inputs, s_labels) in enumerate(train_iter):
            s_inputs = s_inputs.to(device)
            s_labels = s_labels.to(device)

            try: 
                t_inputs, _ = next(t_iter)
            except:
                t_iter = iter(target_train_loader)
                t_inputs, _ = next(t_iter) 

            t_inputs = t_inputs.to(device)
            
            s_logits, s_feature_maps = model(s_inputs)
            t_logits, t_feature_maps = model(t_inputs)
            optim.zero_grad()
            clf_loss = clf_criterion(s_logits, s_labels)
            discr_loss = discr_criterion(s_feature_maps, t_feature_maps)
            loss = clf_loss + discr_weight*discr_loss
            loss.backward()        
            optim.step()  
            epoch_clf_loss += clf_loss.item()
            epoch_discr_loss += discr_loss.item()
            epoch_loss += loss.item()

            train_iter.set_description(f'Classification loss: {epoch_clf_loss/(step+1):.4f} Discriminator loss: {epoch_discr_loss/(step+1):.4f} ')
            
        train_accuracy = test(source_train_loader, model)
        source_test_accuracy = test(source_test_loader, model)
        target_test_accuracy = test(target_test_loader, model)
        plotting.add_loss(epoch_loss, epoch + 1, "training")
        plotting.add_accuracies(train_accuracy, source_test_accuracy, target_test_accuracy, epoch+1, "training")

        
        print(f"Epoch {epoch+1}: Training classification loss: {epoch_clf_loss/len(source_train_loader)}")
        print(f"Epoch {epoch+1}: Training discriminator loss: {epoch_discr_loss/len(source_train_loader)}")
        print(f"Epoch {epoch+1}: Test accuracy: {target_test_accuracy}")

        scheduler.step()

    visualise_feature_maps(model, source_test_loader, "svhn_after_DA", plotting)
    visualise_feature_maps(model, target_test_loader, "mnist_after_DA", plotting)
 
    plotting.close()

if __name__=='__main__':
    main()
