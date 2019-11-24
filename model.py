import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # N, 3, 32, 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), # N, 8, 32, 32
            nn.BatchNorm2d(8),
            Swish(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), # N, 8, 32, 32
            nn.BatchNorm2d(8),
            Swish(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # N, 8, 16, 16
        )

        self.layer1.apply(init_weights)

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # N, 16, 16, 16
            nn.BatchNorm2d(16),
            Swish(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # N, 16, 8, 8           
        )

        self.layer2.apply(init_weights)

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # N, 32, 8, 8
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # N, 32, 8, 8
            nn.BatchNorm2d(32),
            Swish(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # N, 32, 4, 4
        )
        
        self.layer2.apply(init_weights)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # N, 64, 4, 4
        self.pool4_avg = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)  # N, 64, 1, 1
        self.pool4_max = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)  # N, 64, 1, 1
        nn.init.xavier_uniform_(self.conv4.weight)

        self.clf = nn.Linear(128, 10)
        nn.init.xavier_uniform_(self.clf.weight)

        
    def forward(self, input_):
        feature_maps = []
        
        x = self.layer1(input_)        
        
        x = self.layer2(x)              
        feature_maps.append(x)       
        
        x = self.layer3(x)
        feature_maps.append(x)       
        
        x = self.conv4(x)              
        avg = self.pool4_avg(x)         
        maxx = self.pool4_max(x)        
        x = torch.cat((avg, maxx), 1)  
        feature_maps.append(x)       
        
        x = x.reshape(x.size(0), -1)
        out = self.clf(x)
        
        return out, feature_maps
