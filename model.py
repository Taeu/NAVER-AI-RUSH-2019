import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torchvision.models as models

from octconv import OctConv2d, OctReLU, OctMaxPool2d


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class OctCNN(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        
        self.convs = nn.Sequential(OctConv2d('first', in_channels=1, out_channels=32, kernel_size=3),
                                    OctReLU(),
                                    OctConv2d('regular', in_channels=32, out_channels=64, kernel_size=3),
                                    OctReLU(),
                                    OctConv2d('regular', in_channels=64, out_channels=128, kernel_size=3),
                                    OctReLU(),
                                    OctMaxPool2d(2),
                                    OctConv2d('regular', in_channels=128, out_channels=128, kernel_size=3),
                                    OctReLU(),
                                    OctConv2d('last', in_channels=128, out_channels=128, kernel_size=3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(6272, 256),
                                nn.Dropout(0.5),
                                nn.Linear(256, out_size))
    
    
    def forward(self, x):  
        x = self.convs(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.fc(x)
        return x