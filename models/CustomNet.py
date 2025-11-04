import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
      super(CustomNet, self).__init__()
      self.model = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=5, padding=2),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(64, 192, kernel_size=3, padding=1),
          nn.BatchNorm2d(192),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(192, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
      )
      self.fc = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),
          nn.Linear(512, 200)
      )
    def forward(self, x):
      x = self.model(x)
      x = self.fc(x)
      return x