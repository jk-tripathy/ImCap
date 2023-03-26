import torch
import torch.nn as nn
from transformers import ResNetModel
import torch.nn.functional as F


class Encoder_CNN(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.lin = nn.Linear(147852, context_dim)

    def forward(self, x):
        x = x.squeeze(dim=1)
        output = self.conv(x)
        output = F.relu(output)
        output = torch.flatten(output, start_dim=1)
        output = self.lin(output)
        output = F.relu(output)
        return output


class Encoder_ResNet(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.lin = nn.Linear(100352, context_dim)

    def forward(self, x):
        x = x.squeeze(dim=1)
        with torch.no_grad():
            outputs = self.model(pixel_values=x)

        output = torch.flatten(outputs.last_hidden_state, start_dim=1)
        output = self.lin(output)
        output = F.relu(output)
        return output
