import torch
import torch.nn as nn
from transformers import ResNetModel


class Encoder_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = x['pixel_values'].squeeze(dim=1)
        output = self.conv(x)
        return output


class Encoder_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-152")

    def forward(self, x):
        x['pixel_values'] = x['pixel_values'].squeeze(dim=1)
        with torch.no_grad():
            outputs = self.model(**x)
        return outputs.last_hidden_state
