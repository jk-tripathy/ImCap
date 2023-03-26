import torch
import lightning.pytorch as pl
from torch import optim, nn
from encoder import Encoder_CNN, Encoder_ResNet
from decoder import Decoder_Vanilla


class Model(pl.LightningModule):
    def __init__(self, context_dim, embedding_dim, hidden_dim, vocab_size, num_layers=1, baseline=True):
        super().__init__()
        self.baseline = baseline
        if self.baseline:
            self.enc_cnn = Encoder_CNN(context_dim)
            self.dec_vanilla = Decoder_Vanilla(context_dim, embedding_dim, hidden_dim, vocab_size, num_layers)
        else:
            self.enc_resnet = Encoder_ResNet(context_dim)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, image, caption, target):
        if self.baseline:
            img_ctx = self.enc_cnn(image)
            output = self.dec_vanilla(caption, img_ctx)
        else:
            img_ctx = self.enc_resnet(image)

        output = output[:, -1, :]
        target = torch.flatten(target)
        loss = self.criterion(output, target)
        return output, loss

    def training_step(self, batch, idx):
        output, loss = self(**batch)
        self.log("train loss ", loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, idx):
        output, loss = self(**batch)
        self.log("val loss ", loss, prog_bar=True, logger=True)

    def test_step(self, batch, idx):
        output, loss = self(**batch)
        self.log("test loss ", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
