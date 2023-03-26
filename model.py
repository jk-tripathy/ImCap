import lightning.pytorch as pl
from torch import optim
from encoder import Encoder_CNN, Encoder_ResNet


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.enc_cnn = Encoder_CNN()
        self.enc_resnet = Encoder_ResNet()

    def forward(self, image, caption):
        output = self.enc_cnn(image)
        # output = self.enc_resnet(image)
        return output

    def training_step(self, batch, idx):
        _ = self(**batch)

    def validation_step(self, batch, idx):
        _ = self(**batch)

    def test_step(self, batch, idx):
        _ = self(**batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
