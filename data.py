import os
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor


class FlikrData(Dataset):
    def __init__(self, df, image_path):
        self.df = df
        self.image_path = image_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        item = self.df.iloc[index]
        img = Image.open(os.path.join(self.image_path, item.image))
        caption = item.caption

        proc_img = image_processor(img, return_tensors="pt")

        return {'image': proc_img, 'caption': caption}


class FlikrDataModule(pl.LightningDataModule):
    def __init__(self, df, image_path, seed, batch_size):
        self.df = df
        self.image_path = image_path
        self.seed = seed
        self.batch_size = batch_size

    def setup(self):
        train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=self.seed, shuffle=False)
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.05, random_state=self.seed, shuffle=False)
        self.train_dataset = FlikrData(self.train_df, self.image_path)
        self.dev_dataset = FlikrData(self.val_df, self.image_path)
        self.test_dataset = FlikrData(self.test_df, self.image_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
