import os
import torch
import lightning.pytorch as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor


class FlikrData(Dataset):
    def __init__(self, df, image_path, word_to_idx, max_len):
        self.df = df
        self.image_path = image_path
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-152")
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        img = Image.open(os.path.join(self.image_path, item.image))
        proc_img = self.image_processor(img, return_tensors="pt")

        proc_caption = torch.LongTensor(item.caption)

        proc_target = torch.LongTensor([item.target])

        return {'image': proc_img, 'caption': proc_caption, 'target': proc_target}


class FlikrDataModule(pl.LightningDataModule):
    def __init__(self, df, image_path, seed, batch_size, max_len, vocab_size, word_to_idx, idx_to_word):
        super().__init__()
        self.df = df
        self.image_path = image_path
        self.seed = seed
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

    def setup(self, stage='fit', usePretrainedTokenizer=False):
        train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=self.seed, shuffle=False)
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.05, random_state=self.seed, shuffle=False)

        if stage == 'fit':
            self.train_dataset = FlikrData(self.train_df, self.image_path, self.word_to_idx, self.max_len)
            self.dev_dataset = FlikrData(self.val_df, self.image_path, self.word_to_idx, self.max_len)
            self.test_dataset = FlikrData(self.test_df, self.image_path, self.word_to_idx, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
