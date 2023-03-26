import os
import torch
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor


def preproc_text(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace('[^A-Za-z]', ''))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = 'startseq ' + data['caption'] + ' endseq'

    max_len = 0
    idx_to_word = {
        0: '<PAD>',
        1: '<UNK>'
    }
    word_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    for item in data['caption']:
        words = item.split()
        if max_len < len(words):
            max_len = len(words)

        for word in item.split():
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                idx_to_word[len(idx_to_word)] = word

    return idx_to_word, word_to_idx, max_len


class FlikrData(Dataset):
    def __init__(self, df, image_path, usePretrainedTokenizer=False):
        self.df = df
        self.image_path = image_path

        if not usePretrainedTokenizer:
            self.idx_to_word, self.word_to_idx, self.max_len = preproc_text(self.df)
            self.vocab_size = len(self.idx_to_word)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        item = self.df.iloc[index]
        img = Image.open(os.path.join(self.image_path, item.image))
        proc_img = image_processor(img, return_tensors="pt")['pixel_values']

        caption = item.caption

        proc_caption = [self.word_to_idx[word] for word in caption.split()]
        proc_caption.extend([self.word_to_idx['<PAD>'] for _ in range(self.max_len - len(proc_caption))])
        proc_caption = torch.LongTensor(proc_caption)

        return {'image': proc_img, 'caption': proc_caption}


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
