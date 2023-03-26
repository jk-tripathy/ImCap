import os
import torch
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor


class FlikrData(Dataset):
    def __init__(self, df, image_path):
        self.df = df
        self.image_path = image_path
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        img = Image.open(os.path.join(self.image_path, item.image))
        proc_img = self.image_processor(img, return_tensors="pt")

        proc_caption = torch.LongTensor(item.caption)

        return {'image': proc_img['pixel_values'], 'caption': proc_caption}


class FlikrDataModule(pl.LightningDataModule):
    def __init__(self, df, image_path, seed, batch_size):
        super().__init__()
        self.df = df
        self.image_path = image_path
        self.seed = seed
        self.batch_size = batch_size

    def build_vocab(self, df):
        tokenizer = get_tokenizer("basic_english")

        idx_to_word = {
            0: '<PAD>',
            1: '<UNK>',
            2: '<START>',
            3: '<END>',
        }
        word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
        }
        for item in df['caption']:
            item = tokenizer(item)
            for word in item:
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
                    idx_to_word[len(idx_to_word)] = word

        vocab_size = len(idx_to_word)

        return vocab_size, word_to_idx, idx_to_word

    def pad_data(self, df, isTrain):
        image = []
        caption = []

        df['caption'] = '<START> ' + df['caption'] + ' <END>'
        if isTrain:
            max_len = max([len(item.split()) for item in df['caption']])
        else:
            max_len = self.max_len

        for row in df.iterrows():
            tokens = row[1]['caption'].split()
            proc_seq = [self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<UNK>'] for word in tokens]
            proc_seq.extend([self.word_to_idx['<PAD>'] for _ in range(max_len - len(proc_seq))])
            caption.append(proc_seq)
            image.append(row[1]['image'])

        new_df = pd.DataFrame({
            'image': image,
            'caption': caption,
        })

        return new_df, max_len

    def get_params(self):
        return self.max_len, self.vocab_size, self.word_to_idx, self.idx_to_word, self.pretrained_embeds

    def setup(self, stage='fit', doBaseline=True):
        train_val_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=self.seed, shuffle=False)
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=0.05, random_state=self.seed, shuffle=False)

        if doBaseline:
            self.pretrained_embeds = None
            self.vocab_size, self.word_to_idx, self.idx_to_word = self.build_vocab(self.train_df)
        else:
            glove_vectors = GloVe()
            glove_vocab = vocab(glove_vectors.stoi)
            glove_vocab.insert_token("<PAD>", 0)
            glove_vocab.insert_token("<UNK>", 1)
            glove_vocab.insert_token("<START>", 2)
            glove_vocab.insert_token("<END>", 3)
            glove_vocab.set_default_index(1)
            self.pretrained_embeds = glove_vectors.vectors
            self.word_to_idx = glove_vocab.get_stoi()
            self.idx_to_word = glove_vocab.get_itos()
            self.vocab_size = len(self.idx_to_word)

        self.train_df, self.max_len = self.pad_data(self.train_df, isTrain=True)
        self.val_df, _ = self.pad_data(self.val_df, isTrain=False)
        self.test_df, _ = self.pad_data(self.test_df, isTrain=False)

        if stage == 'fit':
            self.train_dataset = FlikrData(self.train_df, self.image_path)
            self.dev_dataset = FlikrData(self.val_df, self.image_path)
        elif stage == 'predict':
            self.test_dataset = FlikrData(self.test_df, self.image_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
