import os
import torch
import pandas as pd
import lightning.pytorch as pl
from data import FlikrDataModule
from model import Model


def prepare_data(df):
    df['caption'] = df['caption'].apply(lambda x: x.lower())
    df['caption'] = df['caption'].apply(lambda x: x.replace('[^A-Za-z]', ''))
    df['caption'] = df['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    df['caption'] = 'startseq ' + df['caption'] + ' endseq'

    max_len = 0
    idx_to_word = {
        0: '<PAD>',
        1: '<UNK>'
    }
    word_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    for item in df['caption']:
        words = item.split()
        if max_len < len(words):
            max_len = len(words)

        for word in item.split():
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                idx_to_word[len(idx_to_word)] = word

    vocab_size = len(idx_to_word)

    image = []
    caption = []
    target = []

    for row in df.iterrows():
        seq = row[1]['caption'].split()

        for i in range(1, len(seq)):
            cut_seq = seq[:i]
            proc_seq = [word_to_idx[word] if word in word_to_idx else word_to_idx['<UNK>'] for word in cut_seq]
            proc_seq.extend([word_to_idx['<PAD>'] for _ in range(max_len - len(proc_seq))])
            caption.append(proc_seq)
            target.append(word_to_idx[seq[i]])
            image.append(row[1]['image'])

    new_df = pd.DataFrame({
        'image': image,
        'caption': caption,
        'target': target
    })

    return new_df, max_len, vocab_size, word_to_idx, idx_to_word


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    image_path = 'flikr8k/Images/'
    caption_path = 'flikr8k/captions.txt'

    df = pd.read_csv(caption_path)
    new_df, max_len, vocab_size, word_to_idx, idx_to_word = prepare_data(df)

    dm = FlikrDataModule(
        df=new_df,
        image_path=image_path,
        seed=0,
        batch_size=8,
        max_len=max_len,
        vocab_size=vocab_size,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
    )
    dm.setup()

    model = Model(
        context_dim=512,
        embedding_dim=512,
        hidden_dim=512,
        vocab_size=dm.vocab_size
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=1,
        num_sanity_val_steps=0,
        default_root_dir=os.getcwd()
    )

    trainer.fit(model, dm)
