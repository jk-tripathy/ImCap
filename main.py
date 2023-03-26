import os
import torch
import pandas as pd
import pytorch_lightning as pl
from data import FlikrDataModule
from model import Model


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    image_path = 'flikr8k/Images/'
    caption_path = 'flikr8k/captions.txt'

    df = pd.read_csv(caption_path)

    dm = FlikrDataModule(
        df=df,
        image_path=image_path,
        seed=0,
        batch_size=16,
    )
    doBaseline = False
    dm.setup(doBaseline=doBaseline)
    max_len, vocab_size, word_to_idx, idx_to_word, pretrained_embeds = dm.get_params()

    model = Model(
        context_dim=512,
        embedding_dim=512,
        hidden_dim=512,
        max_len=max_len,
        vocab_size=dm.vocab_size,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        pretrained_embeds=pretrained_embeds,
        doBaseline=doBaseline
    )

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator='gpu',
        devices=1,
        num_sanity_val_steps=25,
        log_every_n_steps=100,
        default_root_dir=os.getcwd(),
        enable_checkpointing=False,
    )

    trainer.fit(model, dm)
