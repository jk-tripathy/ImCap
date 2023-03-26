import pandas as pd
from data import FlikrDataModule, FlikrData, preproc_text

if __name__ == "__main__":
    image_path = 'flikr8k/Images/'
    caption_path = 'flikr8k/captions.txt'

    df = pd.read_csv(caption_path)
    print(df.head(10))
    print(df.shape)

    dm = FlikrDataModule(
        df=df,
        image_path=image_path,
        seed=0,
        batch_size=8
    )

    dm.setup()

    print(len(dm.train_dataloader()))
    print(len(dm.val_dataloader()))
    print(len(dm.test_dataloader()))

    tdm = FlikrData(df, image_path)
    print(tdm.__getitem__(69))



