import torch
import pytorch_lightning as pl
from torch import optim, nn
from torchmetrics import BLEUScore
from encoder import Encoder_CNN, Encoder_ResNet
from decoder import Decoder_Vanilla, Decoder_Word2Vec


class Model(pl.LightningModule):
    def __init__(self, context_dim, embedding_dim, hidden_dim, max_len, vocab_size, word_to_idx, idx_to_word,
                 pretrained_embeds, num_layers=4, doBaseline=True):
        super().__init__()
        self.save_hyperparameters()
        self.doBaseline = doBaseline
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        if self.doBaseline:
            self.enc_cnn = Encoder_CNN(context_dim)
            self.dec_vanilla = Decoder_Vanilla(context_dim, embedding_dim, hidden_dim, vocab_size, num_layers)
        else:
            self.enc_resnet = Encoder_ResNet(context_dim)
            self.dec_word2vec = Decoder_Word2Vec(pretrained_embeds, context_dim, embedding_dim, hidden_dim, vocab_size, num_layers)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.bleu1 = BLEUScore(n_gram=1)
        self.bleu4 = BLEUScore(n_gram=4)

    def forward(self, image, caption, hidden=None):
        if self.doBaseline:
            img_ctx = self.enc_cnn(image)
            output, hidden = self.dec_vanilla(caption, img_ctx)
        else:
            img_ctx = self.enc_resnet(image)
            output, hidden = self.dec_word2vec(caption, img_ctx, hidden)
        return output, hidden

    def training_step(self, batch, idx):
        image, caption = batch.values()
        output, _ = self(image, caption)
        output = torch.permute(output, (0, 2, 1))
        loss = self.criterion(output, caption)
        self.log("train loss ", loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, idx):
        hidden = None
        images, captions = batch.values()

        preds = torch.LongTensor([]).cuda()

        pred_text = []
        caption_text = []

        for image, caption in zip(images, captions):
            inp_text = '<START>'
            inp = [self.word_to_idx[inp_text]]
            inp.extend([self.word_to_idx['<PAD>'] for _ in range(self.max_len - len(inp))])
            inp = torch.LongTensor([inp]).cuda()
            for i in range(1, self.max_len):
                output, hidden = self(image, inp, hidden)
                output_slice = output[:, -1, :]
                output_word_idx = torch.argmax(output_slice, dim=1).item()
                inp_text = inp_text + ' ' + self.idx_to_word[output_word_idx]

                if output_word_idx == self.word_to_idx['<END>'] or i == self.max_len - 1:
                    preds = torch.cat((preds, output))
                    pred_text.append(inp_text)
                    caption_text.append(' '.join([self.idx_to_word[word.item()] for word in caption]))
                    break
                else:
                    inp[0, i] = output_word_idx

        preds = torch.permute(preds, (0, 2, 1))
        loss = self.criterion(preds, captions)
        bleu1_score = self.bleu1(pred_text, caption_text).item()
        bleu4_score = self.bleu4(pred_text, caption_text).item()

        self.log("val loss", loss, prog_bar=True, logger=True)
        self.log("val BLEU1", bleu1_score, prog_bar=True, logger=True)
        self.log("val BLEU4", bleu4_score, prog_bar=True, logger=True)
        return {'val loss': loss, 'val BLEU1': bleu1_score, 'val BLEU4': bleu4_score}

    def test_step(self, batch, idx):
        hidden = None
        images, captions = batch.values()

        preds = torch.LongTensor([]).cuda()

        pred_text = []
        caption_text = []

        for image, caption in zip(images, captions):
            inp_text = '<START>'
            inp = [self.word_to_idx[inp_text]]
            inp.extend([self.word_to_idx['<PAD>'] for _ in range(self.max_len - len(inp))])
            inp = torch.LongTensor([inp]).cuda()
            for i in range(1, self.max_len):
                output, hidden = self(image, inp, hidden)
                output_slice = output[:, -1, :]
                output_word_idx = torch.argmax(output_slice, dim=1).item()
                inp_text = inp_text + ' ' + self.idx_to_word[output_word_idx]

                if output_word_idx == self.word_to_idx['<END>'] or i == self.max_len - 1:
                    preds = torch.cat((preds, output))
                    pred_text.append(inp_text)
                    caption_text.append(' '.join([self.idx_to_word[word.item()] for word in caption]))
                    break
                else:
                    inp[0, i] = output_word_idx

        preds = torch.permute(preds, (0, 2, 1))
        loss = self.criterion(preds, captions)
        bleu1_score = self.bleu1(pred_text, caption_text).item()
        bleu4_score = self.bleu4(pred_text, caption_text).item()

        self.log("test loss", loss, prog_bar=True, logger=True)
        self.log("test BLEU1", bleu1_score, prog_bar=True, logger=True)
        self.log("test BLEU4", bleu4_score, prog_bar=True, logger=True)
        return {'test loss': loss, 'test BLEU1': bleu1_score, 'test BLEU4': bleu4_score}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
