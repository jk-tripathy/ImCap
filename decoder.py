import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder_Vanilla(nn.Module):
    # Using LSTM with context input in every time step
    def __init__(self, context_dim, embedding_dim, hidden_dim, vocab_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(context_dim + embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, caption, img_ctx):
        _, T = caption.size()
        emb = self.embed(caption)
        img_ctx = torch.cat([img_ctx.unsqueeze(1)] * T, 1)
        lstm_inp = torch.cat([emb, img_ctx], -1)
        output, _ = self.lstm(lstm_inp)
        output = self.linear(output)
        output = F.relu(output)
        return output
