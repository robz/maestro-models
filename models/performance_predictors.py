import torch
import torch.nn as nn
import torch.nn.functional as F


from maestro_data_utils import NUM_CHANNELS


class PerformanceRNN(nn.Module):
  def __init__(
    self, 
    in_channels=NUM_CHANNELS,
    hidden_channels=256,
    num_layers=2,
    dropout=0.1
  ):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, num_layers=num_layers, dropout=dropout)
    self.linear_out = nn.Linear(in_features=hidden_channels, out_features=in_channels)

  # x is [batch, seq_len]
  # returns [batch, channels, seq_len]
  def forward(self, x):
    x = x.transpose(0, 1) # LSTM requires [seq_len, batch, channels]
    x = self.embedding(x)
    x = self.lstm(x)[0]
    x = self.linear_out(x)
    return x.transpose(0, 1).transpose(1, 2) # CEL requires [batch, channels, seq_len]


def get_casual_mask(sz):
  mask = torch.triu(torch.ones(sz, sz)).T
  return (mask - 1).masked_fill_(mask == 0, float('-inf'))


class PerformanceTransformer(nn.Module):
  def __init__(
    self, 
    in_channels=NUM_CHANNELS,
    max_seq_len=2048,
    nhead=1, 
    hidden_channels=256,
    dim_feedforward=512, 
    num_layers=2,
    dropout=0.1,
    max_batch_size=4,
  ):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.register_buffer('pos_encoding', get_pos_enc(max_seq_len, hidden_channels))
    self.register_buffer('mask', get_casual_mask(max_seq_len))
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.linear_out = nn.Linear(in_features=hidden_channels, out_features=in_channels)

  # x is [batch, seq_len]
  # returns [batch, channels, seq_len]
  def forward(self, x):
    x = x.transpose(0, 1) # transformer requires [seq_len, batch, channels]
    x = self.embedding(x)
    seq_len = x.shape[0]
    x += self.pos_encoding[:seq_len]
    x = self.encoder(x, self.mask[:seq_len, :seq_len])
    x = self.linear_out(x)
    return x.transpose(0, 1).transpose(1, 2) # CEL requires [batch, channels, seq_len]


class PerformanceWavenet(nn.Module):
  def __init__(
    self, 
    in_channels=NUM_CHANNELS,
    hidden_channels=256,
    num_layers=8,
  ):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    layers = []
    for i in range(num_layers):
      layers.append(nn.Conv1d(
        in_channels=hidden_channels,
        out_channels=hidden_channels,
        kernel_size=2,
        dilation=2**i,
      ))
    self.layers = nn.ModuleList(layers)
    self.linear_out = nn.Linear(in_features=hidden_channels, out_features=in_channels)
    self.receptive_field = 2 ** num_layers

  # x is [batch, seq_len]
  # returns [seq_len, batch, channels]
  def forward(self, x, pad=True):
    x = self.embedding(x)
    x = x.transpose(1, 2) # conv1d requires [batch, channels, seq_len]
    for i, layer in enumerate(self.layers):
      if pad:
        # pad only left to preserve causality
        x = F.pad(x, (2 ** i, 0))
      x = layer(x)
    x = x.transpose(1, 2) # Linear requires [*, channels]
    x = self.linear_out(x)
    return x.transpose(1, 2) # CEL requires [batch, channels, seq_len]
