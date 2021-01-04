import torch
import torch.nn as nn
import torch.nn.functional as F

from data.maestro_data_utils import NUM_CHANNELS
from models.model_utils import get_pos_enc, SaveArgsModule


class PerformanceRNN(SaveArgsModule):
  def _init(
    self,
    in_channels=NUM_CHANNELS,
    hidden_channels=256,
    num_layers=2,
    dropout=0.1,
    prefix=''
  ):
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, num_layers=num_layers, dropout=dropout)
    self.linear_out = nn.Linear(in_features=hidden_channels, out_features=in_channels)
    self.name = F'{prefix}performance_rnn_ic{in_channels}_hc{hidden_channels}_nl{num_layers}_d{int(dropout * 100)}'

  # x is [batch, seq_len]
  # returns [batch, channels, seq_len]
  def forward(self, x):
    x = x.transpose(0, 1) # LSTM requires [seq_len, batch, channels]
    x = self.embedding(x)
    x = self.lstm(x)[0]
    x = self.linear_out(x)
    return x.transpose(0, 1).transpose(1, 2) # CEL requires [batch, channels, seq_len]

  # prime is [seq_len]
  # returns [steps]
  def forward_steps(self, steps, prime=torch.tensor([256], device='cuda'), greedy=False):
    ret = torch.empty(steps, device='cuda')
    x = prime[:, None] # LSTM requires [seq_len, batch, channels]
    hidden = None
    for i in range(steps):
      output, hidden = self.lstm(self.embedding(x), hidden)
      output = self.linear_out(output[-1:])
      if greedy:
        x = torch.argmax(output, 2)
      else:
        distribution = torch.distributions.Categorical(torch.softmax(output, 2))
        x = distribution.sample()
      ret[i] = x[0, 0]
    return ret.to('cpu')


def get_casual_mask(sz):
  mask = torch.triu(torch.ones(sz, sz)).T
  return (mask - 1).masked_fill_(mask == 0, float('-inf'))


class PerformanceTransformer(SaveArgsModule):
  def _init(
    self,
    in_channels=NUM_CHANNELS,
    max_seq_len=2048,
    nhead=2,
    hidden_channels=256,
    dim_feedforward=512,
    num_layers=3,
    dropout=0.1,
    max_batch_size=4,
    prefix='',
  ):
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.register_buffer('pos_encoding', get_pos_enc(max_seq_len, hidden_channels))
    self.register_buffer('mask', get_casual_mask(max_seq_len))
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.linear_out = nn.Linear(in_features=hidden_channels, out_features=in_channels)
    self.max_seq_len = max_seq_len
    self.name = F'{prefix}performance_trans_ic{in_channels}_hc{hidden_channels}_df{dim_feedforward}_nl{num_layers}_nh{nhead}_d{int(dropout * 100)}_msl{max_seq_len}_mbs{max_batch_size}'

  # x is [batch, seq_len]
  # returns [batch, channels, seq_len]
  def forward(self, x):
    x = x.transpose(0, 1) # transformer requires [seq_len, batch, channels]
    seq_len = x.shape[0]
    x = self.embedding(x) + self.pos_encoding[:seq_len]
    x = self.encoder(x, self.mask[:seq_len, :seq_len])
    x = self.linear_out(x)
    return x.transpose(0, 1).transpose(1, 2) # CEL requires [batch, channels, seq_len]

  # prime is [seq_len]
  # returns [steps]
  def forward_steps(self, steps, prime, greedy=False):
    ret = torch.empty(steps, device='cuda')

    prime_len = len(prime)
    x = prime[:, None] # transformer requires [seq_len, batch, channels]
    x = self.embedding(x) + self.pos_encoding[:prime_len]

    for seq_len in range(prime_len, steps + prime_len):
      output = self.encoder(x, self.mask[:seq_len, :seq_len])[-1:]
      output = self.linear_out(output)
      if greedy:
        output = torch.argmax(output, 2)
      else:
        distribution = torch.distributions.Categorical(torch.softmax(output, 2))
        output = distribution.sample()
      ret[seq_len - prime_len] = output[0, 0]
      output = self.embedding(output) + self.pos_encoding[seq_len:seq_len+1]
      x = torch.cat([x[-self.max_seq_len:], output])

    return ret.to('cpu')


class PerformanceWavenet(SaveArgsModule):
  def _init(
    self,
    in_channels=NUM_CHANNELS,
    hidden_channels=256,
    num_layers=8,
    prefix='',
  ):
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
    self.linear_out = nn.Conv1d(
      in_channels=hidden_channels,
      out_channels=in_channels,
      kernel_size=1,
    )
    self.receptive_field = 2 ** num_layers
    self.name = F'{prefix}performance_wavenet_ic{in_channels}_hc{hidden_channels}_nl{num_layers}'

  # x is [batch, seq_len]
  # returns [seq_len, batch, channels]
  def forward(self, x, pad=True):
    x = self.embedding(x)
    x = x.transpose(1, 2) # conv1d requires [batch, channels, seq_len]
    for i, layer in enumerate(self.layers):
      prev = x
      if pad:
        # pad only left to preserve causality
        x = F.pad(x, (2 ** i, 0))
      x = F.relu(layer(x)) + prev
    x = self.linear_out(x)
    return x

  # prime is [seq_len]
  # returns [steps]
  def forward_steps(self, steps, prime=None, greedy=False):
    if len(prime) < self.receptive_field:
      print(F'warning: prime len smaller than receptive field, so padding ({len(prime)} < {self.receptive_field})')
      pad = True
    else:
      prime = prime[-self.receptive_field:] # only the receptive field matters
      pad = False

    ret = torch.empty(steps, device='cuda')

    output = self.embedding(prime)
    output = output.T[None, ...] # conv1d requires [batch, channels, seq_len]
    for i in range(steps):
      x = output

      for j, layer in enumerate(self.layers):
        prev = x
        if pad:
          # pad only left to preserve causality
          x = F.pad(x, (2 ** j, 0))
        x = F.relu(layer(x))
        x = x + prev[:, :, -x.shape[2]:]
      x = self.linear_out(x)

      if greedy:
        x = torch.argmax(x, 1) # [batch, seq_len]
      else:
        x = torch.softmax(x, 1)
        x = x.transpose(1, 2) # Categorical expects [*, channesl]
        x = torch.distributions.Categorical(x).sample()

      ret[i] = x[0, 0]
      x = self.embedding(x).transpose(1, 2) # conv1d requires [batch, channels, seq_len]
      output = torch.cat([output[:, :, -self.receptive_field:], x], axis=2)

    return ret.to('cpu')
