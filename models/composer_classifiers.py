import torch
import torch.nn as nn
import torch.nn.functional as F


from data.maestro_data_utils import NUM_CHANNELS
from models.model_utils import get_pos_enc, SaveArgsModule


class LSTMClassifier(SaveArgsModule):
  def _init(
    self, 
    num_composers,
    in_channels=NUM_CHANNELS,
    hidden_channels=256,
    num_layers=2,
    dropout=0.1
  ):
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, num_layers=num_layers, dropout=dropout)
    self.linear = nn.Linear(in_features=hidden_channels * 2, out_features=num_composers)
    self.hidden_channels = hidden_channels
    self.name = F'classifier_lstm_nc{num_composers}_ic{in_channels}_hc{hidden_channels}_nl{num_layers}_d{int(dropout * 100)}'

  # x is [batch, seq_len]
  # returns [batch, num_composers]
  def forward(self, x):
    x = x.transpose(0, 1) # LSTM requires [seq_len, batch, channels]
    x = self.embedding(x)
    x = self.lstm(x)[0]
    x = torch.cat([x[0, :, -self.hidden_channels:], x[-1, :, :self.hidden_channels]], dim=-1)
    x = self.linear(x)
    return x


class TransformerClassifier(SaveArgsModule):
  def _init(
    self, 
    num_composers,
    in_channels=NUM_CHANNELS,
    max_seq_len=2048,
    nhead=1, 
    hidden_channels=256,
    dim_feedforward=256, 
    num_layers=2,
    dropout=0.1,
    max_batch_size=4,
  ):
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.register_buffer('pos_encoding', get_pos_enc(max_seq_len, hidden_channels), persistent=False)
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.linear_out = nn.Linear(in_features=hidden_channels, out_features=num_composers)
    #self.register_buffer('query', torch.ones(1, max_batch_size, hidden_channels))
    #self.attention_pool = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=nhead)
    self.name = F'classifier_trans_nc{num_composers}_ic{in_channels}_hc{hidden_channels}_df{dim_feedforward}_nl{num_layers}_nh{nhead}_d{int(dropout * 100)}_msl{max_seq_len}_mbs{max_batch_size}'

  # x is [batch, seq_len]
  # returns [batch, num_composers]
  def forward(self, x):
    x = x.transpose(0, 1) # transformer requires [seq_len, batch, channels]
    x = self.embedding(x)
    x += self.pos_encoding[:x.shape[0]]
    x = self.encoder(x)
    x = torch.mean(x, dim=0)
    #x = self.attention_pool(self.query[:, :x.shape[1]], x, x)[0][0]
    x = self.linear_out(x)
    return x


class ConvClassifier(SaveArgsModule):
  def _init(
    self, 
    num_composers,
    in_channels=NUM_CHANNELS,
    hidden_channels=NUM_CHANNELS,
    use_attention_pooling=False,
    max_batch_size=4,
    attention_dropout=0.0,
  ):
    self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=hidden_channels)
    self.conv1 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3)
    self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3)
    self.linear = nn.Linear(in_features=hidden_channels, out_features=num_composers)
    self.use_attention_pooling = use_attention_pooling
    if use_attention_pooling:
      self.register_buffer('query', torch.ones(1, max_batch_size, hidden_channels))
      self.attention_pool = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=1, dropout=attention_dropout)
    self.name = F'classifier_conv_nc{num_composers}_ic{in_channels}_hc{hidden_channels}_uap{int(use_attention_pooling)}_ad{int(attention_dropout * 100)}_mbs{max_batch_size}'
      
  # x is [batch, seq_len]
  # returns [batch, num_composers]
  def forward(self, x):
    x = self.embedding(x)
    x = x.transpose(1, 2) # conv1d requires [batch, channels, seq_len]
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    if self.use_attention_pooling:
      x = x.transpose(1, 2).transpose(0, 1) # transformer requires [seq_len, batch, channels]
      x = self.attention_pool(self.query[:, :x.shape[1]], x, x)[0][0]
    else:
      x = torch.mean(x, dim=2)
    x = self.linear(x)
    return x
