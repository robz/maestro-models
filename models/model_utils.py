import torch
 

def get_pos_enc(seq_len, d_model):
  pos = torch.arange(0, seq_len).unsqueeze(1)
  i2 = torch.arange(0, d_model, 2)
  x = pos / torch.pow(10_000, i2 / d_model)
  pos_enc = torch.empty(seq_len, 1, d_model)
  pos_enc[:, 0, 0::2] = torch.sin(x)
  pos_enc[:, 0, 1::2] = torch.cos(x)
  return pos_enc
