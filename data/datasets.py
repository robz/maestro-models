import random
import torch

from data.maestro_data_utils import getTopComposers, MAESTRO_DIRECTORY
from data.gpu_usage import gpu_usage


def augmentIdxs(x):
  pitch_transposition = random.randint(-3, 3)
  time_stretch = 0.95 + random.randint(0, 4) * 0.025

  note_on_idxs = (x >= 0) * (x < 128)
  x[note_on_idxs] = torch.clamp(x[note_on_idxs] + pitch_transposition, 0, 127).long()

  note_off_idxs = (x >= 128) * (x < 256)
  x[note_off_idxs] = torch.clamp(x[note_off_idxs] + pitch_transposition, 128, 255).long()

  time_shifts = (x >= 256) * (x < 356)
  x[time_shifts] = torch.clamp((x[time_shifts] - 256) * time_stretch, 0, 99).long() + 256

  return pitch_transposition, time_stretch


class MaestroMidiDataset(torch.utils.data.Dataset):
  def __init__(self, df, split, directory=MAESTRO_DIRECTORY, num_tracks=None, max_seq_len=10000, device='cpu'):
    self.df = df
    self.split = split
    self.num_tracks = num_tracks if num_tracks is not None else len(df)
    self.max_seq_len = max_seq_len
    self.device = device

    # load the tensors on to the GPU
    self.cache = dict()
    print('filling cache for', split)
    if device == 'cuda':
      gpu_usage.printm()
    for i in range(self.num_tracks):
      filename = self.df['midi_filename'][i]
      self.cache[filename] = torch.load(directory + '-tensors/' + filename + '.pt').to(self.device)
    if device == 'cuda':
      gpu_usage.printm()

  def __len__(self):
    return self.num_tracks

  def __getitem__(self, i, augment_data=True):
    filename = self.df['midi_filename'][i]
    x = self.cache[filename]
    # apply data augmentation: random clipping + pitch transposition + stretching delta times
    start = random.randint(0, max(0, len(x) - self.max_seq_len - 1)) if augment_data else 0
    end = start + self.max_seq_len
    x = x[start:end].long()
    pitch_transposition = augmentIdxs(x)[0] if augment_data else 0
    return {'input': x, 'start': start, 'pitch_transposition': pitch_transposition}

  @classmethod
  def batch_collate(Dataset, data):
    min_len = min([e['input'].shape[0] for e in data])
    return torch.stack([e['input'][:min_len] for e in data])

  @classmethod
  def get_dataloaders(Dataset, df, batch_size=4, shuffle=True, **dataset_args):
    dataloaders = {}
    for split in ['train', 'validation', 'test']:
      df_split = df[df['split'] == split]
      df_split.reset_index(drop=True, inplace=True)
      dataloaders[split] = torch.utils.data.DataLoader(
        Dataset(df=df_split, split=split, **dataset_args),
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        collate_fn=Dataset.batch_collate
      )
    return dataloaders


class MaestroMidiComposer(MaestroMidiDataset):
  def __init__(self, composers, **args):
    super().__init__(**args)
    self.composer_to_idx = {
      composer: torch.tensor(i, device=self.device)
      for i, composer in enumerate(composers)
    }
    self.idx_to_composer = list(composers)

  def __getitem__(self, i, augment_data=True):
    x = super().__getitem__(i, augment_data)['input']
    composer = self.df['canonical_composer'][i]
    return {'input': x, 'output': self.composer_to_idx[composer]}

  @classmethod
  def batch_collate(Dataset, data):
    return {
      'output': torch.stack([e['output'] for e in data]),
      'input': super().batch_collate(data),
    }

  @classmethod
  def get_dataloaders(Dataset, df, num_composers, **dataloader_args):
    composers = getTopComposers(df, num_composers)
    composer_df = df[df.canonical_composer.isin(composers)]
    return super().get_dataloaders(df=composer_df, composers=composers, **dataloader_args)


class MaestroMidiCasual(MaestroMidiDataset):
  def __getitem__(self, i, augment_data=True):
    x = super().__getitem__(i, augment_data)['input']
    return {'input': x[:-1], 'output': x[1:]}

  @classmethod
  def batch_collate(Dataset, data):
    input = super().batch_collate(data)
    seq_len = input.shape[1]
    return {
      'output': torch.stack([e['output'][:seq_len] for e in data]),
      'input': input,
    }

  @classmethod
  def get_dataloaders(Dataset, **dataloader_args):
    if 'max_seq_len' in dataloader_args:
      dataloader_args['max_seq_len'] += 1
    else:
      dataloader_args['max_seq_len'] = 10001
    return super().get_dataloaders(**dataloader_args)


class MaestroMidiCasualComposerConditioning(MaestroMidiCasual):
  def __init__(self, composers, **args):
    super().__init__(**args)
    self.composer_to_idx = {
      composer: torch.tensor(i, device=self.device)
      for i, composer in enumerate(composers)
    }
    self.idx_to_composer = list(composers)

  def __getitem__(self, i, augment_data=True):
    composer = self.df['canonical_composer'][i]
    return {
      **super().__getitem__(i, augment_data),
      'global_conditioning': self.composer_to_idx[composer],
    }

  @classmethod
  def batch_collate(Dataset, data):
    return {
      **super().batch_collate(data),
      'global_conditioning': torch.stack([e['global_conditioning'] for e in data]),
    }

  @classmethod
  def get_dataloaders(Dataset, num_composers, **dataloader_args):
    df = dataloader_args['df']
    composers = getTopComposers(df, num_composers)
    composer_df = df[df.canonical_composer.isin(composers)]
    return super().get_dataloaders(df=composer_df, composers=composers, **dataloader_args)
