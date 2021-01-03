import os
import time
import torch

from data.maestro_data_utils import getDataFrame, MAESTRO_DIRECTORY
from data.midi_utils import midiToIdxs, readMidi


def preprocessAndSaveToDisk(df, directory=MAESTRO_DIRECTORY, override=False):
  dirname = F'{directory}-tensors'
  if os.path.isdir(dirname):
    if override:
      print(F'overriding existing directory {dirname}')
    else:
      print(F'directory {dirname} already exists, exiting')
      return

  os.makedirs(dirname)

  num_tracks = len(df)
  print(F'Processing {num_tracks} tracks and saving to {dirname}')
  print('This will take about 10 minutes.')
  print('Tracks processed: ', end='', flush=True)

  start = time.time() 
  for i in range(num_tracks):
    filename = df['midi_filename'][i]
    mf = readMidi(F'{directory}/{filename}')
    x = torch.ShortTensor(midiToIdxs(mf))
    os.makedirs(F'{dirname}/{os.path.dirname(filename)}', exist_ok=True)
    torch.save(x, F'{dirname}/{filename}.pt')
    if i % 10 == 0:
      print(i + 1, end=' ', flush=True)
  print()

  time_elapsed = time.time() - start
  print(F'done. time elapsed: {time_elapsed}')


if __name__ == "__main__":
  preprocessAndSaveToDisk(getDataFrame())  
