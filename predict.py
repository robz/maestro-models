import argparse
import os
import torch

from models import performance_predictors
from training.epoch_monitor import SAVE_MODEL_DIRECTORY
from data.midi_utils import idxsToMidi, writeMidi


SAVE_MIDI_DIRECTORY = os.getcwd() + '/music'

parser = argparse.ArgumentParser(
  usage="%(prog)s [TYPE] --model [MODEL] --file [FILE] --steps [N]",
  description="Sample from a model for some number of steps",
)
parser.add_argument('type', choices=[
  'performance_predictor',
  'synthetic_predictor',
])
parser.add_argument(
  "--version",
  action="version",
  version = F"{parser.prog} version 1.0.0",
)
parser.add_argument('--model', choices=['conv', 'lstm', 'transformer'], required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--greedy', action='store_true')

args = parser.parse_args()
print('args:', args)


max_seq_len = 2048

if args.type == 'performance_predictor':
  prime = torch.tensor([256], device='cuda')
elif args.type == 'synthetic_predictor':
  prime = torch.tensor([0], device='cuda')

if args.model == 'conv':
  model = performance_predictors.PerformanceWavenet()
elif args.model == 'lstm':
  model = performance_predictors.PerformanceRNN()
elif args.model == 'transformer':
  model = performance_predictors.PerformanceTransformer(
    max_seq_len=max_seq_len,
  )

name = F'{SAVE_MODEL_DIRECTORY}/{args.file}'
state = torch.load(name)
model.setArgs(state['args'])
model.load_state_dict(state['state_dict'])
val_loss = state['val_loss']
print(F'loaded {name} which had validation loss {val_loss}')
print('predicting...')
result = model.to('cuda').forward_steps(
  args.steps,
  prime=prime,
  greedy=args.greedy,
).to('cpu')

if args.type == 'performance_predictor':
  os.makedirs(SAVE_MIDI_DIRECTORY, exist_ok=True)
  midiname = F'{SAVE_MIDI_DIRECTORY}/{args.file}.mid'
  mf, errors = idxsToMidi(result.long().numpy())
  print('errors:', errors)
  writeMidi(mf, filename=midiname)
  print(F'saved to {midiname}')
elif args.type == 'synthetic_predictor':
  print('prediction:', result)
