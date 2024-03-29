import argparse

from data.datasets import MaestroMidiComposer, MaestroMidiCasual, SyntheticCopySymbolDataset
from data.maestro_data_utils import getDataFrame
from models import composer_classifiers, performance_predictors
from training.epoch_monitor import EpochMonitor
from training.train_model import train

import json


def model_params_json(s):
  params = json.loads(s)
  assert isinstance(params, dict)
  return params


parser = argparse.ArgumentParser(
  usage="%(prog)s [TYPE] --model [MODEL] --epochs [N]",
  description="Train a model",
)
parser.add_argument('type', choices=[
  'composer_classifier',
  'performance_predictor',
  'synthetic_predictor',
])
parser.add_argument(
  "-v", "--version", action="version",
  version = f"{parser.prog} version 1.0.0"
)
parser.add_argument('--model', choices=['conv', 'lstm', 'transformer'], required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--max_seq_len', type=int, default=2048)
parser.add_argument('--num_composers', type=int, default=5)
parser.add_argument('--restore', type=str)
parser.add_argument('--tensorboard_dir', type=str)
parser.add_argument('--model_params', type=model_params_json, default={})
parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()
print('args:', args)


df = getDataFrame()

if args.type == 'composer_classifier':
  dataloaders = MaestroMidiComposer.get_dataloaders(
    num_composers=args.num_composers,
    df=df,
    max_seq_len=args.max_seq_len,
    device='cuda',
  )
  if args.model == 'conv':
    model = composer_classifiers.ConvClassifier(num_composers=args.num_composers, **args.model_params)
  elif args.model == 'lstm':
    model = composer_classifiers.LSTMClassifier(num_composers=args.num_composers, **args.model_params)
  elif args.model == 'transformer':
    model = composer_classifiers.TransformerClassifier(
      num_composers=args.num_composers,
      max_seq_len=args.max_seq_len,
      **args.model_params
    )
elif args.type == 'performance_predictor':
  dataloaders = MaestroMidiCasual.get_dataloaders(
    df=df,
    max_seq_len=args.max_seq_len,
    device='cuda'
  )
  if args.model == 'conv':
    model = performance_predictors.PerformanceWavenet( **args.model_params)
  elif args.model == 'lstm':
    model = performance_predictors.PerformanceRNN( **args.model_params)
  elif args.model == 'transformer':
    model = performance_predictors.PerformanceTransformer(
      max_seq_len=args.max_seq_len,
      **args.model_params
    )
elif args.type == 'synthetic_predictor':
  prefix = F'{args.type}_'
  num_symbols = 10
  dataloaders = SyntheticCopySymbolDataset.get_dataloaders(
    item_len=args.max_seq_len,
    device='cuda',
  )
  if args.model == 'conv':
    model = performance_predictors.PerformanceWavenet(
      prefix=prefix,
      in_channels=num_symbols,
      **args.model_params
    )
  elif args.model == 'lstm':
    model = performance_predictors.PerformanceRNN(
      prefix=prefix,
      in_channels=num_symbols,
      **args.model_params
    )
  elif args.model == 'transformer':
    model = performance_predictors.PerformanceTransformer(
      prefix=prefix,
      in_channels=num_symbols,
      max_seq_len=args.max_seq_len,
      **args.model_params
    )


epoch_monitor = EpochMonitor(model.name, args.tensorboard_dir, verbose=args.verbose)
if args.restore is not None:
  epoch_monitor.restore(model, args.restore)


print('training...')
train(
  model.to('cuda'),
  dataloaders['train'],
  dataloaders['validation'],
  num_epochs=args.epochs,
  epoch_cb=epoch_monitor.epoch_cb
)
