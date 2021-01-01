from data.maestro_data_utils import getDataFrame
from data.datasets import MaestroMidiComposer
from models.composer_classifiers import ConvClassifier
from training.epoch_monitor import EpochMonitor
from training.train_model import train

df = getDataFrame()

print('loading dataset...')
dataloaders = MaestroMidiComposer.get_dataloaders(df, num_composers=5, device='cuda')

model = ConvClassifier(num_composers=5).to('cuda')
epoch_monitor = EpochMonitor()

print('training...')
best_model = train(
  model,
  dataloaders['train'], 
  dataloaders['validation'], 
  num_epochs=5, 
  epoch_cb=epoch_monitor.epoch_cb
)
