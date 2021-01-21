from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import torch


SAVE_MODEL_DIRECTORY = os.getcwd() + '/saved_models'


class EpochMonitor:
  def __init__(self, name, tensorboard_dir, model_dir=SAVE_MODEL_DIRECTORY, model_save_period=50, verbose=False):
    self.name = name
    self.model_dir = model_dir
    self.verbose = verbose
    self.train_losses = []
    self.train_accs = []
    self.val_losses = []
    self.val_accs = []
    self.model_save_period = model_save_period
    self.best_val_loss = float('inf')
    self.start_epoch = 0
    if not os.path.isdir(model_dir):
      os.mkdir(model_dir)
    self.sw = SummaryWriter(F'{tensorboard_dir}/{name}') if tensorboard_dir is not None else None

  def restore(self, model, model_name):
    filename = F'{self.model_dir}/{model_name}'
    state = torch.load(filename)
    model.setArgs(state['args'])
    model.load_state_dict(state['state_dict'])
    args = state['args']
    self.best_val_loss = state['best_val_loss']
    self.start_epoch = state['epoch'] + 1
    val_loss = state['val_loss']
    print(F'restored from {filename} with val loss {val_loss:.3f} best val loss {self.best_val_loss:.3f} and args {args}')

  def epoch_cb(self, epoch, train_loss, train_acc, val_loss, val_acc, time_elapsed, model):
    epoch += self.start_epoch
    print(F'epoch {epoch}; train loss: {train_loss:.3f}, acc: {train_acc:.3f}; validation loss: {val_loss:.3f}, acc: {val_acc:.3f}; time elapsed: {time_elapsed:.3f}')
    self.train_losses.append(train_loss)
    self.train_accs.append(train_acc)
    self.val_losses.append(val_loss)
    self.val_accs.append(val_acc)
    if self.sw is not None:
      self.sw.add_scalar('Loss/train', train_loss, epoch)
      self.sw.add_scalar('Loss/validation', val_loss, epoch)
      self.sw.add_scalar('Accuracy/train', train_acc, epoch)
      self.sw.add_scalar('Accuracy/validation', val_acc, epoch)

    state = {
      'state_dict': model.state_dict(),
      'val_loss': val_loss,
      'best_val_loss': self.best_val_loss,
      'args': model.getArgs(),
      'epoch': epoch,
    }
    self.save(state, F'{self.model_dir}/{self.name}-recent.pt')
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      self.save(state, F'{self.model_dir}/{self.name}-best.pt')
    if epoch % self.model_save_period == 0:
      self.save(state,  F'{self.model_dir}/{self.name}-epoch{epoch}.pt')


  def save(self, state, name):
    if self.verbose:
      print(F'saving model {name}')
    torch.save(state, name)


  def plot(self):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.title('loss over epoch')
    plt.plot(self.train_losses, label='train')
    plt.plot(self.val_losses, label='validation')
    plt.legend()

    plt.subplot(122)
    plt.title('accuracy over epoch')
    plt.plot(self.train_accs, label='train')
    plt.plot(self.val_accs, label='validation')
    plt.legend()

    plt.show()
