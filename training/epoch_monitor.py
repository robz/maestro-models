import matplotlib.pyplot as plt
import os
import torch


SAVE_MODEL_DIRECTORY = os.getcwd() + '/saved_models'


class EpochMonitor:
  def __init__(self, name, directory=SAVE_MODEL_DIRECTORY, model_save_period=50):
    self.name = name
    self.directory = directory
    self.train_losses = []
    self.train_accs = []
    self.val_losses = []
    self.val_accs = []
    self.model_save_period = model_save_period
    self.best_val_loss = 0
    if not os.path.isdir(directory):
      os.mkdir(directory)

  # state_name can be best, recent, epochn
  def restore(self, model, state_name):
    filename = F'{self.directory}/{self.name}-{state_name}.pt'
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])
    print(F'restoring from {filename}')
    self.best_val_loss = state['val_loss']

  def epoch_cb(self, epoch, train_loss, train_acc, val_loss, val_acc, time_elapsed, model):
    print(F'epoch: {epoch} train loss: {train_loss:.3f}, acc: {train_acc:.3f}; validation loss: {val_loss:.3f}, acc: {val_acc:.3f}; time elapsed: {time_elapsed:.3f}')
    self.train_losses.append(train_loss)
    self.train_accs.append(train_acc)
    self.val_losses.append(val_loss)
    self.val_accs.append(val_acc)
    state = {'state_dict': model.state_dict(), 'val_loss': val_loss}
    torch.save(state, F'{self.directory}/{self.name}-recent.pt')
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      torch.save(state, F'{self.directory}/{self.name}-best.pt')
    if epoch % self.model_save_period == 0:
      torch.save(state, F'{self.directory}/{self.name}-epoch{epoch}.pt')


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
