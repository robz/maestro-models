import matplotlib.pyplot as plt
import os
import torch


SAVE_MODEL_DIRECTORY = os.getcwd() + '/saved_models'


class EpochMonitor:
  def __init__(self, name, directory=SAVE_MODEL_DIRECTORY):
    self.name = name
    self.directory = directory
    self.train_losses = []
    self.train_accs = []
    self.val_losses = []
    self.val_accs = []

    if not os.path.isdir(directory):
      os.mkdir(directory)

  def epoch_cb(self, epoch, train_loss, train_acc, val_loss, val_acc, time_elapsed, model):
    print(F'epoch: {epoch} train loss: {train_loss:.3f}, acc: {train_acc:.3f}; validation loss: {val_loss:.3f}, acc: {val_acc:.3f}; time elapsed: {time_elapsed:.3f}')
    self.train_losses.append(train_loss)
    self.train_accs.append(train_acc)
    self.val_losses.append(val_loss)
    self.val_accs.append(val_acc)
    torch.save(model.state_dict(), F'{self.directory}/{self.name}-recent.pt')

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
