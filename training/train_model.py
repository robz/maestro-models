import time
import torch
import torch.nn as nn


def get_accuracy(predictions, output):
  return (torch.argmax(predictions, 1) == output).sum().item() / output.numel()


def train(model, train_iter, val_iter, num_epochs, epoch_cb):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  lossfn = nn.CrossEntropyLoss()

  train_len = len(train_iter)
  val_len = len(val_iter)
  best_val_acc = 0.0
  best_model = None

  for epoch in range(num_epochs):
    train_loss = train_acc = val_loss = val_acc = 0.0
    start = time.time()

    model.train()
    for i, batch in enumerate(train_iter):
      optimizer.zero_grad()
      predictions = model(batch['input'])
      loss = lossfn(predictions, batch['output'])
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      train_acc += get_accuracy(predictions, batch['output'])

    model.eval()
    with torch.no_grad():
      for i, batch in enumerate(val_iter):
        predictions = model(batch['input'])
        val_loss += lossfn(predictions, batch['output']).item()
        val_acc += get_accuracy(predictions, batch['output'])

    train_loss /= train_len
    train_acc /= train_len
    val_loss /= val_len
    val_acc /= val_len
    time_elapsed = time.time() - start

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_model = model.state_dict()

    epoch_cb(epoch, train_loss, train_acc, val_loss, val_acc, time_elapsed)

  model.load_state_dict(best_model)
  return model
