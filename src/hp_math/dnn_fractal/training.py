'''
Training loop for deep neural network.
'''

import os
import torch

import numpy as np

#from hp_math.dnn_fractal.mlp_model import simple_model
from torch.optim import Adam

def compute_accuracy(pred, true):
  new_data = torch.topk(pred, 1)[1].squeeze().cpu().detach().numpy()
  acc = np.mean(new_data == true.cpu().detach().numpy())

  return acc

def validation(md, validation_set, epoch, loss_func, sv_model_path, device):
  md.eval()

  loss_term = 0
  acc_term = 0
  n_batches_term = 0

  for batch, (X, y) in enumerate(validation_set):
      X = X.to(device)
      y = y.to(device)

      prediction = md(X)

      loss_val = loss_func(prediction, y)

      loss_term += loss_val.item()
      acc_term += compute_accuracy(prediction, y)
      n_batches_term += 1

  ave_loss = loss_term/n_batches_term
  ave_acc = (acc_term/n_batches_term)

  #print("--- Validation ---")
  #print(f"Epoch: {epoch}, Average Loss: {ave_loss:.5f}, Accuracy: {ave_acc:.5f}%")
  #print()

  sv_path = os.path.join(sv_model_path, f"model_{epoch}.params")


  #torch.save(md.state_dict(), sv_path)

  return ave_loss, ave_acc

def train_model_seq(model, train_set, validation_set, loss_func, optimizer, **kwargs):

    basePath = kwargs.get("basePath", "cool")
    device = kwargs.get("device", "cpu")

    num_epochs = kwargs.get("num_epochs", 60)
    sv_model_path = kwargs.get("sv_dir", basePath + os.sep + "lenet_runs")
    max_iter = 75
    max_threshold = 1e-5


    os.makedirs(sv_model_path, exist_ok=True)

    prev_loss = float('inf')
    n_epoch = 0
    n_batches = 0

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_acc = 0

        optimizer.zero_grad()
        for _, (X, y) in enumerate(train_set):

            X = X.to(device)
            y = y.to(device)

            prediction = model(X)

            loss = loss_func(input = prediction, target = y)

            loss.backward()

            total_loss += loss.item()
            n_batches += 1

            total_acc += (compute_accuracy(prediction,  y))

        optimizer.step()
        n_epoch += 1
        if n_epoch >= max_iter:
            break

        if np.abs(total_loss - prev_loss) < max_threshold:
            break

        if np.isnan(total_loss):
            total_loss = -1
            break
        prev_loss = total_loss
    
    validation_loss_acc = validation(model, validation_set, epoch, loss_func, sv_model_path, device)
    train_loss, train_acc = (total_loss/n_batches, total_acc/n_batches)

    return train_loss, validation_loss_acc[0], validation_loss_acc[1], n_batches, train_acc

def main():
    pass

if __name__ == "__main__":
    main()