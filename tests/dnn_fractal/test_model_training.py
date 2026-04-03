import os
import torch
import numpy as np

from hp_math.dnn_fractal.mlp_model import simple_model
from hp_math.dnn_fractal.training import train_model_seq
from hp_math.dnn_fractal.dataloader_for_fractal import synthetic_dataset



def load_data(data_pth, lb_pth):
    dataset = synthetic_dataset(data_pth, lb_pth)
    train_set, val_set = torch.utils.data.random_split(dataset, [.8, .2])
    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    return dataloader_train, dataloader_val

def test_training():
    
    base_pth = "/home/jbauer/code/"

    dt_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/inputs.npy"
    lb_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/labels.npy"

    print()
    print(dt_fl)
    print()
    assert os.path.exists(dt_fl), "The data file does not exist at the specified path."
    assert os.path.exists(lb_fl), "The labels file does not exist at the specified path."

    train_dataloader, val_dataloader = load_data(dt_fl, lb_fl)

    loss_func = torch.nn.CrossEntropyLoss()
    #initial_model, sv_fl_for_md = load_model(None, base_pth)

    lr = 0.0005
    beta = 0.5
    initial_model = simple_model(inp_dim=1500, output_dim=3, n_layers=2, width=10)
    optimizer_sgd = torch.optim.SGD(initial_model.parameters(), lr=lr, momentum=beta)
    train_loss, validation_loss, validation_acc, n_batches, train_acc = train_model_seq(initial_model, train_dataloader, val_dataloader, loss_func, optimizer_sgd)

    assert train_loss is not None, "Train loss should not be None."
    assert train_acc <= 1.0, f"Train accuracy should be less than 1.0 but is {train_acc}."
    assert validation_loss is not None, "Validation loss should not be None."
    assert validation_acc <= 1.0, f"Validation accuracy should be less than 1.0 but is {validation_acc}."
    assert n_batches > 0, f"Number of batches should be positive but is {n_batches}."


def test_training_large():
    
    base_pth = "/home/jbauer/code/"

    dt_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/inputs.npy"
    lb_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/labels.npy"

    print()
    print(dt_fl)
    print()
    assert os.path.exists(dt_fl), "The data file does not exist at the specified path."
    assert os.path.exists(lb_fl), "The labels file does not exist at the specified path."

    train_dataloader, val_dataloader = load_data(dt_fl, lb_fl)

    loss_func = torch.nn.CrossEntropyLoss()
    #initial_model, sv_fl_for_md = load_model(None, base_pth)

    lr = 0.05
    beta = 1
    initial_model = simple_model(inp_dim=1500, output_dim=3, n_layers=2, width=10)
    optimizer_sgd = torch.optim.SGD(initial_model.parameters(), lr=lr, momentum=beta)
    train_loss, validation_loss, validation_acc, n_batches, train_acc = train_model_seq(initial_model, train_dataloader, val_dataloader, loss_func, optimizer_sgd)


    assert np.isnan(train_loss) == False, "Train loss should not be NaN."
    assert np.isnan(train_acc) == False, "Train accuracy should not be NaN."
    assert train_loss is not None, "Train loss should not be None."
    assert train_acc <= 1.0, f"Train accuracy should be less than or equal to 1.0 but is {train_acc}."
    assert validation_loss is not None, "Validation loss should not be None."
    assert validation_acc <= 1.0, f"Validation accuracy should be less than 1.0 but is {validation_acc}."
    assert n_batches > 0, f"Number of batches should be positive but is {n_batches}."