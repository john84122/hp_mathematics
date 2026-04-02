import os
import torch
import itertools

import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

from mlp_model import simple_model
from dataloader_for_fractal import synthetic_dataset
from training import train_model_seq


sv_fl_for_md = "/hp_mathematics/data/models/initial_model.pth"
base_pth = "/Users/johannesbauer/Documents/Coding"

def load_data(data_pth, lb_pth):
    dataset = synthetic_dataset(data_pth, lb_pth)
    train_set, val_set = torch.utils.data.random_split(dataset, [.8, .2])
    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    return dataloader_train, dataloader_val


def load_model(fl = None, bs_path=None):

    model = simple_model(inp_dim=1500, output_dim=3, n_layers=2, width=10)

    if fl is not None:
        model.load_state_dict(torch.load(fl))

    else:
        torch.save(model.state_dict(), bs_path + "/hp_mathematics/data/models/initial_model.pth")

        return model, bs_path + "/hp_mathematics/data/models/initial_model.pth"

    return model, None

def training_given_parameters(params):

    lr_beta, train_dataloader, val_dataloader, loss_func, sv_fl_for_md, base_pth = params

    lr, beta = lr_beta

    initial_model = load_model(sv_fl_for_md, base_pth)[0]

    optimizer_sgd = torch.optim.SGD(initial_model.parameters(), lr=lr, momentum=beta)
    train_loss, validation_loss, validation_acc, n_batches, train_acc = train_model_seq(initial_model, train_dataloader, val_dataloader, loss_func, optimizer_sgd)

    return train_loss, validation_loss, validation_acc, n_batches, train_acc

def create_fractal(device):

    base_pth = "/Users/johannesbauer/Documents/Coding"

    dt_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/inputs.npy"
    lb_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/labels.npy"

    print()
    print(dt_fl)
    print()
    assert os.path.exists(dt_fl), "The data file does not exist at the specified path."
    assert os.path.exists(lb_fl), "The labels file does not exist at the specified path."

    train_dataloader, val_dataloader = load_data(dt_fl, lb_fl)

    initial_model, sv_fl_for_md = load_model(None, base_pth)

    loss_func = torch.nn.CrossEntropyLoss()

    first_run = True

    lst_of_train_loss = []
    lst_of_val_loss = []
    lst_of_val_acc = []
    lst_of_n_batches = []
    lst_of_train_acc = []

    param_products = list(itertools.product(np.arange(0, 3, 0.05), np.arange(0, 3, 0.05)))

    param_products_vals = [(k, train_dataloader, val_dataloader, loss_func, sv_fl_for_md, base_pth) for k in param_products]

    with Pool(2) as pool:
        results = pool.map(training_given_parameters, param_products_vals)  
    
    lst_of_train_loss = [result[0] for result in results]
    lst_of_val_loss = [result[1] for result in results]
    lst_of_n_batches = [result[3] for result in results]
    lst_of_val_acc = [result[2] for result in results]
    lst_of_train_acc = [result[4] for result in results]

    np.savez(base_pth + "/hp_mathematics/data/synthetic_blobs/training_results_3_3_0.05.npz", train_loss=np.array(lst_of_train_loss), train_acc=np.array(lst_of_train_acc), val_loss=np.array(lst_of_val_loss), val_acc=np.array(lst_of_val_acc), n_batches=np.array(lst_of_n_batches))

    return lst_of_train_loss, lst_of_val_loss, lst_of_val_acc, lst_of_n_batches, lst_of_train_acc

def main():
    create_fractal("cpu")


if __name__ == "__main__":
    main()