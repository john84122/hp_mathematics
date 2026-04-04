'''
The sequential version of fractal generation code.
'''

import os
import time
import torch

torch.set_num_threads(1)

import numpy as np

from tqdm import tqdm
from mlp_model import simple_model
from dataloader_for_fractal import synthetic_dataset
from training import train_model_seq


def load_data(data_pth, lb_pth):
    '''
    Loads the data from paths to synthetic dataset and makes a pytorch dataloaders.
    
    Inputs:
        - data_pth (str): Path to the synthetic dataset inputs.
        - lb_pth (str): Path to the synthetic dataset labels.
    '''
    dataset = synthetic_dataset(data_pth, lb_pth)
    train_set, val_set = torch.utils.data.random_split(dataset, [.8, .2])
    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    return dataloader_train, dataloader_val


def load_model(fl = None, bs_path=None):
    '''
    Loads the model from a specified save file. If no save file is given, it initializes random parameters.

    Inputs:
        - fl (str): Path to save file fo the model.
        - bs_path (str): Base path for where the hp_mathematics folder is.

    '''

    model = simple_model(inp_dim=1500, output_dim=3, n_layers=2, width=10)

    if fl is not None:
        model.load_state_dict(torch.load(fl))

    else:
        torch.save(model.state_dict(), bs_path + "/hp_mathematics/data/models/initial_model.pth")

        return model, bs_path + "/hp_mathematics/data/models/initial_model.pth"

    return model, None

def create_fractal(device):
    '''
    A function that when called creates the fractal only for single CPU usage.

    Inputs:
    - device (str): The device used for producing the fractal. For this script, only CPU is valid.
    '''

    #base_pth = "/Users/johannesbauer/Documents/Coding"
    base_pth = "/home/jbauer/code/"


    # Save paths to the inputs and labels of the synthetic dataset.
    dt_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/inputs.npy"
    lb_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/labels.npy"

    print()
    print(dt_fl)
    print()
    assert os.path.exists(dt_fl), "The data file does not exist at the specified path."
    assert os.path.exists(lb_fl), "The labels file does not exist at the specified path."

    # Defines the train and validation dataloader.
    train_dataloader, val_dataloader = load_data(dt_fl, lb_fl)

    # Does the analysis 
    initial_model, sv_fl_for_md = load_model(None, base_pth)

    # Defines the loss function
    loss_func = torch.nn.CrossEntropyLoss()

    first_run = True

    lst_of_train_loss = []
    lst_of_val_loss = []
    lst_of_val_acc = []
    lst_of_n_batches = []
    lst_of_train_acc = []


    dim_v = 0.1

    # Loops over learning rate and beta values to train the model.
    for lr in tqdm(np.arange(0, 0.8, dim_v)):
        for beta in np.arange(0, 0.8, dim_v):
            if first_run:
                first_run = False

            else:
                initial_model = load_model(sv_fl_for_md, base_pth)[0]

            optimizer_sgd = torch.optim.SGD(initial_model.parameters(), lr=lr, momentum=beta)
            train_loss, validation_loss, validation_acc, n_batches, train_acc = train_model_seq(initial_model, train_dataloader, val_dataloader, loss_func, optimizer_sgd)

            lst_of_train_loss.append(train_loss)
            lst_of_train_acc.append(train_acc)
            lst_of_val_loss.append(validation_loss)
            lst_of_val_acc.append(validation_acc)
            lst_of_n_batches.append(n_batches)

    # Saves output of optimization in a .npz file.
    np.savez(base_pth + f"/hp_mathematics/data/synthetic_blobs/training_results_1p5_1p5_{dim_v}_seq.npz", train_loss=np.array(lst_of_train_loss), train_acc=np.array(lst_of_train_acc), val_loss=np.array(lst_of_val_loss), val_acc=np.array(lst_of_val_acc), n_batches=np.array(lst_of_n_batches))

    return lst_of_train_loss, lst_of_val_loss, lst_of_val_acc, lst_of_n_batches, lst_of_train_acc

    #validation(md, validation_set, epoch, loss_func, sv_model_path, device)
def main():
    '''
    The main function that is used for running and doing timing of the fractal generation code.
    '''

    t = time.time()

    create_fractal("cpu")

    total = time.time() - t

    print(f"Total time taken: {total:.2f} seconds")

if __name__ == "__main__":
    main()