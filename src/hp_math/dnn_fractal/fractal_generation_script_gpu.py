'''
Script to generate fractals using multiple GPUs.
'''

import os
import torch
import itertools

import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, current_process, Queue

from mlp_model import simple_model
from dataloader_for_fractal import synthetic_dataset
from training import train_model_seq

import time

from tqdm.contrib.concurrent import process_map

cuda_queue = None

def _pool_initializer(q):

    global cuda_queue
    cuda_queue = q

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

def training_given_parameters(params):
    '''
    The training function that is parallelized over multiple learning rates and momentum pairs.

    Inputs:
        - params (tuple): A tuple containing the learning rate, momentum, train dataloader, validation dataloader, save folder for model, loss function, and base path.
    '''

    try:
        cuda_id = cuda_queue.get(timeout=60)

    except:
        raise RuntimeError("Timed out waiting for the device")
    
    try:
        device = f"cuda:{cuda_id}"
        lr_beta, loss_func, sv_fl_for_md, base_pth = params

        dt_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/inputs.npy"
        lb_fl = base_pth + "/hp_mathematics/data/synthetic_blobs/labels.npy"

        assert os.path.exists(dt_fl), "The data file does not exist at the specified path."
        assert os.path.exists(lb_fl), "The labels file does not exist at the specified path."

        train_dataloader, val_dataloader = load_data(dt_fl, lb_fl)

        lr, beta = lr_beta

        initial_model = load_model(sv_fl_for_md, base_pth)[0]

        initial_model.to(device)

        optimizer_sgd = torch.optim.SGD(initial_model.parameters(), lr=lr, momentum=beta)
        train_loss, validation_loss, validation_acc, n_batches, train_acc = train_model_seq(initial_model, train_dataloader, val_dataloader, loss_func, optimizer_sgd, device=device)

        initial_model.cpu()

    finally:
        cuda_queue.put(cuda_id)

    print(f"done with: {lr}, {beta}")

    return train_loss, validation_loss, validation_acc, n_batches, train_acc

def create_fractal():
    '''
    A function that when called creates the fractal. Input is the device used for producing the fractal.

    - To change number of devices, you need to change in the code  with n_gpus and n_processes_per_gpu.
    '''

    base_pth = "/home/jbauer/code" #"/Users/johannesbauer/Documents/Coding"

    initial_model, sv_fl_for_md = load_model(None, base_pth)

    loss_func = torch.nn.CrossEntropyLoss()

    lst_of_train_loss = []
    lst_of_val_loss = []
    lst_of_val_acc = []
    lst_of_n_batches = []
    lst_of_train_acc = []

    n_steps = 500
    scale_val = np.linspace(1/n_steps, 0.8, n_steps)

    # Creates the pairs of learning rates and momentum.
    param_products = list(itertools.product(scale_val, scale_val))
    param_products_vals = [(k, loss_func, sv_fl_for_md, base_pth) for k in param_products]

    # Defines the number of GPUs being used nad processes run per GPU.
    n_gpus = 4
    n_processes_per_gpu = 10

    # Creates the paths to the GPUs.

    cuda_queue = Queue()

    for gpu_ids in range(n_gpus):
        for _ in range(n_processes_per_gpu):
            cuda_queue.put(gpu_ids)

    # Partitions task onto multiple GPUs without overlapping processes on the same GPU.
    with Pool(n_gpus*n_processes_per_gpu, initializer=_pool_initializer, initargs=(cuda_queue,)) as pool:
        results = pool.map(training_given_parameters, param_products_vals)  
    
    lst_of_train_loss = [result[0] for result in results]
    lst_of_val_loss = [result[1] for result in results]
    lst_of_n_batches = [result[3] for result in results]
    lst_of_val_acc = [result[2] for result in results]
    lst_of_train_acc = [result[4] for result in results]

    np.savez(base_pth + f"/hp_mathematics/data/synthetic_blobs/training_results_new_{n_steps}_gpu.npz", train_loss=np.array(lst_of_train_loss), train_acc=np.array(lst_of_train_acc), val_loss=np.array(lst_of_val_loss), val_acc=np.array(lst_of_val_acc), n_batches=np.array(lst_of_n_batches))

    return lst_of_train_loss, lst_of_val_loss, lst_of_val_acc, lst_of_n_batches, lst_of_train_acc

def main():
    '''
    The main function that is used for running and doing timing of the fractal generation code.
    '''

    t = time.time()

    create_fractal()

    total = time.time() - t

    print()
    print(f"Total time taken: {total:.2f} seconds")

if __name__ == "__main__":
    main()