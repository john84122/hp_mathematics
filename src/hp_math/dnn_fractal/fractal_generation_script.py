import torch

import numpy as np

from mlp_model import MLP
from dataloader_for_fractal import FractalDataset
from training import train_model_seq


def load_data(sv_fl):
    dataset = FractalDataset(sv_fl)
    train_set, val_set = torch.utils.data.random_split(dataset, [.8, .2])
    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    return dataloader_train, dataloader_val


def load_model(fl = None):

    model = MLP(inp_dim=1500, output_dim=3, n_layers=2, width=10)

    if fl is not None:
        model.load_state_dict(torch.load(fl))

    else:
        torch.save(model.state_dict(), "/home/jbauer/code/hp_mathematics/data/models/initial_model.pth")

        return model, "/home/jbauer/code/hp_mathematics/data/models/initial_model.pth"

    return model, None

def create_fractal(device):

    sv_fl = "/home/jbauer/code/hp_mathematics/data/synthetic_blobs/fractal_blob_1000.pt"

    train_dataloader, val_dataloader = load_data(sv_fl)

    initial_model, sv_fl_for_md = load_model()

    loss_func = torch.nn.CrossEntropyLoss()

    for lr in np.arange(0, 10, 0.5):
        for beta in np.arange(0, 10, 0.5):
            optimizer_sgd = torch.optim.SGD(initial_model.parameters(), lr=lr, momentum=beta)
            out = train_model_seq(initial_model, train_dataloader, val_dataloader, loss_func, optimizer_sgd, device, num_epochs=10)


    #validation(md, validation_set, epoch, loss_func, sv_model_path, device)
def main():
    pass


if __name__ == "__main__":
    main()