import torch
import torch.nn as nn

class simple_model(nn.Module):
    def __init__(self, inp_dim, output_dim, n_layers, width):
        super().__init__()
        self.flatten = nn.Flatten()

        all_layers = []
        if n_layers == 1:

            all_layers.append(nn.Linear(inp_dim, output_dim))
            all_layers.append(nn.ReLU())

        elif n_layers > 1:

            all_layers.append(nn.Linear(inp_dim, width))
            all_layers.append(nn.ReLU())

            for _ in range(n_layers - 1):
                all_layers.append(nn.Linear(width, width))
                all_layers.append(nn.ReLU())

            all_layers.append(nn.Linear(width, output_dim))

        self.linear_layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_layers(x)
        return logits


def main():
    pass

if __name__ == "__main__":
    main()