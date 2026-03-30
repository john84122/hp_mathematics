import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, inp_dim, output_dim, n_layers, width):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential([])

        if k == 1:

            self.linear_layers.append(nn.Linear(inp_dim, output_dim))
            self.linear_layers.append(nn.ReLU())

        elif k > 1:

            self.linear_layers.append(nn.Linear(inp_dim, width))
            self.linear_layers.append(nn.ReLU())

            for k in range(n_layers - 1):
                self.linear_layers.append(nn.Linear(width, width))
                self.linear_layers.append(nn.ReLU())

            self.linear_layers.append(nn.Linear(width, output_dim))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_layer(x)
        return logits


def main():
    pass

if __name__ == "__main__":
    main()