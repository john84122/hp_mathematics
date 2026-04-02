import torch
from hp_math.dnn_fractal.mlp_model import simple_model

def test_loading():
    
    model = simple_model(inp_dim=1500, output_dim=3, n_layers=2, width=10)

def test_model_forward():

    model = simple_model(inp_dim=1500, output_dim=3, n_layers=2, width=10)

    x = torch.randn(1, 1500)

    out = model(x)

    assert out.shape == (1, 3), f"Expected output shape (1, 3), but got {out.shape}"