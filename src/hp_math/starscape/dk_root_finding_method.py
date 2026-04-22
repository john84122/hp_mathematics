'''
Code for NUMBA single GPU optimization for fractals.
'''

import os

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import math

from numba import cuda
from tqdm import tqdm


@cuda.jit(device=True)
def compute_poly_value(k, coefs, degs):

    sum_v = 0

    for (coeff, deg) in zip(coefs, degs):
        sum_v += coeff*(k**deg)

    return sum_v

# https://www-sciencedirect-com.libweb.lib.utsa.edu/science/article/pii/S1877750316304641
@cuda.jit(device=True)
def compute_fractional_val_dk(initial_vals, idx, current_root):

    strange_sum = 1

    for n_idx in range(len(initial_vals)):
        if n_idx != idx:
            strange_sum *= (current_root - initial_vals[n_idx])

    return strange_sum

@cuda.jit
def apply_to_input(initial_vals, coefs, degs, final_out):
    idx = cuda.grid(1)
    current_root = initial_vals[idx]

    if idx >= initial_vals.shape[0]:
        return

    z_eval = compute_poly_value(current_root, coefs, degs)

    new_prod_val = compute_fractional_val_dk(initial_vals, idx, current_root)

    if new_prod_val == 0.0:
        final_out[idx] = current_root
        return

    new_guess = current_root - (z_eval/new_prod_val)

    final_out[idx] = new_guess

@cuda.jit
def check_solutions(initial_guess, new_guess, error):
    idx = cuda.grid(1)

    if idx >= initial_guess.shape[0]:
        return

    diff = new_guess[idx] - initial_guess[idx]
    error[idx] = (diff.real ** 2 + diff.imag ** 2) ** 0.5

def dk_method(initial_vals, coeffs, deg, threads_per_block, max_iter, max_error):
    vals_d   = cuda.to_device(initial_vals.astype(np.complex128))
    coeffs_d = cuda.to_device(coeffs.astype(np.complex128))
    degs_d   = cuda.to_device(deg.astype(np.float64))

    m = len(initial_vals)

    blocks = (m + threads_per_block - 1) // threads_per_block
    griddim   = (blocks,)
    blockdim  = (threads_per_block,)
 
    final_components = np.zeros(initial_vals.shape, dtype=np.complex128)
    final_gpu_comp = cuda.to_device(final_components)

    errors_comp = np.zeros(initial_vals.shape)
    error_gpu = cuda.to_device(errors_comp)

    error = 1
    iter_val = 0

    while (error > max_error) and (iter_val < max_iter):

        apply_to_input[griddim, blockdim](vals_d, coeffs_d, degs_d, final_gpu_comp)
        cuda.synchronize()

        check_solutions[griddim, blockdim](vals_d, final_gpu_comp, error_gpu)
        cuda.synchronize()

        error_host = error_gpu.copy_to_host()

        error = float(np.max(error_host))
        iter_val += 1
    
        vals_d, final_gpu_comp = final_gpu_comp, vals_d

    final_found_roots = vals_d.copy_to_host()

    return final_found_roots, iter_val

def main():
    n_roots = 3
    coeffs = np.array([ 1.0+0j, -1.0+0j], dtype=np.complex128)
    degs   = np.array([ n_roots,     0.0   ], dtype=np.float64)
 
    angles  = np.linspace(0, 2 * np.pi, n_roots + 1, endpoint=False)[1:]
    initial = np.exp(1j * angles) * 0.9
 
    print("Initial guesses:", initial)
    roots, iters = dk_method(initial, coeffs, degs, 64, 1000, max_error=1e-10)
    print(f"Number of iterations: {iters}")
    print()
    print("Roots found :", roots)
    print("Residuals   :", np.abs(roots**3 - 1))

if __name__ == "__main__":
    main()