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


griddim = 1,2
blockdim = 3,4

@cuda.jit(device=True)
def compute_poly_value(k, coefs, degs):

    sum_v = 0

    for (coeff, deg) in zip(coefs, degs):
        sum_v += coeff*(k**deg)

    return sum_v

@cuda.jit(device=True)
def compute_deriv_poly(k, coefs, degs):

    sum_v = 0

    for (coeff, deg) in zip(coefs, degs):

        sum_v += (coeff * (deg))*(k**(deg - 1))

    return sum_v

@cuda.jit(device=True)
def compute_fractional_val(initial_vals, idx, current_root):

    strange_sum = 0

    for n_idx in range(len(initial_vals)):
        if n_idx != idx:
            strange_sum += 1/(current_root - initial_vals[n_idx])

    return strange_sum

@cuda.jit
def apply_to_input(initial_vals, coefs, degs, final_out):
    idx = cuda.grid(1)
    current_root = initial_vals[idx]

    if idx >= initial_vals.shape[0]:
        return

    z_eval = compute_poly_value(current_root, coefs, degs)
    z_eval_deriv = compute_deriv_poly(current_root, coefs, degs)

    if z_eval_deriv == 0.0:
        final_out[idx] = current_root
        return

    good_fraction = z_eval/z_eval_deriv

    new_sum_val = compute_fractional_val(initial_vals, idx, current_root)

    denominator = 1 - (good_fraction*new_sum_val)

    if denominator == 0.0:
        final_out[idx] = current_root
        return

    new_guess = current_root - (good_fraction/denominator)

    final_out[idx] = new_guess

@cuda.jit
def check_solutions(initial_guess, new_guess, error):
    idx = cuda.grid(1)

    if idx >= initial_guess.shape[0]:
        return

    diff = new_guess[idx] - initial_guess[idx]
    error[idx] = (diff.real ** 2 + diff.imag ** 2) ** 0.5

def EA_method(polies, initial_vals, coeffs, deg, threads_per_block, max_iter, max_error):
    vals_d   = cuda.to_device(initial_vals.astype(np.complex128))
    coeffs_d = cuda.to_device(coeffs.astype(np.complex128))
    degs_d   = cuda.to_device(deg.astype(np.float64))

    m = len(initial_vals)
    n_terms = len(coeffs)

    blocks = (m + threads_per_block - 1) // threads_per_block
    griddim   = (blocks,)
    blockdim  = (threads_per_block,)
 
    final_components = np.zeros(initial_vals.shape)
    final_gpu_comp = cuda.to_device(final_components)

    errors_comp = np.zeros(initial_vals.shape, dtype=np.complex128)
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
    pass

if __name__ == "__main__":
    main()