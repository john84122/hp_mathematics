'''
Code for NUMBA single GPU optimization for fractals.
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import math

from numba import cuda
from tqdm import tqdm

@cuda.jit(device=True)
def paralell_fractal_generation(real, imag, escape_radius, n_terms):

    c = real + (imag*1j)

    c_const = -2.1+ (-1.35j)

    iter_val = 0
    while (iter_val < n_terms) and ((c.real**2) + (c.imag**2) < escape_radius):
        c_re = math.sin(c.real)*math.cosh(c.imag)
        c_imag = math.cos(c.real)*math.sinh(c.imag)
        c = c_re + (c_imag*1j) + c_const
        iter_val += 1

    return iter_val

@cuda.jit
def compute_gpu_kernel(lims_x, lims_y, d_img, n_terms, escape_rad):
    '''
    I found a implementation that gives a good start. Adapted it to the code I have now.
    https://developer.nvidia.com/blog/numba-python-cuda-acceleration/
    '''

    height = d_img.shape[0]
    width = d_img.shape[1]

    pixel_size_x = (2 * lims_x) / width
    pixel_size_y = (2 * lims_y) / height

    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = -lims_x + x * pixel_size_x

        for y in range(startY, height, gridY):
            imag = -lims_y + y * pixel_size_y 
            iter_val = paralell_fractal_generation(real, imag, escape_rad, n_terms)
            d_img[y, x] = min(iter_val* 255//max(n_terms, 1), 255)

def gpu_computation(lims_x, lims_y, n_pts, escape_radius):

    d_image = np.zeros((n_pts, n_pts), dtype = np.uint8)

    blockdim = (16, 16)
    griddim  = (math.ceil(n_pts / blockdim[0]), math.ceil(n_pts / blockdim[1]))

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)

    d_image = cuda.to_device(d_image)

    start.record()
    compute_gpu_kernel[griddim, blockdim](lims_x, lims_y, d_image, n_pts, escape_radius) 
    end.record()
    end.synchronize()

    out = d_image.copy_to_host()

    dt = cuda.event_elapsed_time(start, end)

    return dt

def collect_times_for_varying_computations(l_xy, n_pts_max, save_fl, escape_radius):

    dictionary_of_values = {
        "l_xy": [],
        "n_pts": [],
        "time": [],
    }

    for n_pts in tqdm(range(10, n_pts_max, 10), desc="Number of Points"):

        time_elapsed = gpu_computation(lims_x = l_xy, lims_y = l_xy, n_pts = n_pts, escape_radius=escape_radius)
        dictionary_of_values["l_xy"].append(l_xy)
        dictionary_of_values["n_pts"].append(n_pts)
        dictionary_of_values["time"].append(time_elapsed)

    pandata = pd.DataFrame(dictionary_of_values)

    pandata.to_csv(save_fl)
    print()
    print("Done!")

    return pandata

def main():
    sv_file = "/home/jbauer/code/hp_mathematics/timing_results/fractal_runs/gpu_global_run_normal_gpu.csv"
    out = collect_times_for_varying_computations(100, 500, sv_file, 100, )

if __name__ == "__main__":
    main()