import os

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import math

from numba import cuda
from tqdm import tqdm

@cuda.jit(device=True)
def paralell_fractal_generation(real, imag):

    c = real + (imag*1j)

    N = 100
    k = 100

    c_const = -2.1+ (-1.35j)

    iter = 0
    while (iter < N) and ((c.real**2) + (c.imag**2) < k):
        c_re = math.sin(c.real)*math.cosh(c.imag)
        c_imag = math.cos(c.real)*math.sinh(c.imag)
        c = c_re + (c_imag*1j) + c_const
        iter += 1

    return iter

@cuda.jit
def compute_gpu_kernel(lims_x, lims_y, d_img = None):
    '''
    I found a implementation of this that is far better than what I wrote
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
            d_img[y, x] = paralell_fractal_generation(real, imag)


def gpu_computation(lims_x, lims_y, n_pts):

    d_image = np.zeros((n_pts, n_pts), dtype = np.uint8)

    blockdim = (16, 16)
    griddim  = (math.ceil(n_pts / blockdim[0]), math.ceil(n_pts / blockdim[1]))

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)

    d_image = cuda.to_device(d_image)
    start.record()
    compute_gpu_kernel[griddim, blockdim](lims_x, lims_y, d_image) 
    cuda.synchronize()
    end.record()
    out = d_image.copy_to_host()

    # For Checking material
    #plt.imshow(out.astype(float))
    #print(np.unique(out))
    #plt.savefig("sine_wave_new.png", bbox_inches='tight')
    #plt.show()
    #plt.close()
    #assert 1 == 0

    dt = cuda.event_elapsed_time(start, end)

    return dt

def collect_times_for_varying_computations(l_xy_max, n_pts_max, save_fl):

    dictionary_of_values = {
        "l_xy": [],
        "n_pts": [],
        "time": [],
    }

    for n_pts in tqdm(range(10, n_pts_max, 10), desc="Number of Points"):
        for l_x_y in range(1, l_xy_max):

            time = gpu_computation(lims_x = l_xy_max, lims_y = l_xy_max, n_pts = n_pts)
            dictionary_of_values["l_xy"].append(l_x_y)
            dictionary_of_values["n_pts"].append(n_pts)
            dictionary_of_values["time"].append(time)

    pandata = pd.DataFrame(dictionary_of_values)

    pandata.to_csv(save_fl)
    print()
    print("Done!")

    return pandata

def main():
    sv_file = "/home/jbauer/code/hp_mathematics/timing_results/fractal_runs/gpu_global_run_normal_64_8.csv"
    out = collect_times_for_varying_computations(100, 500, sv_file)

if __name__ == "__main__":
    main()