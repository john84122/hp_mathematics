import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from numba import njit, prange, set_num_threads

@njit(parallel=True)
def compute_par(complex_values):
        
    N = 100
    k = 100

    c_const = -2.1+ (-1.35j)

    iter_lst = np.zeros(len(complex_values))

    for idx in prange(len(complex_values)):

        c = complex_values[idx]
        iter = 0
        while (iter < N) and (np.abs(c) < k):
            c = np.sin(c) + c_const
            iter += 1

        iter_lst[idx] = iter
    
    return iter_lst

def compute_parallel(lims_x, lims_y, n_pts = 1000):

    real_part = np.linspace(-lims_x, lims_x, n_pts)
    imag_part = np.linspace(-lims_y, lims_y, n_pts)

    X, Y = np.meshgrid(real_part, imag_part)

    complex_lattice = X + 1j*Y
    complex_values = complex_lattice.flatten()


    t = time.time()

    d_image = compute_par(complex_values)

    iter_vals = np.array(d_image).reshape(complex_lattice.shape)


    # Code to check whether the fractal was actually formed.
    #plt.imshow(iter_vals)
    #plt.savefig("sine_wave_cpu.png", bbox_inches='tight')
    #assert 1 == 0
    total = time.time() - t

    return total


def collect_times_for_varying_computations(l_xy_max, n_pts_max, save_fl):

    ## Need to set it up at the begging
    set_num_threads(32)

    dictionary_of_values = {
        "l_xy": [],
        "n_pts": [],
        "time": [],
    }

    for n_pts in tqdm(range(10, n_pts_max, 10), desc="Number of Points"):
        for l_x_y in range(1, l_xy_max):

            time = compute_parallel(lims_x = l_x_y, lims_y = l_x_y, n_pts = n_pts)
            dictionary_of_values["l_xy"].append(l_x_y)
            dictionary_of_values["n_pts"].append(n_pts)
            dictionary_of_values["time"].append(time)

    pandata = pd.DataFrame(dictionary_of_values)

    pandata.to_csv(save_fl)
    print()
    print("Done!")

    return pandata

def main():
    sv_file = "/home/jbauer/code/hp_mathematics/timing_results/cpu_p_32_new.csv"
    out = collect_times_for_varying_computations(100, 500, sv_file)

if __name__ == "__main__":
    main()