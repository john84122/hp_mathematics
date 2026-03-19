import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit, prange

from multiprocessing import Pool

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

    _ = compute_par(complex_values)

    total = time.time() - t

    return total


def collect_times_for_varying_computations(l_xy_max, n_pts_max, save_fl):

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
    sv_file = "/Users/johannesbauer/Documents/Coding/hp_mathematics/timing_results/fractal_runs/cpu_p_4_run.csv"
    out = collect_times_for_varying_computations(100, 500, sv_file)

if __name__ == "__main__":
    main()