'''
Sequential code for fractal generation.
'''

import time
import numpy as np
import pandas as pd

from tqdm import tqdm

def compute_sequential(lims_x, lims_y, n_pts = 1000):

    real_part = np.linspace(-lims_x, lims_x, n_pts)
    imag_part = np.linspace(-lims_y, lims_y, n_pts)

    X, Y = np.meshgrid(real_part, imag_part)

    complex_lattice = X + 1j*Y
    complex_values = complex_lattice.flatten()
    
    N = 100
    k = 100

    c_const = -2.1+ (-1.35j)

    iter_lst = []

    for c in complex_values:
        iter = 0
        while (iter < N) and (np.abs(c) < k):
            c = np.sin(c) + c_const
            iter += 1

        iter_lst.append(iter)

    return iter_lst


def time_computation(**kwargs):
    l_x = kwargs.get("lims_x", 1000)
    l_y = kwargs.get("lims_y", 1000)
    n_pts = kwargs.get("n_pts", 100)

    t = time.time()

    _ = compute_sequential(l_x, l_y, n_pts)

    total = time.time() - t

    return total

def collect_times_for_varying_computations(n_pts_max, save_fl):

    dictionary_of_values = {
        "l_xy": [],
        "n_pts": [],
        "time": [],
    }

    for n_pts in tqdm(range(10, n_pts_max, 10), desc="Number of Points"):
        l_x_y = 1

        time = time_computation(lims_x = l_x_y, lims_y = l_x_y, n_pts = n_pts)
        dictionary_of_values["l_xy"].append(l_x_y)
        dictionary_of_values["n_pts"].append(n_pts)
        dictionary_of_values["time"].append(time)

    pandata = pd.DataFrame(dictionary_of_values)

    pandata.to_csv(save_fl)
    print()
    print("Done!")

    return pandata

def main():
    sv_file = "/Users/johannesbauer/Documents/Coding/hp_mathematics/timing_results/fractal_runs/sequential_run_range_1.csv"
    out = collect_times_for_varying_computations(500, sv_file)

if __name__ == "__main__":
    main()