import numpy as np
import pandas as pd

from itertools import product
from math import gcd
from hp_math.starscape.ea_root_finding_method import EA_method
#from hp_math.starscape.dk_root_finding_method import dk_method


import time

from multiprocessing import Pool

from tqdm import tqdm

import cupy as cp

def convert_list_to_poly(poly_lst):
    '''
    Converts the a list form of a polynomial to a polynomial
    '''

    max_val = np.max([p_v[1] for p_v in poly_lst]) + 1

    base_poly = list(np.zeros(max_val, dtype=int))

    for comp_poly in poly_lst:
        base_poly[max_val - comp_poly[1] - 1] = comp_poly[0]
    
    return base_poly

def fill_free_variable(int_lst, input_lst):
    
    idx_for_ints = 0
    coef_lst = []
    deg_lst = []
    for k in input_lst:
        if k[0] == "v":
            coef_lst.append(int_lst[idx_for_ints])
            deg_lst.append(k[1])
            idx_for_ints += 1
        else:
            coef_lst.append(k[0])
            deg_lst.append(k[1])


    return coef_lst,deg_lst

def form_combinatoric_forms(input_lst, n_free, scale_val):

    all_free_values = list(range(-scale_val, scale_val + 1))

    lst_of_polynomials = [fill_free_variable(int_lst, input_lst) for int_lst in product(all_free_values, repeat = n_free)]

    return lst_of_polynomials
            
def find_max_degree(poly):
    
    length_of_poly = len(poly)
    first_place = np.argmax(np.array(poly) != 0)

    degree = length_of_poly - first_place - 1

    return degree
    

def root_finding(poly):

    if gcd(*poly) != 1:
        return []

    deg = find_max_degree(poly)
    scale = 1/(2*poly[len(poly)- deg - 1])


    roots = np.roots(poly)
    lst_of_roots = [(r, deg, scale) for r in roots if r.imag != 0]
    return lst_of_roots

def pooling_function(input_poly):
    coeff_lst, deg_lst = input_poly

    coeff_arr = np.array(coeff_lst)
    deg_arr = np.array(deg_lst)

    n_roots = np.max(deg_lst)
    angles  = np.linspace(0, 2 * np.pi, n_roots + 1, endpoint=False)[1:]
    initial = np.exp(1j * angles) * 2
    root_pairs = EA_method(initial, coeff_arr, deg_arr, 64, 300, 1e-8)

    return root_pairs

def compute_algebraic_starscape(input_lst, scale_val, n_free):
    
    all_combinatorial_forms = form_combinatoric_forms(input_lst, n_free, scale_val)

    lst_of_all_roots = []


    with Pool(4) as p:
        lst_of_all_roots = p.map(pooling_function, all_combinatorial_forms)

    return lst_of_all_roots

def compute_poly_form_all_free(deg):

    poly_form = []

    poly_form += [("v", deg), ("v", deg//2), (1, 0)]

    #for k in range(deg + 1):
    #    poly_form.append(("v", k))

    return poly_form

def timing_algebraic_starscape(sv_fl):
    dictionary = {
        "timing": [],
        "scale_val": [],
        "degree": [],
        "number_of_roots_found": []
    }

    relevant_deg = [100, 500, 1000]#list(range(2, 5))
    for deg in relevant_deg:
        
        poly_form = compute_poly_form_all_free(deg)
        
        for scale in [1]:
            time_start = time.time()

            rts = compute_algebraic_starscape(poly_form, scale, deg+1)
            time_end = time.time() - time_start

            dictionary["timing"].append(time_end)
            dictionary["scale_val"].append(scale)
            dictionary["degree"].append(deg)
            dictionary["number_of_roots_found"].append(len(rts))

            print(f"Done with scale: {scale}.")
            print()

    pd_of_res = pd.DataFrame(dictionary)

    pd_of_res.to_csv(sv_fl)

    print("Saved and Done")

def timing_algebraic_starscape_root_finding(sv_fl):
    dictionary = {
        "timing": [],
        "degree": [],
        "number_of_roots_found": []
    }

    relevant_deg = list(range(10, 1500, 10))
    for deg in tqdm(relevant_deg):


        time_start = time.time()

        rts = pooling_function((list(np.ones(deg+1, dtype=int)), list(range(deg+1))))
        time_end = time.time() - time_start

        dictionary["timing"].append(time_end)
        dictionary["degree"].append(deg)
        dictionary["number_of_roots_found"].append(len(rts))

    pd_of_res = pd.DataFrame(dictionary)

    pd_of_res.to_csv(sv_fl)

    print("Saved and Done")

def main():
    sv_file = "/home/jbauer/code/hp_mathematics/timing_results/algebraic_starscape_runs/gpu_alg_4_dk_large_rts.csv"

    timing_algebraic_starscape_root_finding(sv_file)

if __name__ == "__main__":
    main()