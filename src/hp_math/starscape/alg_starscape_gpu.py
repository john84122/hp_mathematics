import numpy as np
from itertools import product
from math import gcd
from hp_math.starscape.ea_root_finding_method import EA_method

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
    initial = np.exp(1j * angles) * 0.9
    root_pairs = EA_method(initial, coeff_arr, deg_arr, 64, 200, 1e-8)

    return root_pairs

def compute_algebraic_starscape(input_lst, scale_val, n_free):
    
    all_combinatorial_forms = form_combinatoric_forms(input_lst, n_free, scale_val)

    lst_of_all_roots = []


    with Pool(3) as p:
        lst_of_all_roots = p.map(pooling_function, all_combinatorial_forms)

    return lst_of_all_roots

def main():
    input_form = [("v", 3),("v", 2), ("v", 1), ("v", 0)]

    out = time.time()

    out_starscape = compute_algebraic_starscape(input_form, scale_val=3, n_free=4)

    print("Timing:", time.time() - out)

if __name__ == "__main__":
    main()