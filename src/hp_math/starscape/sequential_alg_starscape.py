
import numpy as np
import pandas as pd

from itertools import product
from math import gcd

import time

def convert_list_to_poly(poly_lst):
    '''
    Converts the a list form of a polynomial to a polynomial
    '''

    max_val = np.max([p_v[1] for p_v in poly_lst]) + 1

    base_poly = list(np.zeros(max_val, dtype=int))

    for comp_poly in poly_lst:
        base_poly[max_val - comp_poly[1] - 1] = comp_poly[0]
    
    return base_poly

def fill_free_variable(int_lst, general_form):
    
    idx_for_ints = 0
    val_lst = []
    for idx, k in enumerate(general_form):
        if k == "v":
            val_lst.append(int_lst[idx_for_ints])
            idx_for_ints += 1
        else:
            val_lst.append(k)


    return val_lst

def form_combinatoric_forms(input_lst, n_free, scale_val):

    all_free_values = list(range(-scale_val, scale_val + 1))

    general_form = convert_list_to_poly(input_lst)

    lst_of_polynomials = [fill_free_variable(comb_lst, general_form) for comb_lst in product(all_free_values, repeat = n_free)]

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

def compute_algebraic_starscape(input_lst, scale_val, n_free):
    
    all_combinatorial_forms = form_combinatoric_forms(input_lst, n_free, scale_val)

    lst_of_all_roots = []

    for poly_form in all_combinatorial_forms:

        root_pairs = root_finding(poly_form)

        lst_of_all_roots += root_pairs

    return lst_of_all_roots


def compute_poly_form_all_free(deg):

    poly_form = []

    for k in range(deg + 1):
        poly_form.append(("v", k))

    return poly_form

def timing_algebraic_starscape(sv_fl):
    dictionary = {
        "timing": [],
        "scale_val": [],
        "degree": [],
        "number_of_roots_found": []
    }

    relevant_deg = [3, 5, 10, 15] + list(range(100, 1000, 100))

    for deg in relevant_deg:
        
        poly_form = compute_poly_form_all_free(deg)
        
        for scale in [2]:
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

def main():
    sv_file = "/home/jbauer/code/hp_mathematics/timing_results/algebraic_starscape_runs/sequential_alg.csv"

    timing_algebraic_starscape(sv_file)

if __name__ == "__main__":
    main()