import numpy as np
from itertools import product
from math import gcd
from multiprocessing import Pool
from numba import njit

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

    lst_of_polynomials = [fill_free_variable(int_lst, general_form) for int_lst in product(all_free_values, repeat = n_free)]

    return lst_of_polynomials
            
def find_max_degree(poly):
    
    length_of_poly = len(poly)
    first_place = np.argmax(np.array(poly) != 0)

    degree = length_of_poly - first_place - 1

    return degree

@njit
def gcd_array(arr):

    result = arr[0]
    for i in range(1, len(arr)):
        a, b = result, arr[i]
        while b:
            a, b = b, a % b
        result = a
    return abs(result)

def root_finding(poly):

    if gcd(poly) != 1:
        return None

    length_of_poly = len(poly)
    first_place = np.argmax(np.array(poly) != 0)

    deg = length_of_poly - first_place - 1
    scale = 1/(2*poly[len(poly)- deg - 1])


    roots = np.roots(np.array(poly))
    lst_of_roots = [(r, deg, scale) for r in roots if r.imag != 0]
    return lst_of_roots

def compute_algebraic_starscape(input_lst, scale_val, n_free):
    
    all_combinatorial_forms = form_combinatoric_forms(input_lst, n_free, scale_val)

    with Pool(3) as p:
        lst_of_all_roots = p.map(root_finding, all_combinatorial_forms)

    return lst_of_all_roots

def main():
    input_form = [("v", 10),("v", 3),("v", 2), ("v", 1), ("v", 0)]

    out_starscape = compute_algebraic_starscape(input_form, scale_val=10, n_free=5)

if __name__ == "__main__":
    main()