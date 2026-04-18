import numpy as np
from itertools import combinations
from math import gcd

def convert_list_to_poly(poly_lst):
    '''
    Converts the a list form of a polynomial to a polynomial
    '''

    max_val = np.max([p_v[1] for p_v in poly_lst])

    base_poly = np.zeros(max_val)

    for comp_poly in poly_lst:
        base_poly[comp_poly[1]] = comp_poly[0]
    
    return base_poly

def fill_free_variable(int_lst, general_form):
    
    idx_for_ints = 0
    for idx, k in enumerate(general_form):
        if k == "v":
            general_form[idx] = int_lst[idx_for_ints]

            idx_for_ints += 1

    return general_form

def form_combinatoric_forms(input_lst, n_free, scale_val):

    all_combinatorial_types = []

    all_free_values = list(range(-scale_val, scale_val + 1))

    general_form = convert_list_to_poly(input_lst)

    lst_of_polynomials = [fill_free_variable(int_lst, general_form) for int_lst in combinations(general_form, n_free)]

    return lst_of_polynomials
            
def find_max_degree(poly):
    
    length_of_poly = len(poly)
    first_place = np.argmax(poly)

    degree = length_of_poly - first_place

    return degree

def root_finding(poly):
    
    if gcd(poly) == 1:
        return []

    deg = find_max_degree(poly)

    roots = np.roots(poly)
    lst_of_roots = [(r, deg) for r in roots if r.imag != 0]
    return lst_of_roots

def compute_algebraic_starscape(input_lst, scale_val, n_free):
    
    all_combinatorial_forms = form_combinatoric_forms(input_lst, n_free, scale_val)

    lst_of_all_roots = []

    for poly_form in all_combinatorial_forms:
        root_pairs = root_finding(poly_form)

        lst_of_all_roots += root_pairs
    return lst_of_all_roots

def main():
    pass

if __name__ == "__main__":
    main()