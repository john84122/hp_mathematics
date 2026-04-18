
from hp_math.starscape.sequential_alg_starscape import *

def test_finding_max_root():
    poly_1 = [0, 12, 342, 2]
    deg_1 = find_max_degree(poly_1)

    assert deg_1 == 2, f"The found degree is {deg_1}, but it should be 2."

    poly_2 = [10, 12, 342, 2, 23, 54]
    deg_2 = find_max_degree(poly_2)

    assert deg_2 == 5, f"The found degree is {deg_2}, but it should be 5."

    poly_3 = [0, 0, 0, 0, 0, 2]
    deg_3 = find_max_degree(poly_3)

    assert deg_3 == 0, f"The found degree is {deg_3}, but it should be 0."