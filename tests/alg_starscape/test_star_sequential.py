
from hp_math.starscape.sequential_alg_starscape import *

def test_finding_max_deg():
    poly_1 = [0, 12, 342, 2]
    deg_1 = find_max_degree(poly_1)

    assert deg_1 == 2, f"The found degree is {deg_1}, but it should be 2."

    poly_2 = [10, 12, 342, 2, 23, 54]
    deg_2 = find_max_degree(poly_2)

    assert deg_2 == 5, f"The found degree is {deg_2}, but it should be 5."

    poly_3 = [0, 0, 0, 0, 0, 2]
    deg_3 = find_max_degree(poly_3)

    assert deg_3 == 0, f"The found degree is {deg_3}, but it should be 0."



def test_root_finding():

    roots = root_finding([0, 23, 0 , 0])

    assert len(roots) == 0, f"The found roots are {roots}, but there should be no roots."

    roots_1 = root_finding([0, 0, 0 , 0])

    assert len(roots_1) == 0, f"The found roots are {roots_1}, but there should be no roots."

    roots_3 = root_finding([0, 1, 2 , 0])

    assert len(roots_3) == 0, f"The found roots are {roots_3}, but there should be no roots."

    roots_4 = root_finding([0, 1, 0 , 1])

    assert len(roots_4) == 2, f"The found roots are {roots_4}, but there should be no roots."

def test_fill_free_variable():
    k = [5, 23, 5, 1]
    gen_form = ["v", 32, 1, "v", "v", 3, "v"]

    correct_form = [5, 32, 1, 23, 5, 3, 1]

    out = fill_free_variable(k, gen_form)

    for x, y in zip(out, correct_form):
        assert x == y, f"The output {x} should be equal to {y}."

def test_fill_poly():

    new_input = [("v", 5), ("v", 2), (4, 1), ("v", 0)]
    out = convert_list_to_poly(new_input)

    answer = ["v", 0, 0, "v", 4, "v"]

    for k, true_k in zip(out, answer):
        assert k == true_k, f"The output {k} should be equal to {true_k}."

def test_compute_alg_starscape():
    new_input = [("v", 5), ("v", 2), (4, 1), ("v", 0)]
    out = compute_algebraic_starscape(new_input, 10, 3)

    length_of_out = len(out)

    assert length_of_out != 0, f"You are wrong, the length of output is {length_of_out}."

def test_form_combinatoric_forms():

    new_input = [("v", 5), ("v", 2), (4, 1), ("v", 0)]
    out = form_combinatoric_forms(new_input, 3, 10)

    idx_lst = np.random.randint(0, len(out), 15)

    for idx in idx_lst:
        new_data = out[idx]
        new_data_1 = out[idx + 1]

        diff = np.sum(np.abs(np.array(new_data) - np.array(new_data_1)))

        assert diff > 0.000001, f"There should not be an equality, but both values are {new_data}."