'''
A simple script for conducting this analysis.
'''

import numpy as np

# https://www.sciencedirect.com/science/article/pii/S1877750316304641#enun0035
def dk_operation(initial_value):
    
    for i in range(len(initial_value)):

def main(initial_values, epsilon_val = 1e-8):

    diff = 1000

    while diff > epsilon_val:
        new_values = dk_operation(initial_values)

        diff = np.max(np.abs(new_values - initial_values))
        initial_values = new_values

if __name__ == "__main__":
    main()