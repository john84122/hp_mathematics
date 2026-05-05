# Fractal and Starscape Generation

This project is intended to provide parallelized code in order to produce mathematical visualizations. These are:

1. Complex Fractals
2. Algebraic Starscapes
3. Neural NetworkBasked Fractals.

The hope is to understand how python can be used to speedup the production of these fractals.

## Code Base

There are two main areas where our code is in.

### Source Scripts

- This is where the timing scripts for fractal generation lie in. Therea re three main directories:


1. dnn_fractal - Directory with scripts for DNN Fractal generation.
2. fractal - Directory with scripts for complex fractal generation.
3. starscape - Directory with scripts for algebraic starscape generation.

- There be many scripts within these, but the three that are most important in each are the:

1. fractal_generation_"word".py - Scripts in the fractal generation directory that build neural network fractals.
2. fractal_"word".py - Scripts in the fractal generation directory that build fractals.
3. alg_starscape_"word".py - Scripts in the starscape directory that build starscapes.

- The "word" within the file names are either "cpu", "GPU", or "sequential". Each of these scripts key in whether a multi CPU, GPU, or sequential implementation is written for the generation of the object in these scripts.

- Other scripts are either helper functions, like root finding, or additional scripts that help with the generation.

### Jupyter Notebooks:

- These are scripts that help with visualizing speedups as well as timings.

## Save Files:

- All timed save files are found in the timing_results directory. In addition, some visuals are generated in the visuals file.