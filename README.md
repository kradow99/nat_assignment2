# Natural Computing 2020/2021: Assignment 2

## Requirements

Install d3 to package.json

`npm install d3 --save`

and definition file for node

`npm install @types/node`

## Task 1: PSO

### Running the PSO

To run the PSO weight updater on the neural network, first compile:

`tsc main.ts`

then, run:

`node main.js`

The output will be one line for each 50 iterations, where is shown the global best fitness, training loss and test loss found so far.

It is possible to tune the number of iterations `n_iter` of the PSO and network architecture `networkShape`. The first element of `networkShape` must coincide with the number of input features, which can be found in `plauground_pso/src/aux.ts`. Please uncomment there the InputFeatures in `INPUTS` you want to use.

### PSO Parameters Optimisation

To run the PSO parameters optimisator on the neural network, first compile:

`tsc pso_param_opt.ts`

then, run:

`node pso_param_opt.js`

The output are 3 lines for each parameters combination. The first one, is the parameters themselves (`omega`, `alpha1` and `alpha2`) The second one is the fitness, training loss and test loss for the best execution of the PSO achieved (by default, the script runs `n = 20` PSOs with each parameter combination). The third line is the same as the second one but with the mean values of all the `n` runs.

To see a graphical representation of the results in a heatmap format, please save the output in a file:

`node pso_param_opt.js > output.dat`

and then run:

`python3 pso_param_opt.py`

where `output.dat` is needed to plot the alphas with their respective best and mean performance achieved.

Note that for good relaible results, this procces is time expensive, since a lot of iterations and parameter combinations must be done. These number of iterations and range of the parameters can be tuned though, as well as the network architecture (the same way as when runing the PSO)

## Task 2: Genetic Algorithm

## Task 3: Genetic Programming
