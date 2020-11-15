"use strict";
exports.__esModule = true;
exports.oneStepGA = exports.buildPop = exports.Population = exports.Individual = exports.evaluatePopFitnessNN = exports.getFitnessNN = exports.evaluatePopFitnessPSO = exports.getFitnessPSO = exports.buildNNFromGA = void 0;
/* A simple genetic algorithm and the interaction with the PSO/NN via the fitness function */
var POPSIZE = 5;
var CROSSPROB = 0.7;
var MUTPROB = 0.01;
var ELITISM = true;
var N_ITER = 100; // this is for the low leveel algorithm not for the GA
var MAX_COST = 1;
var INPUTDIM = 2;
var pso = require("./pso");
var nn = require("./nn");
var aux = require("./aux");
// Auxiliar functions
function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}
function buildNNFromGA(config) {
    var activation = nn.Activations.TANH;
    var outputActivation = nn.Activations.SIGMOID;
    var regularization = null;
    var shape = [INPUTDIM].concat(config).concat([1]); // input and output dimensions are fixed
    var network = nn.buildNetwork(shape, activation, outputActivation, regularization, aux.constructInputIds());
    return network;
}
exports.buildNNFromGA = buildNNFromGA;
function getFitnessPSO(config, n_iter, trainData, valData) {
    var network = buildNNFromGA(config);
    var swarm = pso.buildSwarm(nn.countWeights(network));
    for (var i = 0; i <= n_iter; i++) {
        swarm.updateSwarm(network, trainData);
    }
    // Compute the loss.
    var valLoss = aux.getLoss(network, valData);
    return MAX_COST - valLoss;
}
exports.getFitnessPSO = getFitnessPSO;
function evaluatePopFitnessPSO(individuals, n_iter, trainData, valData) {
    var fitness_values = [];
    for (var _i = 0, individuals_1 = individuals; _i < individuals_1.length; _i++) {
        var i = individuals_1[_i];
        fitness_values.push(getFitnessPSO(i.config, n_iter, trainData, valData));
    }
    return fitness_values;
}
exports.evaluatePopFitnessPSO = evaluatePopFitnessPSO;
function getFitnessNN(config, n_iter, trainData, valData) {
    var network = buildNNFromGA(config);
    for (var i = 0; i <= n_iter; i++) {
        aux.oneStepNN(trainData, valData, network);
    }
    var valLoss = aux.getLoss(network, valData);
    return MAX_COST - valLoss;
}
exports.getFitnessNN = getFitnessNN;
function evaluatePopFitnessNN(individuals, n_iter, trainData, valData) {
    var fitness_values = [];
    for (var _i = 0, individuals_2 = individuals; _i < individuals_2.length; _i++) {
        var i = individuals_2[_i];
        fitness_values.push(getFitnessNN(i.config, n_iter, trainData, valData));
    }
    return fitness_values;
}
exports.evaluatePopFitnessNN = evaluatePopFitnessNN;
// Class for the individual of the population
var Individual = /** @class */ (function () {
    function Individual(dim, min_ranges, max_ranges) {
        this.config = [];
        this.min_ranges = [];
        this.max_ranges = [];
        this.dim = dim;
        this.min_ranges = min_ranges;
        this.max_ranges = max_ranges;
        for (var i = 0; i < this.dim; i++) {
            this.config.push(getRandomInt(this.min_ranges[i], this.max_ranges[i]));
        }
    }
    Individual.prototype.mutate = function () {
        for (var chromosome in this.config) {
            if (Math.random() < MUTPROB) {
                this.config[chromosome] = getRandomInt(this.min_ranges[chromosome], this.max_ranges[chromosome]);
            }
        }
    };
    return Individual;
}());
exports.Individual = Individual;
var Population = /** @class */ (function () {
    function Population() {
        this.individuals = [];
    }
    Population.prototype.selection = function (trainData, valData) {
        var _this = this;
        var intermediatePop = [];
        var popFitness = [];
        if (this.lowLevelAlgo === 'pso') {
            popFitness = evaluatePopFitnessPSO(this.individuals, N_ITER, trainData, valData);
        }
        else if (this.lowLevelAlgo === 'gd') {
            popFitness = evaluatePopFitnessNN(this.individuals, N_ITER, trainData, valData);
        }
        else {
            popFitness = null;
            console.log("Low level algorithm not recognized");
        }
        var tempPop = this.individuals; // temporary array used only for sorting
        this.individuals.sort(function (a, b) {
            var fitness_a = popFitness[tempPop.indexOf(a)];
            var fitness_b = popFitness[tempPop.indexOf(b)];
            if (fitness_a > fitness_b) {
                return -1;
            }
            if (fitness_b > fitness_a) {
                return 1;
            }
            return 0;
        });
        popFitness.sort(function (a, b) {
            if (a > b) {
                return -1;
            }
            if (b > a) {
                return 1;
            }
            return 0;
        });
        this.bestFit = popFitness[0];
        this.totalFit = popFitness.reduce(function (a, b) { return a + b; }, 0);
        this.averageFit = this.totalFit / this.individuals.length;
        if (ELITISM) {
            intermediatePop.push(this.individuals[0]);
        }
        var relativeFitness = popFitness.map(function (x) { return x / _this.averageFit; });
        var i = 0;
        while (intermediatePop.length < this.individuals.length) {
            if (relativeFitness[i] >= 1) {
                intermediatePop.push(this.individuals[i]);
                relativeFitness[i] = relativeFitness[i] - 1;
            }
            else if (relativeFitness[i] > 0) {
                if (Math.random() < relativeFitness[i]) {
                    intermediatePop.push(this.individuals[i]);
                }
            }
        }
        this.individuals = intermediatePop;
    };
    Population.prototype.mutation = function () {
        for (var _i = 0, _a = this.individuals; _i < _a.length; _i++) {
            var member = _a[_i];
            member.mutate();
        }
    };
    Population.prototype.crossover = function () {
        var intermediatePop = [];
        var midPoint = this.individuals.length / 2;
        var parentPool = this.individuals.slice(midPoint - 1, POPSIZE);
        for (var _i = 0, _a = this.individuals.slice(0, midPoint); _i < _a.length; _i++) {
            var parent1 = _a[_i];
            var parent2 = parentPool[Math.floor(Math.random() * parentPool.length)];
            if (Math.random() < CROSSPROB) {
                var crossOverPoint = getRandomInt(0, parent1.dim);
                for (var chromosome = crossOverPoint; chromosome <= parent1.dim; chromosome++) {
                    var temp1 = parent1[chromosome];
                    var temp2 = parent2[chromosome];
                    parent1[chromosome] = temp2;
                    parent2[chromosome] = temp1;
                }
            }
            intermediatePop.push(parent1);
            intermediatePop.push(parent2);
            parentPool.splice(parentPool.indexOf(parent2), 1);
        }
        this.individuals = intermediatePop;
    };
    Population.prototype.getBestConfig = function (trainData, valData) {
        var popFitness = [];
        if (this.lowLevelAlgo === 'pso') {
            popFitness = evaluatePopFitnessPSO(this.individuals, N_ITER, trainData, valData);
        }
        else if (this.lowLevelAlgo === 'gd') {
            popFitness = evaluatePopFitnessNN(this.individuals, N_ITER, trainData, valData);
        }
        else {
            popFitness = null;
            console.log("Low level algorithm not recognized");
        }
        var tempPop = this.individuals; // temporary array used only for sorting
        this.individuals.sort(function (a, b) {
            var fitness_a = popFitness[tempPop.indexOf(a)];
            var fitness_b = popFitness[tempPop.indexOf(b)];
            if (fitness_a > fitness_b) {
                return -1;
            }
            if (fitness_b > fitness_a) {
                return 1;
            }
            return 0;
        });
        var bestIndividual = this.individuals[0];
        console.log(popFitness);
        console.log("Fitness: ", popFitness[tempPop.indexOf(bestIndividual)]);
        var shape = [INPUTDIM].concat(bestIndividual.config).concat([1]);
        return shape;
    };
    return Population;
}());
exports.Population = Population;
function buildPop(dim, min_ranges, max_ranges, lowLevelAlgo) {
    var pop = new Population;
    pop.totalFit = 0;
    pop.averageFit = 0;
    pop.bestFit = 0;
    pop.lowLevelAlgo = lowLevelAlgo;
    for (var i = 0; i < POPSIZE; i++) {
        var member = new Individual(dim, min_ranges, max_ranges);
        pop.individuals.push(member);
    }
    return pop;
}
exports.buildPop = buildPop;
function oneStepGA(pop, trainData, valData) {
    pop.selection(trainData, valData);
    pop.crossover();
    pop.mutation();
}
exports.oneStepGA = oneStepGA;
