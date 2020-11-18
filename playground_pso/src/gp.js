"use strict";
/* A simple genetic algorithm and the interaction with the PSO/NN via the fitness function and the Node class */
// Constant parameters
exports.__esModule = true;
exports.oneStepGP = exports.buildPop = exports.Population = exports.Individual = exports.evaluatePopFitnessNN = exports.getFitnessNN = exports.evaluatePopFitnessPSO = exports.getFitnessPSO = void 0;
var MAXLAYERSIZE = 2; // This refers only to the hidden layers
var LAYERS = 2; // This refers only to the hidden layers
var NACTIVATIONS = 6;
var LAYERPADDING = 1;
var MUTPROB = 0.01;
var CROSSPROB = 0.7;
var ELITISM = true;
var INPUTDIM = 2;
var MAX_COST = 1;
var N_ITER = 10;
var POPSIZE = 5;
var nn = require("./nn");
var aux = require("./aux");
var pso = require("./pso");
// Auxiliary functions
function mapToMatrix(arrayIndex) {
    if (arrayIndex % (LAYERS * MAXLAYERSIZE + 1) == 0) {
        //this corresponds to the case of an activation index
        console.log("Index corresponds to activation key");
        return null;
    }
    else {
        var subIndex = arrayIndex % (LAYERS * MAXLAYERSIZE + 1);
        var rowIndex = null;
        var colIndex = null;
        if (subIndex % LAYERS === 0) {
            colIndex = Math.floor(subIndex / LAYERS) - 1;
            rowIndex = LAYERS - 1;
        }
        else {
            colIndex = Math.floor(subIndex / LAYERS);
            rowIndex = LAYERS - (LAYERS - subIndex % LAYERS) - 1;
        }
        return [rowIndex, colIndex];
    }
}
function verifyLayer(index1, index2) {
    var coord1 = mapToMatrix(index1);
    var coord2 = mapToMatrix(index2);
    var layer1 = coord1[1];
    var layer2 = coord2[1];
    return (layer1 - layer2) >= LAYERPADDING;
}
function mapActivationKey(key) {
    var activationArray = [nn.Activations.RELU, nn.Activations.SIGMOID, nn.Activations.LINEAR,
        nn.Activations.SIN, nn.Activations.RBF, nn.Activations.TANH];
    return activationArray[key];
}
function getFitnessPSO(individual, n_iter, trainData, valData) {
    individual.translate(nn.Activations.TANH, nn.Activations.SIGMOID, aux.constructInputIds(), null);
    var network = individual.phenotype;
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
        fitness_values.push(getFitnessPSO(i, n_iter, trainData, valData));
    }
    return fitness_values;
}
exports.evaluatePopFitnessPSO = evaluatePopFitnessPSO;
function getFitnessNN(individual, n_iter, trainData, valData) {
    individual.translate(nn.Activations.TANH, nn.Activations.SIGMOID, aux.constructInputIds(), null);
    var network = individual.phenotype;
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
        fitness_values.push(getFitnessNN(i, n_iter, trainData, valData));
    }
    return fitness_values;
}
exports.evaluatePopFitnessNN = evaluatePopFitnessNN;
var Individual = /** @class */ (function () {
    function Individual() {
        this.config = [];
        for (var current_node = 1; current_node <= MAXLAYERSIZE * LAYERS; current_node++) {
            this.config.push(aux.getRandomInt(0, NACTIVATIONS));
            for (var current_conn = 1; current_conn < MAXLAYERSIZE * LAYERS; current_conn++) {
                var currentConIndex = (current_node - 1) * (MAXLAYERSIZE * LAYERS + 1) + current_conn;
                if (verifyLayer(current_node, currentConIndex)) {
                    this.config.push(Math.round(Math.random()));
                }
                else {
                    this.config.push(0); // We only allow connections between nodes up to a LAYERPADDING difference of Layers
                }
            }
        }
    }
    Individual.prototype.mutate = function () {
        for (var _i = 0, _a = this.config; _i < _a.length; _i++) {
            var chromosome = _a[_i];
            if (Math.random() < MUTPROB) {
                if (chromosome % (LAYERS * MAXLAYERSIZE + 1) == 0) { // Check if it corresponds to an activation node
                    this.config[chromosome] = aux.getRandomInt(0, NACTIVATIONS);
                }
                else {
                    if (this.config[chromosome] == 0) {
                        this.config[chromosome] = 1;
                    }
                    else {
                        this.config[chromosome] = 0;
                    }
                }
            }
        }
    };
    // Largely based on nn.buildNetwork but allowing different activation function for each unit
    Individual.prototype.translate = function (inputActivation, outputActivation, inputIds, regularization) {
        var numLayers = LAYERS + 2;
        var id = 1;
        /** List of layers, with each layer being a list of nodes. */
        var network = [];
        for (var layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            var isOutputLayer = layerIdx === numLayers - 1;
            var isInputLayer = layerIdx === 0;
            var isFirstHidden = layerIdx >= LAYERPADDING;
            var currentLayer = [];
            network.push(currentLayer);
            var nodeId = '';
            var numNodes = 0;
            if (isInputLayer) {
                numNodes = INPUTDIM;
                for (var i = 0; i < numNodes; i++) {
                    nodeId = inputIds[i];
                    var node = new nn.Node(nodeId, inputActivation);
                    currentLayer.push(node);
                }
            }
            else if (isOutputLayer) {
                nodeId = id.toString();
                id++;
                var node = new nn.Node(nodeId, outputActivation);
                currentLayer.push(node);
                for (var j = 0; j < network[layerIdx - 1].length; j++) {
                    // We make the ouput layer fully connected
                    var prevNode = network[layerIdx - 1][j];
                    var link = new nn.Link(prevNode, node, regularization);
                    prevNode.outputs.push(link);
                    node.inputLinks.push(link);
                }
            }
            else if (isFirstHidden) {
                numNodes = MAXLAYERSIZE;
                for (var i = 0; i < numNodes; i++) {
                    nodeId = id.toString();
                    id++;
                    var activationKey = this.config[i * (MAXLAYERSIZE * LAYERS + 1)];
                    var activation = mapActivationKey(activationKey);
                    var node = new nn.Node(nodeId, activation);
                    currentLayer.push(node);
                    for (var j = 0; j < network[layerIdx - 1].length; j++) {
                        // For the first hidden layer ignore genotype and make it fully connected
                        var prevNode = network[layerIdx - 1][j];
                        var link = new nn.Link(prevNode, node, regularization);
                        prevNode.outputs.push(link);
                        node.inputLinks.push(link);
                    }
                }
            }
            else {
                numNodes = MAXLAYERSIZE;
                for (var i = 0; i < numNodes; i++) {
                    nodeId = id.toString();
                    id++;
                    var nodeArrayidx = ((layerIdx - 1) * MAXLAYERSIZE + i) * (MAXLAYERSIZE * LAYERS + 1);
                    var activationKey = this.config[nodeArrayidx];
                    var activation = mapActivationKey(activationKey);
                    var node = new nn.Node(nodeId, activation);
                    currentLayer.push(node);
                    for (var k = LAYERPADDING; k > 0; k--) {
                        for (var j = 0; j < network[layerIdx - k].length; j++) {
                            var PrevLayer = layerIdx - k - 1;
                            var arrayIndex = nodeArrayidx + (PrevLayer * MAXLAYERSIZE) + (j + 1);
                            var isDead = this.config[arrayIndex] === 1;
                            if (!isDead) {
                                var prevNode = network[layerIdx - k][j];
                                var link = new nn.Link(prevNode, node, regularization);
                                prevNode.outputs.push(link);
                                node.inputLinks.push(link);
                            }
                        }
                    }
                }
            }
        }
        this.phenotype = network;
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
                var crossOverPoint = aux.getRandomInt(0, parent1.config.length - 1);
                for (var chromosome = crossOverPoint; chromosome < parent1.config.length; chromosome++) {
                    var temp1 = parent1.config[chromosome];
                    var temp2 = parent2.config[chromosome];
                    parent1.config[chromosome] = temp2;
                    parent2.config[chromosome] = temp1;
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
function buildPop(lowLevelAlgo) {
    var pop = new Population;
    pop.totalFit = 0;
    pop.averageFit = 0;
    pop.bestFit = 0;
    pop.lowLevelAlgo = lowLevelAlgo;
    for (var i = 0; i < POPSIZE; i++) {
        var member = new Individual();
        pop.individuals.push(member);
    }
    return pop;
}
exports.buildPop = buildPop;
function oneStepGP(pop, trainData, valData, withCrossOver) {
    pop.selection(trainData, valData);
    if (withCrossOver) {
        pop.crossover();
    }
    pop.mutation();
}
exports.oneStepGP = oneStepGP;
