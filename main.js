"use strict";
exports.__esModule = true;
var ga = require("./playground_pso/src/ga");
var aux = require("./playground_pso/src/aux");
var dataset = require("./playground_pso/src/dataset");
// Constant Parameters
var NUM_SAMPLES_CLASSIFY = 500;
var percTrainData = 0.50;
// TASK1: PSO
var data = dataset.classifySpiralData(NUM_SAMPLES_CLASSIFY, 0);
dataset.shuffle(data);
// Split into train and test data.
var splitIndex = Math.floor(data.length * percTrainData);
var trainData = data.slice(0, splitIndex);
var testData = data.slice(splitIndex);
// variable parameters (the current values are only to test the code, we should change them for the task)
var n_iter = 25;
var networkShape = [2, 2, 1];
var swarm = aux.runPSO(n_iter, trainData, testData, networkShape);
// TASK 2: GA
// Split training data again
splitIndex = Math.floor(trainData.length * percTrainData);
var trainDataGA = trainData.slice(0, splitIndex);
var pop = ga.buildPop(2, [2, 2], [10, 10], 'gd');
ga.oneStepGA(pop, trainDataGA, trainData);
var bestConfig = pop.getBestConfig(trainDataGA, trainData);
console.log(bestConfig);
// Run a final training with the best config
// TASK 3: GP
