import * as nn from "./playground_pso/src/nn";
import * as pso from "./playground_pso/src/pso";
import * as ga from "./playground_pso/src/ga";
import * as gp from "./playground_pso/src/gp";
import * as aux from "./playground_pso/src/aux";
import * as dataset from "./playground_pso/src/dataset";
import {Example2D} from "./playground_pso/src/dataset";
import {
    State,
    datasets,
    getKeyFromValue,
} from "./playground_pso/src/state";

// Constant Parameters
const NUM_SAMPLES_CLASSIFY = 500;
const percTrainData = 0.50;

// TASK1: PSO




//let data = dataset.classifySpiralData(NUM_SAMPLES_CLASSIFY, 0);
//dataset.shuffle(data);
let data = aux.loadDataFile('./two_spirals.dat')
//console.log(data)
// Split into train and test data.
let splitIndex = Math.floor(data.length * percTrainData);
let trainData = data.slice(0, splitIndex);
let testData = data.slice(splitIndex);

// variable parameters (the current values are only to test the code, we should change them for the task)
let n_iter = 1000;
let networkShape = [2, 2, 1];

let swarm = aux.runPSO(n_iter, trainData, testData, networkShape)

// TASK 2: GA

// Split training data again
splitIndex = Math.floor(trainData.length * percTrainData);
let trainDataGA = trainData.slice(0, splitIndex);
let pop = ga.buildPop(2, [2, 2], [10, 10], 'gd');
ga.oneStepGA(pop, trainDataGA, trainData);
let bestConfig = pop.getBestConfig(trainDataGA, trainData);
console.log(bestConfig);

// Run a final training with the best config

// TASK 3: GP

let popGP = gp.buildPop('gd');
console.log(popGP);
gp.oneStepGP(popGP, trainDataGA, trainData, true);
console.log(popGP);
