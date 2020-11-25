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

let data = aux.loadDataFile('./two_spirals.dat');
// To use randomly generated data, uncomment the two following lines
//let data = dataset.classifySpiralData(NUM_SAMPLES_CLASSIFY, 0);
//dataset.shuffle(data);

// Split into train and test data.
let splitIndex = Math.floor(data.length * percTrainData);
let trainData = data.slice(0, splitIndex);
let testData = data.slice(splitIndex);

// variable parameters
let n_iter = 3000;
let networkShape = [6, 8, 1];

let swarm = aux.runPSO(n_iter, trainData, testData, networkShape)
