import * as nn from "./playground_pso/src/nn";
import * as pso from "./playground_pso/src/pso";
import * as aux from "./playground_pso/src/aux";
import * as gp from "./playground_pso/src/gp";
import * as dataset from "./playground_pso/src/dataset";
import {Example2D} from "./playground_pso/src/dataset";
import {
    State,
    datasets,
    getKeyFromValue,
} from "./playground_pso/src/state";

const percTrainData = 0.50;

let data = aux.loadDataFile('./two_spirals.dat');
dataset.shuffle(data);

// Split into train and test data.
let splitIndex = Math.floor(data.length * percTrainData);
let trainData = data.slice(0, splitIndex);
let testData = data.slice(splitIndex);
splitIndex = Math.floor(trainData.length * percTrainData);
let trainDataGP = trainData.slice(0, splitIndex);

// TASK 3: GP

// Baseline

let popSize = 10;
let layers = 5;
let mutProb = 0.1;
let maxLayerSize = 8;
let connReach = 1;
let gensGP = 50;
let lowLevelAlgo = 'gd';
let testIter = 5000; 

function evaluateGP(lowLevelAlgo: string, popSize: number, maxLayerSize: number, layers: number, 
    connReach: number, mutProb: number, withCrossover: boolean, testIter: number) {

    let popGP = gp.buildPop(lowLevelAlgo, popSize, maxLayerSize, layers, connReach, mutProb);
    for (let t = 0; t < gensGP; t++) {
        gp.oneStepGP(popGP, trainDataGP, trainData, withCrossover);
        if (t % 5 == 0) {
            console.log(popGP.bestFit);
        }
    }
    let bestIndividual = popGP.getBestIndividual(trainData, testData, testIter);
    let bestFit = popGP.bestFit;
    let bestError = 1 - bestFit;
    return [bestIndividual.config, bestError];
}



let baseline = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, layers, connReach, mutProb, false, testIter)
let baselineConfig = baseline[0]
let baselineError = baseline[1]

aux.writeValue(baselineConfig, "Baseline Network Configuration", "gp_baseline_config.txt");
aux.writeValue(baselineError, "Baseline Network Error", "gp_baseline_error.txt");

// Experiment 1: max width and depth

let widths = [5, 10, 15, 20, 25]
let widthResults = [];

for (let maxWidth of widths) {
    let result = evaluateGP(lowLevelAlgo, popSize, maxWidth, layers, connReach, mutProb, false, testIter);
    widthResults.push(result);
}

aux.writeExperimentResults(widths, widthResults, "gp_width_exp.txt", "Max nodes per layer", false);

let depths = [5, 10, 15, 20, 25]
let depthResults = [];

for (let numLayers of depths) {
    let result = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, numLayers, connReach, mutProb, false, testIter);
    depthResults.push(result);
}

aux.writeExperimentResults(depths, depthResults, "gp_depth_exp.txt", "Layers", false);

// For more robustness in depth results:

maxLayerSize = 3;

depths = [1, 2, 3, 4, 5]
depthResults = [];

for (let numLayers of depths) {
    let result = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, numLayers, connReach, mutProb, false, testIter);
    depthResults.push(result);
}

aux.writeExperimentResults(depths, depthResults, "gp_overfitting_depth_exp.txt", "Layers", false);

maxLayerSize = 5;

// // Experiment 2: Connection Reach

let reaches = [1, 2, 3, 4, 5];
let reachResults = [];

for (let reach of reaches) {
    let result = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, layers, reach, mutProb, false, testIter);
    reachResults.push(result);
}

aux.writeExperimentResults(reaches, reachResults, "gp_reach_exp.txt", "Max connection reach", false);

// Experiment 3: Mutation probabilities

let probs = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25];
let probResults = [];

for (let prob of probs) {
    let result = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, layers, connReach, prob, false, testIter);
    probResults.push(result);
}

aux.writeExperimentResults(probs, probResults, "gp_probs_exp.txt", "Mutation probability", false);

// // Experiment 4: Population Size

let popSizes = [10, 20, 30, 40, 50];
let popSizeResults = [];

for (let size of popSizes) {
    let result = evaluateGP(lowLevelAlgo, size, maxLayerSize, layers, connReach, mutProb, false, testIter);
    popSizeResults.push(result);
}

aux.writeExperimentResults(popSizes, popSizeResults, "gp_pop_size_exp.txt", "Population Size", false);

// Experiment 5: Crossover

let crossoverBool = [false, true];
let coResults = [];

for (let coBool of crossoverBool) {
    let result = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, layers, connReach, mutProb, coBool, testIter);
    coResults.push(result);
}

aux.writeExperimentResults([0, 1], coResults, "gp_crossover_exp.txt", "With crossover", false);


// Experiment 6: Low level algorithm

let llAlgo = ['gd', 'pso'];
let llAlgoResults = [];

for (let algo of llAlgo) {
    let result = evaluateGP(algo, popSize, maxLayerSize, layers, connReach, mutProb, false, testIter);
    llAlgoResults.push(result);
}

aux.writeExperimentResults(llAlgo, llAlgoResults, "gp_LowLevelAlgo_exp.txt", "Low level algorithm", false);

// Optimal program
console.log("Running optimal algorithm")
popSize = 10;
maxLayerSize = 25;
layers = 5;
connReach = 5;
mutProb = 0.01
gensGP = 100;
lowLevelAlgo = 'gd';

let optimal = evaluateGP(lowLevelAlgo, popSize, maxLayerSize, layers, connReach, mutProb, false, 10000)
let optimalConfig = optimal[0]
let optimalError = optimal[1]

aux.writeValue(optimalConfig, "Optimal Network Configuration", "gp_optimal_config.txt");
aux.writeValue(baselineError, "Optimal Network Error", "gp_optimal_error.txt");