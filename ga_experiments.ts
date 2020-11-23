import * as nn from "./playground_pso/src/nn";
import * as pso from "./playground_pso/src/pso";
import * as ga from "./playground_pso/src/ga";
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

let data = aux.loadDataFile('./two_spirals.dat');
dataset.shuffle(data);

// Split into train and test data.
let splitIndex = Math.floor(data.length * percTrainData);
let trainData = data.slice(0, splitIndex);
let testData = data.slice(splitIndex);

// TASK 2: GA

splitIndex = Math.floor(trainData.length * percTrainData);
let trainDataGA = trainData.slice(0, splitIndex);

function evaluateGA(lowLevelAlgo: string, popSize: number, layers: number, 
    minRanges: number[], maxRanges: number[], crossProb: number, 
    mutProb: number, generations: number, n_iter: number) {

    let popGA = ga.buildPop(layers, minRanges, maxRanges, lowLevelAlgo,
        popSize, mutProb);
    for (let t = 0; t < generations; t++) {
        ga.oneStepGA(popGA, trainDataGA, trainData);
        if (t % 5 == 0) {
            console.log(popGA.bestFit);
        }
    }

    let bestConfig = popGA.getBestConfig(trainData, testData);
    if (lowLevelAlgo === 'pso') {
        let bestFitness = ga.getFitnessPSO(bestConfig, n_iter, trainData, testData);
        let bestError = 1 - bestFitness;
        console.log("Best Error: ", bestError);
        return [bestConfig, bestError];
    } else if(lowLevelAlgo === 'gd') {
        let bestFitness = ga.getFitnessNN(bestConfig, n_iter, trainData, testData);
        let bestError = 1 - bestFitness;
        console.log("Best Error: ", bestError);
        return [bestConfig, bestError];
    } else {
        console.log("Unknown lower level algorithm");
        return null;
    }

}

// Split training data again

// Baseline

let lowLevelAlgo = 'gd';
let popSize = 10;
let layers = 2;
let minRanges = [2, 2];
let maxRanges = [10, 10];
let crossProb = 0.70; 
let mutProb = 0.01;
let generations = 50;
let n_iter = 5000;

let baselineResults = evaluateGA(lowLevelAlgo, popSize, layers, minRanges, 
    maxRanges, crossProb, mutProb, generations, n_iter);

aux.writeValue(baselineResults[0], "Baseline Config", "ga_baseline_config.txt");
aux.writeValue(baselineResults[1], "Baseline Error", "ga_baseline_error.txt");

// Exeriment 1: Deep thin network

let numLayers = [3, 4, 5, 6, 7, 8, 9, 10];
let minNodes = 2;
let maxNodes = 5;
let expResults = [];
for (let l  of numLayers) {
    let minRanges = [];
    let maxRanges = [];
    for(let i = 0; i < l; i++) {
        minRanges.push(minNodes);
        maxRanges.push(maxNodes);
    }
    let result = evaluateGA(lowLevelAlgo, popSize, l, minRanges, 
        maxRanges, crossProb, mutProb, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(numLayers, expResults, "ga_deep_thin_exp.txt", "Layers", true);

// Exeriment 2: Shallow thick network

let maxNumNodes = [8, 10, 15, 20, 25];
minNodes = 2;
expResults = [];

for (let maxNodes  of maxNumNodes) {
    let minRanges = [];
    let maxRanges = [];
    for(let i = 0; i < layers; i++) {
        minRanges.push(minNodes);
        maxRanges.push(maxNodes);
    }
    let result = evaluateGA(lowLevelAlgo, popSize, layers, minRanges, 
        maxRanges, crossProb, mutProb, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(maxNumNodes, expResults, "ga_shallow_thick_exp.txt", "Maximum Nodes per Layer", true);

// Experiment 3: Bottleneck Network

maxRanges = [8, 5, 3];
minRanges = [2, 2, 2];

let bottleneckResults = evaluateGA(lowLevelAlgo, popSize, layers, minRanges, 
    maxRanges, crossProb, mutProb, generations, n_iter);

aux.writeValue(bottleneckResults[0], "Bottleneck Config", "ga_bottleneck_config.txt");
aux.writeValue(bottleneckResults[1], "Bottleneck Error", "ga_bottleneck_error.txt");

// Experiment 4: Population Size

let sizes = [5, 10, 15, 20, 25];
minRanges = [2, 2];
maxRanges = [10, 10];
expResults = [];

for (let size  of sizes) {

    let result = evaluateGA(lowLevelAlgo, size, layers, minRanges, 
        maxRanges, crossProb, mutProb, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(sizes, expResults, "ga_popsize_exp.txt", "Population Size", false);

// Experiment 5: Crossover Probability

let probs = [0.25, 0.50, 0.75, 0.90];
expResults = [];

for (let p  of probs) {

    let result = evaluateGA(lowLevelAlgo, popSize, layers, minRanges, 
        maxRanges, p, mutProb, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(probs, expResults, "ga_co_probs_exp1.txt", "Crossover Probability", true);

// To see if crossover exploit deeper networks

for (let p  of probs) {

    let result = evaluateGA(lowLevelAlgo, popSize, 4, [2,2,2,2], 
        [8,8,8,8], p, mutProb, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(probs, expResults, "ga_co_probs_exp2.txt", "Crossover Probability", true);

// Experiment 6: Mutation Probability

probs = [0.001, 0.01, 0.05, 0.10];
expResults = [];

for (let p  of probs) {

    let result = evaluateGA(lowLevelAlgo, popSize, layers, minRanges, 
        maxRanges, crossProb, p, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(probs, expResults, "ga_mut_probs_exp1.txt", "Mutation Probability", true);

// Experiment 7: Low level algorithm

let llAlgo = ['gd', 'pso'];
expResults = [];

for (let algo of llAlgo) {
    let result = evaluateGA(algo,popSize, layers, minRanges, 
        maxRanges, crossProb, mutProb, generations, n_iter);
    expResults.push(result);
}

aux.writeExperimentResults(llAlgo, expResults, "ga_LowLevelAlgo_exp1.txt", "Low level algorithm", true);

// With different number of iterations
expResults = [];
let result = evaluateGA(llAlgo[0], popSize, layers, minRanges, 
        maxRanges, crossProb, mutProb, generations, n_iter);
expResults.push(result);
result = evaluateGA(llAlgo[1], popSize, layers, minRanges, 
    maxRanges, crossProb, mutProb, generations, n_iter + 1000);
expResults.push(result);
aux.writeExperimentResults(llAlgo, expResults, "ga_LowLevelAlgo_exp2.txt", "Low level algorithm", true);