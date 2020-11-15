import * as nn from "./playground_pso/src/nn";
import * as pso from "./playground_pso/src/pso";
import * as ga from "./playground_pso/src/ga";
import * as dataset from "./playground_pso/src/dataset";
import {Example2D} from "./playground_pso/src/dataset";
import {
    State,
    datasets,
    getKeyFromValue,
} from "./playground_pso/src/state";

// Copies from ./playground_pso/src/playground.ts
function constructInputIds(): string[] {
    let result: string[] = [];
    for (let inputName in INPUTS) {
        result.push(inputName);
    }
    return result;
}

function constructInput(x: number, y: number): number[] {
    let input: number[] = [];
    for (let inputName in INPUTS) {
        input.push(INPUTS[inputName].f(x, y));
    }
    return input;
  }

// recall to sync with pso inputs
let INPUTS: {[name: string]: InputFeature} = {
    "x": {f: (x, y) => x, label: "X_1"},
    "y": {f: (x, y) => y, label: "X_2"}
    //"xSquared": {f: (x, y) => x * x, label: "X_1^2"},
    //"ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
    //"xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
    //"sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
    //"sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

interface InputFeature {
    f: (x: number, y: number) => number;
    label?: string;
}

function onePSOStep(): number {
    //iter++;
    PSOout = swarm.updateSwarm(network,trainData); /* changes the network to the gbest network from the swarm */
    // Compute the loss.
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
    return PSOout;
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
    let loss = 0;
    for (let i = 0; i < dataPoints.length; i++) {
      let dataPoint = dataPoints[i];
      let input = constructInput(dataPoint.x, dataPoint.y);
      let output = nn.forwardProp(network, input);
      loss += nn.Errors.SQUARE.error(output, dataPoint.label);
    }
    return loss / dataPoints.length;
  }

// Constant Parameters
const NUM_SAMPLES_CLASSIFY = 500;
const percTrainData = 0.50;

// TASK1: PSO

// parameters
let networkShape = [2, 2, 1];
let activation = nn.Activations.TANH;
let outputActivation = nn.Activations.SIGMOID;
let regularization = null;


let data = dataset.classifySpiralData(NUM_SAMPLES_CLASSIFY, 0);
dataset.shuffle(data);
// Split into train and test data.
let splitIndex = Math.floor(data.length * percTrainData);
let trainData = data.slice(0, splitIndex);
let testData = data.slice(splitIndex);

let network = nn.buildNetwork(networkShape, activation, outputActivation,
    regularization, constructInputIds());

let swarm = pso.buildSwarm(nn.countWeights(network))

let iter = 0;
let PSOout = 0;
let lossTrain = 0;
let lossTest = 0;
onePSOStep()

for (var t = 0; t < 20; t++) {
    onePSOStep()
    console.log(lossTrain, lossTest);
}

// TASK 2: GA

// Split training data again
splitIndex = Math.floor(trainData.length * percTrainData);
let trainDataGA = trainData.slice(0, splitIndex);
let pop = ga.buildPop(1, [2], [4], 'pso');
ga.oneStepGA(pop, trainDataGA, trainData);
let bestConfig = pop.getBestConfig(trainDataGA, trainData);
console.log(bestConfig);

// Run a final training with the best config

