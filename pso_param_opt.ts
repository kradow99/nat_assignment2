import * as nn from "./playground_pso/src/nn";
import * as pso_test from "./playground_pso/src/pso_test";
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

function runPSO(n_iter: number, trainData: Example2D[], testData: Example2D[],
    networkShape: number[], omega, alpha1, alpha2, swrmsz): number[] {

        // Define activation functions
        let activation = nn.Activations.TANH;
        let outputActivation = nn.Activations.TANH;
        let regularization = null;
        let network = nn.buildNetwork(networkShape, activation,
            outputActivation,regularization, aux.constructInputIds());
        let swarm = pso_test.buildSwarm(nn.countWeights(network), omega, alpha1, alpha2, swrmsz);
        let ret = [];
        let prev_ret = [];
        let count = 0;
        for (var t = 0; t < n_iter; t++) {
            ret = onePSOStep(swarm, network, trainData, testData);
            if (prev_ret[0] - ret[0] < 0.0001) ++count;
            else count = 0;
            if (count >= 500) return ret; //if stagnation, end
            prev_ret = [...ret]; //copy by value
        }

        return ret;
}

function onePSOStep(swarm: pso_test.Swarm, network: nn.Node[][], trainData: Example2D[], testData: Example2D[]): number[] {

    let fitness = swarm.updateSwarm(network, trainData); /* changes the network to the gbest network from the swarm */
    // Compute the loss.
    let lossTrain = aux.getLoss(network, trainData);
    let lossTest = aux.getLoss(network, testData);
    return [fitness, lossTrain, lossTest];
}

// Constant Parameters
//const NUM_SAMPLES_CLASSIFY = 500;
const percTrainData = 0.50;

//let data = dataset.classifySpiralData(NUM_SAMPLES_CLASSIFY, 0);
//dataset.shuffle(data);
let data = aux.loadDataFile('./two_spirals.dat')
//console.log(data)

// Split into train and test data.
let splitIndex = Math.floor(data.length * percTrainData);
let trainData = data.slice(0, splitIndex);
let testData = data.slice(splitIndex);

// variable parameters (the current values are only to test the code, we should change them for the task)
let n_iter = 3000;
let networkShape = [6, 8, 1];


//for (let omega = 0.8; omega <=1; omega += 0.1 ) { //omega loop
let omega = 0.8
for (let alpha2 = 1.2; alpha2 <=2.5; alpha2 += 0.1 ) { //alpha2 loop
  for (let alpha1 = 0.5; alpha1 <=2.5; alpha1 += 0.1 ) { //alpha1 loop
    let best = [Number.MAX_VALUE,Number.MAX_VALUE,Number.MAX_VALUE];
    let mean = [0,0,0];
    let n = 20;
    for (let i = 0; i < n; ++i) { //100 iterations, get the best fitness parameters
      let ret = runPSO(n_iter, trainData, testData, networkShape, omega, alpha1, alpha2, 10);
      if (ret[0] < best[0]) best = [...ret];
      mean[0] += ret[0];
      mean[1] += ret[1];
      mean[2] += ret[2];
    }
    mean[0] = mean[0]/n;
    mean[1] = mean[1]/n;
    mean[2] = mean[2]/n;
    console.log(omega,alpha1,alpha2);
    console.log(best[0], best[1], best[2]);
    console.log(mean[0], mean[1], mean[2]);
  }
}
//}
