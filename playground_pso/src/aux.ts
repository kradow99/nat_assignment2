// Auxiliary functions to avoid duplicating code across the modeules
import * as nn from "./nn";
import * as pso from "./pso";
import {Example2D} from "./dataset";

export function getRandomInt(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min)) + min;
}

export function oneStepNN(trainData: Example2D[], valData: Example2D[], network: nn.Node[][]): void {
    trainData.forEach((point, i) => {
      let input = constructInput(point.x, point.y);
      nn.forwardProp(network, input);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
      nn.updateWeights(network, 0.03, 0); // For now predefined learning rate and regularization
    });
}

export function constructInput(x: number, y: number): number[] {
    let input: number[] = [];
    for (let inputName in INPUTS) {
      input.push(INPUTS[inputName].f(x, y));
      //if (state[inputName]) {
       // input.push(INPUTS[inputName].f(x, y));
      //}
    }
    return input;
}

export function constructInputIds(): string[] {
    let result: string[] = [];
    for (let inputName in INPUTS) {
        result.push(inputName);
    }
    return result;
}

let INPUTS: {[name: string]: InputFeature} = {
    "x": {f: (x, y) => x, label: "X_1"},
    "y": {f: (x, y) => y, label: "X_2"},
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

export function runPSO(n_iter: number, trainData: Example2D[], testData: Example2D[],
    networkShape: number[]): pso.Swarm {
        
        // Define activation functions
        let activation = nn.Activations.TANH;
        let outputActivation = nn.Activations.SIGMOID;
        let regularization = null;
        let network = nn.buildNetwork(networkShape, activation, 
            outputActivation,regularization, constructInputIds());
        let swarm = pso.buildSwarm(nn.countWeights(network));
        for (var t = 0; t < 100; t++) {
            let losses = onePSOStep(swarm, network, trainData, testData); 
            console.log("Train Loss: ", losses[0], "    Test Loss: ", losses[1]);
        }

        return swarm;


}


function onePSOStep(swarm: pso.Swarm, network: nn.Node[][], trainData: Example2D[], testData: Example2D[]): number[] {

    swarm.updateSwarm(network, trainData); /* changes the network to the gbest network from the swarm */
    // Compute the loss.
    let lossTrain = getLoss(network, trainData);
    let lossTest = getLoss(network, testData);
    return [lossTrain, lossTest];
}

export function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
    let loss = 0;
    for (let i = 0; i < dataPoints.length; i++) {
      let dataPoint = dataPoints[i];
      let input = constructInput(dataPoint.x, dataPoint.y);
      let output = nn.forwardProp(network, input);
      loss += nn.Errors.SQUARE.error(output, dataPoint.label);
    }
    return loss / dataPoints.length;
}
