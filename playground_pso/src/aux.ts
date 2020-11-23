// Auxiliary functions to avoid duplicating code across the modeules
import * as nn from "./nn";
import * as pso from "./pso";
import {Example2D} from "./dataset";
import * as fs from 'fs'

export function loadDataFile(path:string): Example2D[] {
  let points: Example2D[] = [];
  let text = fs.readFileSync(path).toString('utf-8');
  let textByLine = text.split("\n");
  for (let i = 0; i < textByLine.length-1; i++) {
    let line = textByLine[i].replace(/  /g, ' ');
    if (line[0] == ' ') {
      line = line.substring(1)
    }
    let lineSplitted = line.split(' ');
    let x = parseFloat(lineSplitted[0]);
    let y = parseFloat(lineSplitted[1]);
    let label = parseFloat(lineSplitted[2]);
    if (label == 0) label = -1;
    points.push({x, y, label});
  }
  return points;
}

export function writeExperimentResults(parameters, results, 
  filename: string, paramName: string, includeConfig: boolean) {
  let file = fs.createWriteStream(filename);
  file.on('error', function(err) { console.log("Error in writing") });
  for (let i in parameters) {
    let paramValue = String(parameters[i]);
    let error = results[i][1];
    if (includeConfig) {
      file.write(paramName +  " : "  + paramValue + ",  Square error: " + error 
      + "Config" + String(results[i][0]) + "\n");
    } else {
      file.write(paramName +  " : "  + paramValue + ",  Square error: " + error + "\n");
    }
  }
  file.end();
}

export function writeValue(value, label, filename) {

  let message = label + " : " + String(value);
  fs.writeFile(filename, message, function (err) {
    if (err) return console.log(err);
    console.log("Result written");
  })

}

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
    "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
    "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
    //"xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
    "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
    "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

  interface InputFeature {
    f: (x: number, y: number) => number;
    label?: string;
}

export function runPSO(n_iter: number, trainData: Example2D[], testData: Example2D[],
    networkShape: number[]): pso.Swarm {

        // Define activation functions
        let activation = nn.Activations.TANH;
        let outputActivation = nn.Activations.TANH;
        let regularization = null;
        let network = nn.buildNetwork(networkShape, activation,
            outputActivation,regularization, constructInputIds());
        let swarm = pso.buildSwarm(nn.countWeights(network));
        for (var t = 0; t < n_iter+1; t++) {
            let losses = onePSOStep(swarm, network, trainData, testData);
            if (t % 50 == 0) {
              console.log("Fitness:", losses[0].toFixed(4), "Train Loss:", losses[1].toFixed(4), "    Test Loss:", losses[2].toFixed(4), "   Iter:", t);
            }
        }

        return swarm;


}


function onePSOStep(swarm: pso.Swarm, network: nn.Node[][], trainData: Example2D[], testData: Example2D[]): number[] {

    let fitness = swarm.updateSwarm(network, trainData); /* changes the network to the gbest network from the swarm */
    // Compute the loss.
    let lossTrain = getLoss(network, trainData);
    let lossTest = getLoss(network, testData);
    return [fitness, lossTrain, lossTest];
}

export function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
    let loss = 0;
    for (let i = 0; i < dataPoints.length; i++) {
      let dataPoint = dataPoints[i];
      let input = constructInput(dataPoint.x, dataPoint.y);
      let output = nn.forwardProp(network, input);
      let label = dataPoint.label;
      loss += nn.Errors.SQUARE.error(output, label);
    }
    return loss / dataPoints.length;
}
