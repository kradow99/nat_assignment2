"use strict";
exports.__esModule = true;
var nn = require("./playground_pso/src/nn");
var pso = require("./playground_pso/src/pso");
var ga = require("./playground_pso/src/ga");
var dataset = require("./playground_pso/src/dataset");
// Copies from ./playground_pso/src/playground.ts
function constructInputIds() {
    var result = [];
    for (var inputName in INPUTS) {
        result.push(inputName);
    }
    return result;
}
function constructInput(x, y) {
    var input = [];
    for (var inputName in INPUTS) {
        input.push(INPUTS[inputName].f(x, y));
    }
    return input;
}
// recall to sync with pso inputs
var INPUTS = {
    "x": { f: function (x, y) { return x; }, label: "X_1" },
    "y": { f: function (x, y) { return y; }, label: "X_2" }
    //"xSquared": {f: (x, y) => x * x, label: "X_1^2"},
    //"ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
    //"xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
    //"sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
    //"sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};
function onePSOStep() {
    //iter++;
    PSOout = swarm.updateSwarm(network, trainData); /* changes the network to the gbest network from the swarm */
    // Compute the loss.
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
    return PSOout;
}
function getLoss(network, dataPoints) {
    var loss = 0;
    for (var i = 0; i < dataPoints.length; i++) {
        var dataPoint = dataPoints[i];
        var input = constructInput(dataPoint.x, dataPoint.y);
        var output = nn.forwardProp(network, input);
        loss += nn.Errors.SQUARE.error(output, dataPoint.label);
    }
    return loss / dataPoints.length;
}
// Constant Parameters
var NUM_SAMPLES_CLASSIFY = 500;
var percTrainData = 0.50;
// TASK1: PSO
// parameters
var networkShape = [2, 2, 1];
var activation = nn.Activations.TANH;
var outputActivation = nn.Activations.SIGMOID;
var regularization = null;
var data = dataset.classifySpiralData(NUM_SAMPLES_CLASSIFY, 0);
dataset.shuffle(data);
// Split into train and test data.
var splitIndex = Math.floor(data.length * percTrainData);
var trainData = data.slice(0, splitIndex);
var testData = data.slice(splitIndex);
var network = nn.buildNetwork(networkShape, activation, outputActivation, regularization, constructInputIds());
var swarm = pso.buildSwarm(nn.countWeights(network));
var iter = 0;
var PSOout = 0;
var lossTrain = 0;
var lossTest = 0;
onePSOStep();
for (var t = 0; t < 20; t++) {
    onePSOStep();
    console.log(lossTrain, lossTest);
}
// TASK 2: GA
// Split training data again
splitIndex = Math.floor(trainData.length * percTrainData);
var trainDataGA = trainData.slice(0, splitIndex);
var pop = ga.buildPop(1, [2], [4], 'pso');
ga.oneStepGA(pop, trainDataGA, trainData);
var bestConfig = pop.getBestConfig(trainDataGA, trainData);
console.log(bestConfig);
