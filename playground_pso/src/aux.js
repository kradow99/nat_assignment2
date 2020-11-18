"use strict";
exports.__esModule = true;
exports.getLoss = exports.runPSO = exports.constructInputIds = exports.constructInput = exports.oneStepNN = exports.getRandomInt = void 0;
// Auxiliary functions to avoid duplicating code across the modeules
var nn = require("./nn");
var pso = require("./pso");
function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}
exports.getRandomInt = getRandomInt;
function oneStepNN(trainData, valData, network) {
    trainData.forEach(function (point, i) {
        var input = constructInput(point.x, point.y);
        nn.forwardProp(network, input);
        nn.backProp(network, point.label, nn.Errors.SQUARE);
        nn.updateWeights(network, 0.03, 0); // For now predefined learning rate and regularization
    });
}
exports.oneStepNN = oneStepNN;
function constructInput(x, y) {
    var input = [];
    for (var inputName in INPUTS) {
        input.push(INPUTS[inputName].f(x, y));
        //if (state[inputName]) {
        // input.push(INPUTS[inputName].f(x, y));
        //}
    }
    return input;
}
exports.constructInput = constructInput;
function constructInputIds() {
    var result = [];
    for (var inputName in INPUTS) {
        result.push(inputName);
    }
    return result;
}
exports.constructInputIds = constructInputIds;
var INPUTS = {
    "x": { f: function (x, y) { return x; }, label: "X_1" },
    "y": { f: function (x, y) { return y; }, label: "X_2" },
    //"xSquared": {f: (x, y) => x * x, label: "X_1^2"},
    //"ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
    "xTimesY": { f: function (x, y) { return x * y; }, label: "X_1X_2" },
    "sinX": { f: function (x, y) { return Math.sin(x); }, label: "sin(X_1)" },
    "sinY": { f: function (x, y) { return Math.sin(y); }, label: "sin(X_2)" }
};
function runPSO(n_iter, trainData, testData, networkShape) {
    // Define activation functions
    var activation = nn.Activations.TANH;
    var outputActivation = nn.Activations.SIGMOID;
    var regularization = null;
    var network = nn.buildNetwork(networkShape, activation, outputActivation, regularization, constructInputIds());
    var swarm = pso.buildSwarm(nn.countWeights(network));
    for (var t = 0; t < 100; t++) {
        var losses = onePSOStep(swarm, network, trainData, testData);
        console.log("Train Loss: ", losses[0], "    Test Loss: ", losses[1]);
    }
    return swarm;
}
exports.runPSO = runPSO;
function onePSOStep(swarm, network, trainData, testData) {
    swarm.updateSwarm(network, trainData); /* changes the network to the gbest network from the swarm */
    // Compute the loss.
    var lossTrain = getLoss(network, trainData);
    var lossTest = getLoss(network, testData);
    return [lossTrain, lossTest];
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
exports.getLoss = getLoss;
