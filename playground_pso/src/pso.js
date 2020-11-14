"use strict";
exports.__esModule = true;
exports.getFitness2 = exports.getFitness = exports.buildSwarm = exports.Swarm = exports.Particle = void 0;
/* A simple PSO algorithm and the interaction with the NN via the fitness function */
var OMEGA = 0.8;
var ALPHA1 = 1.5;
var ALPHA2 = 1.0;
var SWRMSZ = 20;
var LIMITS = 0.5; // initialisation is within [-LIMITS,LIMITS]^dim
var nn = require("./nn");
//let state = State.deserializeState();
var Particle = /** @class */ (function () {
    function Particle(dim, limits) {
        this.x = [];
        this.v = [];
        this.p = [];
        this.d = dim;
        for (var i = 0; i < this.d; i++) {
            this.x[i] = limits * 2 * (Math.random() - 0.5);
            this.v[i] = 2 * (Math.random() - 0.5);
            this.p[i] = this.x[i];
            this.fp = Number.MAX_VALUE;
        }
    }
    Particle.prototype.updatePersonalBest = function (f) {
        if (f < this.fp) {
            this.fp = f;
            for (var j = 0; j < this.d; j++) {
                this.p[j] = this.x[j];
            }
        }
    };
    Particle.prototype.getVelocity = function () {
        var mean_vel = 0;
        for (var j = 0; j < this.d; j++) {
            mean_vel += this.v[j] * this.v[j];
        }
        mean_vel = Math.sqrt(mean_vel / this.d);
        return (mean_vel);
    };
    Particle.prototype.updateParticleVeloPos = function (g) {
        for (var j = 0; j < this.d; j++) {
            this.v[j] = OMEGA * this.v[j] + ALPHA1 * Math.random() * (this.p[j] - this.x[j]) + ALPHA2 * Math.random() * (g[j] - this.x[j]);
            this.x[j] += this.v[j];
            if (Math.abs(this.x[j]) > 10.0)
                this.x[j] = 10.0 * (Math.random() - 0.5);
            /* if swarm diverges, then rather change parameters! */
        }
    };
    return Particle;
}());
exports.Particle = Particle;
var Swarm = /** @class */ (function () {
    function Swarm() {
        this.particles = [];
        this.g = []; // global best (so far) vector
    }
    Swarm.prototype.updateGlobalBest = function (f, i) {
        if (f < this.fg) {
            this.fg = f;
            for (var j = 0; j < this.dim; j++) {
                this.g[j] = this.particles[i].x[j];
            }
        }
    };
    Swarm.prototype.updateSwarm = function (network, trainData) {
        var f = -1;
        for (var i = 0; i < SWRMSZ; i++) {
            f = getFitness(network, trainData, this.particles[i].x, this.dim);
            this.particles[i].updatePersonalBest(f);
            this.updateGlobalBest(f, i);
        }
        for (var i = 0; i < SWRMSZ; i++) {
            this.particles[i].updateParticleVeloPos(this.g);
        }
        nn.setWeights(network, this.g, this.dim); /* assigns g to weights for visualisation of best */
        f = getFitness(network, trainData, this.g, this.dim);
        //return(this.particles[0].v[1]);
        //return(this.particles[0].getVelocity());
        return f;
    };
    return Swarm;
}());
exports.Swarm = Swarm;
function buildSwarm(nnDim) {
    var swrm = new Swarm;
    swrm.dim = nnDim;
    for (var i = 0; i < SWRMSZ; i++) {
        var part = new Particle(swrm.dim, LIMITS);
        swrm.particles.push(part);
    }
    swrm.fg = Number.MAX_VALUE;
    for (var j = 0; j < this.d; j++) {
        this.g[j] = this.particles[0].x[j];
    }
    return swrm;
}
exports.buildSwarm = buildSwarm;
/* Fitness function using cross-entropy */
function getFitness(network, trainData, x, dim) {
    //if (nnn === dim) return(getLoss(network, trainData));
    var nnn = nn.setWeights(network, x, dim); /* assign x to weights */
    var total = 0;
    for (var i = 0; i < trainData.length; i++) {
        var dataPoint = trainData[i];
        var input = constructInput(dataPoint.x, dataPoint.y);
        var output = nn.forwardProp(network, input);
        var y = dataPoint.label;
        if (y == 1) {
            total -= Math.log(output);
        }
        else
            total -= Math.log(1 - output);
    }
    return total;
}
exports.getFitness = getFitness;
/* Fitness function using sum(|y - sigmoid(ŷ)|)  */
function getFitness2(network, trainData, x, dim) {
    var nnn = nn.setWeights(network, x, dim); /* assign x to weights */
    var total = 0;
    for (var i = 0; i < trainData.length; i++) {
        var dataPoint = trainData[i];
        var input = constructInput(dataPoint.x, dataPoint.y);
        var output = nn.forwardProp(network, input);
        var sigmoid = 1 / (1 + Math.exp(-output * 1000));
        var y = dataPoint.label;
        if (y == -1)
            y = 0;
        total += Math.abs(y - sigmoid);
    }
    return total;
}
exports.getFitness2 = getFitness2;
/* the following ones are copies from playground.ts */
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
var INPUTS = {
    "x": { f: function (x, y) { return x; }, label: "X_1" },
    "y": { f: function (x, y) { return y; }, label: "X_2" }
};
