/* A simple genetic algorithm and the interaction with the PSO/NN via the fitness function and the Node class */
// Constant parameters

const MAXLAYERSIZE = 2; // This refers only to the hidden layers
const LAYERS = 2; // This refers only to the hidden layers
const NACTIVATIONS = 6;
const LAYERPADDING = 1;
const MUTPROB = 0.01;
const CROSSPROB = 0.7;
const ELITISM = true;
const INPUTDIM = 2;
const MAX_COST = 1;
const N_ITER = 10;
const POPSIZE = 5;
import * as nn from "./nn";
import * as aux from "./aux";
import * as pso from "./pso";
import {Example2D} from "./dataset";

// Auxiliary functions
function mapToMatrix(arrayIndex: number): number[] {
    if(arrayIndex % (LAYERS * MAXLAYERSIZE + 1) == 0) {
        //this corresponds to the case of an activation index
        console.log("Index corresponds to activation key");
        return null;
    }
    else {
        let subIndex = arrayIndex % (LAYERS * MAXLAYERSIZE + 1);
        let rowIndex = null;
        let colIndex = null;
        if (subIndex % LAYERS === 0) {
            colIndex = Math.floor(subIndex/LAYERS) - 1;
            rowIndex = LAYERS - 1;
        }
        else {
            colIndex = Math.floor(subIndex/LAYERS);
            rowIndex = LAYERS - (LAYERS - subIndex%LAYERS) - 1;
        }
        return [rowIndex, colIndex];
    }
}

function verifyLayer(index1: number, index2: number):boolean {
    let coord1 = mapToMatrix(index1);
    let coord2 = mapToMatrix(index2);
    let layer1 = coord1[1];
    let layer2 = coord2[1];
    return (layer1 - layer2) >= LAYERPADDING; 
}

function mapActivationKey (key: number): nn.ActivationFunction {

    let activationArray = [nn.Activations.RELU, nn.Activations.SIGMOID, nn.Activations.LINEAR, 
        nn.Activations.SIN, nn.Activations.RBF, nn.Activations.TANH];
    return activationArray[key];
}

export function getFitnessPSO(individual: Individual, n_iter: number, trainData: Example2D[], valData: Example2D[]): number{
    
    individual.translate(nn.Activations.TANH, nn.Activations.SIGMOID, aux.constructInputIds(), null);
    let network = individual.phenotype;
    let swarm = pso.buildSwarm(nn.countWeights(network))
    for (let i = 0; i <= n_iter; i++){
        swarm.updateSwarm(network,trainData);
    }
    // Compute the loss.
    let valLoss = aux.getLoss(network, valData);
    return MAX_COST - valLoss;
}

export function evaluatePopFitnessPSO(individuals: Individual[], n_iter: number, 
    trainData: Example2D[], valData: Example2D[]): number[] {

    let fitness_values: number[] = [];
    for(let i of individuals){
        fitness_values.push(getFitnessPSO(i, n_iter, trainData, valData));
    }
    return fitness_values;
}

export function getFitnessNN(individual: Individual, n_iter: number, trainData: Example2D[], valData: Example2D[]) {
    
    individual.translate(nn.Activations.TANH, nn.Activations.SIGMOID, aux.constructInputIds(), null);
    let network = individual.phenotype;
    for (let i = 0; i <= n_iter; i++){
        aux.oneStepNN(trainData, valData, network);
    }
    let valLoss = aux.getLoss(network, valData);
    return MAX_COST - valLoss;
}

export function evaluatePopFitnessNN(individuals: Individual[], n_iter: number, 
    trainData: Example2D[], valData: Example2D[]): number[] {

    let fitness_values: number[] = [];
    for(let i of individuals){
        fitness_values.push(getFitnessNN(i, n_iter, trainData, valData));
    }
    return fitness_values;
}

export class Individual {
    config: number[] = [];
    phenotype: nn.Node[][];

    constructor() {
        for(let current_node = 1; current_node <= MAXLAYERSIZE*LAYERS; current_node++){
            this.config.push(aux.getRandomInt(0, NACTIVATIONS))
            for (let current_conn  = 1; current_conn < MAXLAYERSIZE*LAYERS; current_conn++) {
                let currentConIndex = (current_node - 1)  *  (MAXLAYERSIZE*LAYERS + 1) +  current_conn;
 
                if (verifyLayer(current_node, currentConIndex)) {
                    this.config.push(Math.round(Math.random()));
                }
                else {
                    this.config.push(0); // We only allow connections between nodes up to a LAYERPADDING difference of Layers
                }
            }
        }
    }

    mutate() {
        for (let chromosome of this.config) {
            if (Math.random() < MUTPROB) {
                if (chromosome % (LAYERS * MAXLAYERSIZE + 1) == 0) { // Check if it corresponds to an activation node
                    this.config[chromosome] = aux.getRandomInt(0, NACTIVATIONS);
                }
                else {
                    if (this.config[chromosome] == 0) {
                        this.config[chromosome] = 1;
                    } 
                    else {
                        this.config[chromosome] = 0;
                    }                    
                }
            }
        }
    }

    // Largely based on nn.buildNetwork but allowing different activation function for each unit
    translate(inputActivation: nn.ActivationFunction, outputActivation: nn.ActivationFunction,
        inputIds: string[], regularization: nn.RegularizationFunction) {
    
        let numLayers = LAYERS + 2;
        let id = 1;
        /** List of layers, with each layer being a list of nodes. */
        let network: nn.Node[][] = [];
        for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            let isOutputLayer = layerIdx === numLayers - 1;
            let isInputLayer = layerIdx === 0;
            let isFirstHidden = layerIdx >= LAYERPADDING ;
            let currentLayer: nn.Node[] = [];
            network.push(currentLayer);
            let nodeId = '';
            let numNodes = 0;

            if (isInputLayer) {
                numNodes = INPUTDIM;
                for (let i = 0; i < numNodes; i++) {
                    nodeId = inputIds[i];
                    let node = new nn.Node(nodeId, inputActivation);
                    currentLayer.push(node);
                }             
            } else if (isOutputLayer) {
                nodeId = id.toString();
                id++;
                let node = new nn.Node(nodeId, outputActivation);
                currentLayer.push(node);
                for (let j = 0; j < network[layerIdx - 1].length; j++) {
                    // We make the ouput layer fully connected
                    let prevNode = network[layerIdx - 1][j];
                    let link = new nn.Link(prevNode, node, regularization);
                    prevNode.outputs.push(link);
                    node.inputLinks.push(link);
                }         

            } else if(isFirstHidden){

                numNodes = MAXLAYERSIZE;
                for (let i = 0; i < numNodes; i++) {
                    nodeId = id.toString();
                    id++;
                    let activationKey = this.config[i*(MAXLAYERSIZE*LAYERS + 1)];
                    let activation = mapActivationKey(activationKey);
                    let node = new nn.Node(nodeId, activation);
                    currentLayer.push(node);
                    for (let j = 0; j < network[layerIdx - 1].length; j++) {
                        // For the first hidden layer ignore genotype and make it fully connected
                        let prevNode = network[layerIdx - 1][j];
                        let link = new nn.Link(prevNode, node, regularization);
                        prevNode.outputs.push(link);
                        node.inputLinks.push(link);
                    }
            }  
            } else {
                numNodes = MAXLAYERSIZE;
                for (let i = 0; i < numNodes; i++) {
                    nodeId = id.toString();
                    id++;
                    let nodeArrayidx = ((layerIdx - 1)*MAXLAYERSIZE + i)*(MAXLAYERSIZE*LAYERS + 1);
                    let activationKey = this.config[nodeArrayidx]
                    let activation = mapActivationKey(activationKey);
                    let node = new nn.Node(nodeId, activation);
                    currentLayer.push(node);
                    for(let k = LAYERPADDING; k > 0; k --) {
                        for (let j = 0; j < network[layerIdx - k].length; j++) {
                            let PrevLayer = layerIdx - k - 1;
                            let arrayIndex = nodeArrayidx + (PrevLayer*MAXLAYERSIZE) + (j + 1);
                            let isDead = this.config[arrayIndex] === 1;
                            if (!isDead) {
                                let prevNode = network[layerIdx - k][j];
                                let link = new nn.Link(prevNode, node, regularization);
                                prevNode.outputs.push(link);
                                node.inputLinks.push(link);
                            }
                        }
                    }

                }
            }
        }
        this.phenotype = network;
    }
}

export class Population {
    
    individuals: Individual[] = [];
    lowLevelAlgo: string; // 'pso' or 'gd'
    totalFit: number;
    bestFit: number;
    averageFit: number;

    selection(trainData: Example2D[], valData: Example2D[]): void {

        let intermediatePop: Individual[] = []
        let popFitness: number[] = []
        if (this.lowLevelAlgo === 'pso') {
            popFitness = evaluatePopFitnessPSO(this.individuals, N_ITER, trainData, valData);
        }
        else if (this.lowLevelAlgo === 'gd'){
            popFitness = evaluatePopFitnessNN(this.individuals, N_ITER, trainData, valData);
        } else {
            popFitness = null;
            console.log("Low level algorithm not recognized");
        }
        let tempPop = this.individuals; // temporary array used only for sorting
        this.individuals.sort(function(a,b) {
            let fitness_a = popFitness[tempPop.indexOf(a)];
            let fitness_b = popFitness[tempPop.indexOf(b)];
            if (fitness_a > fitness_b) {
                return -1;
            } 
            if (fitness_b > fitness_a) {
                return 1;
            }
            return 0;
        })

        popFitness.sort(function(a,b) {
            if (a > b) {
                return -1
            } 
            if (b > a) {
                return 1;
            }
            return 0;
        })

        this.bestFit = popFitness[0];
        this.totalFit = popFitness.reduce((a,b) => a + b, 0);
        this.averageFit = this.totalFit / this.individuals.length;
        if (ELITISM){
            intermediatePop.push(this.individuals[0])
        }
        let relativeFitness = popFitness.map(x => x / this.averageFit);
        let i = 0;
        while (intermediatePop.length < this.individuals.length) {
            if (relativeFitness[i] >= 1) {
                intermediatePop.push(this.individuals[i]);
                relativeFitness[i] = relativeFitness[i] - 1;
            }
            else if(relativeFitness[i] > 0){
                if (Math.random() < relativeFitness[i]) {
                    intermediatePop.push(this.individuals[i]);
                }
            }
            
        }

        this.individuals = intermediatePop;

    }

    mutation() {
        for (let member of this.individuals) {
            member.mutate()
        }
    }

    crossover() {
        
        let intermediatePop: Individual[] = []
        let midPoint = this.individuals.length / 2;
        let parentPool = this.individuals.slice(midPoint - 1, POPSIZE);
        for (let parent1 of this.individuals.slice(0, midPoint)) {
            let parent2 = parentPool[Math.floor(Math.random()*parentPool.length)];
            if (Math.random() < CROSSPROB){
                let crossOverPoint = aux.getRandomInt(0, parent1.config.length - 1);
                for(let chromosome = crossOverPoint; chromosome < parent1.config.length; chromosome++){
                    let temp1 = parent1.config[chromosome];
                    let temp2 = parent2.config[chromosome];
                    parent1.config[chromosome] = temp2;
                    parent2.config[chromosome] = temp1;
                }                 
            }
            intermediatePop.push(parent1);
            intermediatePop.push(parent2);
            parentPool.splice(parentPool.indexOf(parent2), 1);
        }
        this.individuals = intermediatePop;
    }

    getBestConfig(trainData: Example2D[], valData: Example2D[]): number[] {

        let popFitness: number[] = []
        if (this.lowLevelAlgo === 'pso') {
            popFitness = evaluatePopFitnessPSO(this.individuals, N_ITER, trainData, valData);
        }
        else if (this.lowLevelAlgo === 'gd'){
            popFitness = evaluatePopFitnessNN(this.individuals, N_ITER, trainData, valData);
        } else {
            popFitness = null;
            console.log("Low level algorithm not recognized");
        }

        let tempPop = this.individuals; // temporary array used only for sorting

        this.individuals.sort(function(a,b) {
            let fitness_a = popFitness[tempPop.indexOf(a)];
            let fitness_b = popFitness[tempPop.indexOf(b)];
            if (fitness_a > fitness_b) {
                return -1;
            } 
            if (fitness_b > fitness_a) {
                return 1;
            }
            return 0;
        })

        let bestIndividual = this.individuals[0]
        console.log(popFitness);
        console.log("Fitness: ", popFitness[tempPop.indexOf(bestIndividual)]);
        let shape = [INPUTDIM].concat(bestIndividual.config).concat([1])
        return shape;
    }

}

export function buildPop(lowLevelAlgo: string): Population {
    let pop = new Population;
    pop.totalFit = 0;
    pop.averageFit = 0;
    pop.bestFit = 0;
    pop.lowLevelAlgo = lowLevelAlgo;

    for(let i = 0; i < POPSIZE; i++) {
        let member = new Individual();
        pop.individuals.push(member);
    }

    return pop;

}

export function oneStepGP(pop: Population, trainData: Example2D[], valData: Example2D[], withCrossOver: boolean) {
    pop.selection(trainData, valData);
    if (withCrossOver) {
        pop.crossover();
    }
    pop.mutation();
}