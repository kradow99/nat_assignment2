/* A simple genetic algorithm and the interaction with the PSO/NN via the fitness function */
const POPSIZE = 5;
const CROSSPROB = 0.7;
const MUTPROB = 0.01;
const ELITISM = true;
const N_ITER = 100; // this is for the low leveel algorithm not for the GA
const MAX_COST = 1;
const INPUTDIM = 2;
import * as pso from "./pso";
import * as nn from "./nn";
import {Example2D} from "./dataset";
import * as aux from "./aux";

// Auxiliar functions
function getRandomInt(min, max): number {
    return Math.floor(Math.random() * (max - min)) + min;
}

export function buildNNFromGA(config: number[]): nn.Node[][]{

    let activation = nn.Activations.TANH;
    let outputActivation = nn.Activations.SIGMOID;
    let regularization = null;
    let shape = [INPUTDIM].concat(config).concat([1]); // input and output dimensions are fixed
    let network = nn.buildNetwork(shape, activation, outputActivation, regularization, aux.constructInputIds());
    return network;
}
export function getFitnessPSO(config: number[], n_iter: number, trainData: Example2D[], valData: Example2D[]): number{

    let network = buildNNFromGA(config);
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
        fitness_values.push(getFitnessPSO(i.config, n_iter, trainData, valData));
    }
    return fitness_values;
}

export function getFitnessNN(config: number[], n_iter: number, trainData: Example2D[], valData: Example2D[]) {
    
    let network = buildNNFromGA(config);
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
        fitness_values.push(getFitnessNN(i.config, n_iter, trainData, valData));
    }
    return fitness_values;
}

// Class for the individual of the population
export class Individual {
    config: number[] = [];
    dim: number;
    min_ranges: number[] = [];
    max_ranges: number[] = [];

    constructor(dim: number, min_ranges: number[], max_ranges: number[]) {
        this.dim = dim;
        this.min_ranges = min_ranges;
        this.max_ranges = max_ranges;
        for(let i = 0; i < this.dim; i++) {
            this.config.push(getRandomInt(this.min_ranges[i], this.max_ranges[i]))
        } 
    }

    mutate() {
        for(let chromosome in this.config){
            if (Math.random() < MUTPROB) {
                this.config[chromosome] = getRandomInt(this.min_ranges[chromosome], this.max_ranges[chromosome])
            }
        }
    }
}

export class Population {
    individuals: Individual[] = [];
    //mutProb: number;
    //crossProb: number;
    //elitism: boolean;
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
                let crossOverPoint = getRandomInt(0, parent1.dim);
                for(let chromosome = crossOverPoint; chromosome <= parent1.dim; chromosome++){
                    let temp1 = parent1[chromosome];
                    let temp2 = parent2[chromosome];
                    parent1[chromosome] = temp2;
                    parent2[chromosome] = temp1;
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

export function buildPop(dim: number, min_ranges: number[], max_ranges: number[], lowLevelAlgo: string): Population {
    let pop = new Population;
    pop.totalFit = 0;
    pop.averageFit = 0;
    pop.bestFit = 0;
    pop.lowLevelAlgo = lowLevelAlgo;

    for(let i = 0; i < POPSIZE; i++) {
        let member = new Individual(dim, min_ranges, max_ranges);
        pop.individuals.push(member);
    }

    return pop;

}

export function oneStepGA(pop: Population, trainData: Example2D[], valData: Example2D[]) {
    pop.selection(trainData, valData);
    pop.crossover();
    pop.mutation();
}