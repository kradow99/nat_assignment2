/* A simple PSO algorithm and the interaction with the NN via the fitness function */
const LIMITS = 0.5; // initialisation is within [-LIMITS,LIMITS]^dim

import * as nn from "./nn";
import {
  State,
  datasets,
  getKeyFromValue,
} from "./state";
import {Example2D} from "./dataset";
import * as aux from "./aux";

//let state = State.deserializeState();

export class Particle {
	x: number[] = [];
	v: number[] = [];
	p: number[] = [];
	fp: number;
	d: number;
  omega: number;
  alpha1: number;
  alpha2: number;
  swrmsz: number;

	constructor(dim: number, o: number, a1: number, a2: number, sw: number, limits: number) {
		this.d = dim;
    this.omega = o;
    this.alpha1 = a1;
    this.alpha2 = a2;
    this.swrmsz = sw;

		for (let i = 0; i < this.d; i++) {
			this.x[i] = limits * 2 * (Math.random() - 0.5);
			this.v[i] = 2 * (Math.random() - 0.5);
			this.p[i] = this.x[i];
			this.fp = Number.MAX_VALUE;
		}
	}
	updatePersonalBest(f: number) {
		if (f < this.fp) {
			this.fp = f;
			for (let j = 0; j < this.d; j++) {
				this.p[j] = this.x[j];
			}
		}
	}
	getVelocity() {
		let mean_vel = 0;
		for (let j = 0; j < this.d; j++) {
			mean_vel+=this.v[j]*this.v[j];
		}
		mean_vel=Math.sqrt(mean_vel / this.d);
		return(mean_vel);
	}
	updateParticleVeloPos(g: number[]) {
		for (let j = 0; j < this.d; j++) {
			this.v[j] = this.omega * this.v[j] + this.alpha1 * Math.random() * (this.p[j] - this.x[j]) + this.alpha2 * Math.random() * (g[j] - this.x[j]);
			this.x[j] += this.v[j];

			if (Math.abs(this.x[j])>10.0) this.x[j] = 10.0 * (Math.random() - 0.5);
				 /* if swarm diverges, then rather change parameters! */
		}
	}
}

export class Swarm {
	particles: Particle[] = [];
	part: Particle;
	g: number[] = []; // global best (so far) vector
	fg: number;	  // fitness of global best
	dim: number;
  omega: number;
  alpha1: number;
  alpha2: number;
  swrmsz: number;

	updateGlobalBest(f: number, i: number){
		if (f < this.fg) {
			this.fg = f;
			for (let j = 0; j < this.dim; j++) {
				this.g[j] = this.particles[i].x[j];
			}
		}
	}

	updateSwarm(network: nn.Node[][], trainData: Example2D[]): number {

		let f = -1;
		for (let i = 0; i < this.swrmsz; i++) {
			f = getFitness(network,trainData,this.particles[i].x,this.dim);
			this.particles[i].updatePersonalBest(f);
			this.updateGlobalBest(f,i);
		}

		for (let i = 0; i < this.swrmsz; i++) {
			this.particles[i].updateParticleVeloPos(this.g)
		}

		nn.setWeights(network, this.g, this.dim); /* assigns g to weights for visualisation of best */

		f = getFitness(network,trainData,this.g,this.dim);
 		//return(this.particles[0].v[1]);
		//return(this.particles[0].getVelocity());
    return f;
	}
}

export function	buildSwarm(nnDim: number, o: number, a1: number, a2: number, sw: number): Swarm {

  let swrm = new Swarm;
  swrm.omega = o;
  swrm.alpha1 = a1;
  swrm.alpha2 = a2;
  swrm.swrmsz = sw;
	swrm.dim = nnDim;
	for (let i = 0; i < swrm.swrmsz; i++) {
		let part = new Particle(swrm.dim, o, a1, a2, sw, LIMITS);
		swrm.particles.push(part);
	}
	swrm.fg = Number.MAX_VALUE;
	for (let j = 0; j < this.d; j++) {
		this.g[j] = this.particles[0].x[j];
	}

	return swrm
}

/* Fitness function using cross-entropy */
export function getFitness(network: nn.Node[][], trainData: Example2D[], x: number[], dim: number): number {
  let nnn = nn.setWeights(network, x, dim); /* assign x to weights */
  //if (nnn === dim) return(aux.getLoss(network, trainData));
  let total = 0;
  for (let i = 0; i < trainData.length; i++) {
    let dataPoint = trainData[i];
    let input = aux.constructInput(dataPoint.x, dataPoint.y);
    let output = (1+nn.forwardProp(network, input))/2; //output from [-1,1] to [0,1], for the log
    let y = dataPoint.label;
    if(y == 1) {
      total -= Math.log(output);
    }
    else total -= Math.log(1-output);
  }
  return total;
}
