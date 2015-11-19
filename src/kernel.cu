/*   kernel.cu
	Author: Jitesh
	Created on 25/July/2015
	*/

#include <cuda_runtime.h>
#include <curand.h>
#include "Swarm.h"
//periodic boundary implementation
inline __device__ __host__ float doPeriodic(float x, float L){
	x = fmodf(x, L);
	x += L;							// this second fmodf is required because some implementations..
	x = fmodf(x, L);	// ..of fmodf use negative remainders in principle range
	return x; 
}
//distance correction for periodic boundary
inline __device__ __host__ float distCorr(float D, float world){
	D = abs(D);
	D = min(D, world - D);
	return D;
}
//function that calculates distance between two particles/vectors
inline __device__ __host__ float calcDist (float2 a, float2 b, float L){
	float dx, dy, dist;
	dx = a.x - b.x;
	dy = a.y - b.y;
	dx = distCorr(dx, L);
	dy = distCorr(dy, L);
	dist = powf((powf(dx, 2) + powf(dy, 2)), 0.5);
	return dist;
}
//function to calculate distance w.r.t. predator
inline __device__ __host__ float predDist(float2 a, float2 b, float L){
	float dx, dy, dist;
	dx = a.x - b.x;
	dy = a.y - b.y;
	dist = powf((powf(dx, 2) + powf(dy, 2)), 0.5);
	return dist;
}
//function that returns the root/parent of a particle
inline __device__ __host__ int root (int i, int *id){
	while (i != id[i]){
		id[i] = id[id[i]]; //path compression
		i = id[i];
	}	
	return i;
}

//union function
inline __device__ __host__ void unite (int p, int q, int *id, int *sz){
	int i = root(p, id);
	int j = root(q, id);
	if (i != j){
		if (sz[i] <= sz[j]){
			id[i] = j;
			sz[j] += sz[i];
		}
		else{
			id[j] = i;
			sz[i] += sz[j];
		}
	}	
}
//function that checks if two particles are neighbours and align their direction if it is so.
__device__ __host__ void alignmentFunction (Particle *d_particles, int nParticles, float L, int pid, float2 * d_sumdir, float * d_c, float *d_dist, float *d_uniteIdx, float *d_uniteIdy){
	for (int cid = 0; cid < nParticles; cid++){
		//distance
		d_dist[pid*nParticles+cid] = calcDist(d_particles[pid].coord, d_particles[cid].coord, L);			
		//calculate total number of particles in the vicinity and sum their directions
		d_uniteIdx[pid*nParticles+cid] = 0.0;
		d_uniteIdy[pid*nParticles+cid] = 0.0;
		if (d_dist[pid*nParticles+cid] <= Particle::Rs/* && cid != pid*/){
			d_sumdir[pid].x += d_particles[cid].dir.x;
			d_sumdir[pid].y += d_particles[cid].dir.y;
			d_c[pid] = d_c[pid] + 1.0;
			d_uniteIdx[pid*nParticles+cid] = pid;
			d_uniteIdy[pid*nParticles+cid] = cid;
		}
	}
}
//function that checks distance between every particle from predator and if they are with in Rd, applies repulsion
__device__ __host__ void predRepulsion(Particle *d_particles, int nParticles, Predator *d_predators, int nPredators, float L, int pid){
	float dist;
	for (int i = 0; i < nPredators; i++){
		dist = predDist(d_particles[pid].coord, d_predators[i].coord, L);
	if (dist < Particle::Rd && dist < L/2)
		d_particles[pid].dir = d_particles[pid].coord - d_predators[i].coord;
	else if (dist < Particle::Rd && dist >= L/2)
		d_particles[pid].dir = d_particles[pid].coord - d_predators[i].coord;
	}	
}
//kernel that updates position and velocities to each particle
__global__ void updateKernel (Particle *d_particles, int nParticles, Predator *d_predators, int nPredators, float systemSize, float2 *d_sumdir, float *d_c, float *d_dist, float* randArray, float *d_uniteIdx, float *d_uniteIdy){
	float w = 1.0;
	int childId;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nParticles)	//checking idx to be between 0 and maximum number of particles
		return;
	d_sumdir[idx].x = 0.0; d_sumdir[idx].y = 0.0;
	d_c[idx] = 0.0;
	//call alignment kernel/function to calculate average direction of particles in vicinity
	//alignmentKernel <<<(nParticles / 256) + 1, 256 >>>(d_particles, nParticles, systemSize, idx, d_sumdir, d_c, d_dist);
	alignmentFunction (d_particles, nParticles, systemSize, idx, d_sumdir, d_c, d_dist, d_uniteIdx, d_uniteIdy);
	__syncthreads();
	//alignment (update direction with respect to average direction of particles in vicinity)
	d_particles[idx].dir.x = (w * d_particles[idx].dir.x + d_sumdir[idx].x) / (d_c[idx] + w);
	d_particles[idx].dir.y = (w * d_particles[idx].dir.y + d_sumdir[idx].y) / (d_c[idx] + w);
	//prey repulsion from predator
	predRepulsion(d_particles, nParticles, d_predators, nPredators, systemSize, idx);
	__syncthreads();
	//checking of attack and birth of new individual with noise drawn from existing population
	for (int i = 0; i < nPredators; i++){
		float distPrey = calcDist(d_particles[idx].coord, d_predators[i].coord, systemSize);
		if (distPrey < Predator::Ra){
			d_particles[idx].coord.x = randArray[idx] * systemSize;
			d_particles[idx].coord.x = randArray[idx+1] * systemSize;
			childId = randArray[idx] * (nParticles-1);
			d_particles[idx].eta = d_particles[childId].eta;
		}
	}
	//calculate theta
	d_particles[idx].theta = atan2(d_particles[idx].dir.y, d_particles[idx].dir.x);
	//Adding noise to theta
	d_particles[idx].theta = d_particles[idx].theta + (d_particles[idx].eta * ((randArray[idx] * 2.0) - 1) / 2.0);
	//calculate directions from theta
	d_particles[idx].dir.x = cos(d_particles[idx].theta);
	d_particles[idx].dir.y = sin(d_particles[idx].theta);
	//updating velocity of particles
	d_particles[idx].vel = d_particles[idx].dir * Particle::speed;
	__syncthreads();
	//updating coordinates of particles
	d_particles[idx].coord.x += d_particles[idx].vel.x;
	d_particles[idx].coord.y += d_particles[idx].vel.y;
	__syncthreads();
	//implementing periodic boundary
	d_particles[idx].coord.x = doPeriodic(d_particles[idx].coord.x, systemSize);		
	d_particles[idx].coord.y = doPeriodic(d_particles[idx].coord.y, systemSize);		
}
//function to update predator's position
__host__ __device__ void predUpdater(Predator *h_predators, int nPredators, Particle *h_particles, int nParticles, float L){
	float dist1, dist2, preyDist;
	for (int i = 0; i < nPredators; i++){
		int index = 0;
		for (int j = 0; j < nParticles-1; j++){
			dist1 = calcDist(h_particles[index].coord, h_predators[i].coord, L);
			dist2 = calcDist(h_particles[j+1].coord, h_predators[i].coord, L);
			if (dist1 > dist2)
				index = j+1;
		}
		preyDist = predDist(h_particles[index].coord, h_predators[i].coord, L);
		if (preyDist > L / 2 && preyDist < Predator::Rd)
			h_predators[i].dir = h_predators[i].coord - h_particles[index].coord;
		else if (preyDist <= L / 2 && preyDist < Predator::Rd)
			h_predators[i].dir = h_particles[index].coord - h_predators[i].coord;
		h_predators[i].theta = atan2(h_predators[i].dir.y, h_predators[i].dir.x);
		h_predators[i].dir.x = cos(h_predators[i].theta);
		h_predators[i].dir.y = sin(h_predators[i].theta);
		h_predators[i].vel = (h_predators[i].dir * Predator::speed);			
		//updating coordinates of particles
		h_predators[i].coord.x += h_predators[i].vel.x;
		h_predators[i].coord.y += h_predators[i].vel.y;
		//implementing periodic boundary
		h_predators[i].coord.x = doPeriodic(h_predators[i].coord.x, L);		
		h_predators[i].coord.y = doPeriodic(h_predators[i].coord.y, L);
	}
}

//function to seed states
__global__ void init_stuff (curandState* state, unsigned long seed){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}
//function to generate random numbers using seeded states and copy them to randArray
__global__ void make_rand (curandState* state, float* randArray){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	randArray[idx] = curand_uniform(&state[idx]);
}
//function that takes unite ids as input to perform union operation
void Swarm::group(float *h_uniteIdx, float *h_uniteIdy, int nParticles, int *h_id, int *h_sz){
	for (int i = 0; i < nParticles*nParticles; i++){
		unite(h_uniteIdx[i], h_uniteIdy[i], h_id, h_sz);
	}
}
//function to launch random number initialiser kernel
void Swarm::launchRandInit(unsigned long t){
	init_stuff <<<((nParticles - 1) / 256) + 1, 256>>> (d_state, t);
}
//function to launch random number generation and updation kernel
void Swarm::launchUpdateKernel(int nParticles, float systemSize, int nPredators){
	make_rand <<<((nParticles - 1) / 256) + 1, 256>>> (d_state, randArray);
	cudaCopyPred();
	updateKernel <<<((nParticles - 1) / 256) + 1, 256>>> (d_particles, nParticles, d_predators, nPredators, systemSize, d_sumdir, d_c, d_dist, randArray, d_uniteIdx, d_uniteIdy);
	cudaBackCopy();
	predUpdater(h_predators, nPredators, h_particles, nParticles, systemSize);
}
