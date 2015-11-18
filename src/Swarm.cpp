/*   Swarm.cpp
	Author: Jitesh
	Created on 25/July/2015
	*/

#include "Swarm.h"
#include <cuda_runtime.h>
//constructor initialising system size, number of particles
Swarm::Swarm(int n, float L, int nPred){
	systemSize = L;	
	nParticles = n;
	nPredators = nPred;
	h_particles = new Particle[nParticles];	 //Particle type array of size of number of particles
	h_id = new int[nParticles];	//array to store grouping ids of particles
	h_sz = new int[nParticles];	//array to store group size of the group to which a particle belongs
	h_uniteIdx = new float[nParticles*nParticles];	//array to store the ids of particle which are neighbour of each other...
	h_uniteIdy = new float[nParticles*nParticles];	//... These are later used to perform unite operations.
	size = sizeof(Particle) * nParticles;
}
//initialise noise for each particle
void Swarm::init(float noise){
for (int i = 0; i < nParticles; i++){
		h_particles[i].init(systemSize, noise);
	}
}
//initialise predator
void Swarm::initPredator(float predNoise){
	h_predators = new Predator[nPredators];
	for (int i = 0; i < nPredators; i++){
		h_predators[i].init(systemSize, predNoise);
	}
}
//initialise group ids and group size of each particle
void Swarm::initid(){
for (int i = 0; i < nParticles; i++){
		h_id[i] = i;
		h_sz[i] = 1;
	}
}
//Memory allocation on the GPU
int Swarm::allocate(){
	checkCudaErrors(cudaMalloc(&d_particles, size));
	checkCudaErrors(cudaMalloc(&d_predators, sizeof(Predator) * nPredators));
	checkCudaErrors(cudaMalloc(&d_sumdir, sizeof(float2) * nParticles));
	checkCudaErrors(cudaMalloc(&d_c, sizeof(float) * nParticles));
	checkCudaErrors(cudaMalloc(&d_dist, sizeof(float) * nParticles * nParticles));
	checkCudaErrors(cudaMalloc(&d_state, nParticles * sizeof(curandState)));
	checkCudaErrors(cudaMalloc(&randArray, nParticles * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_uniteIdx, nParticles * nParticles * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_uniteIdy, nParticles * nParticles * sizeof(float)));
}
//Copy each attribute of Particle type array to the GPU which was initialised earlier on the CPU
int Swarm::cudaCopy(){
	checkCudaErrors(cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice));
}

int Swarm::cudaCopyPred(){
	checkCudaErrors(cudaMemcpy(d_predators, h_predators, sizeof(Predator) * nPredators, cudaMemcpyHostToDevice));
}
//update function that is called in the main program at each iterations that updates particle velocity and coordinates
int Swarm::update(){
	launchUpdateKernel(nParticles, systemSize, nPredators);	//function to launch updation kernel. (defined in kernel.cu)
}
//function that is used after the last time step in order to perform union operation using unite ids. 
int Swarm::grouping(){
	group(h_uniteIdx, h_uniteIdy, nParticles, h_id, h_sz);	//(defined in kernel.cu)
}
//function to copy Particle type array from GPU to CPU
int Swarm::cudaBackCopy(){
	checkCudaErrors(cudaMemcpy(h_particles, d_particles, size, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(h_predators, d_predators, sizeof(Predator) * nPredators, cudaMemcpyDeviceToHost));
}
//function to copy unite IDs array from GPU to CPU
int Swarm::cudaUniteIdBackCopy(){
	checkCudaErrors(cudaMemcpy(h_uniteIdx, d_uniteIdx, sizeof(float) * nParticles * nParticles, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_uniteIdy, d_uniteIdy, sizeof(float) * nParticles * nParticles, cudaMemcpyDeviceToHost));
}
//function that calculates order parameter at the current state and returns the same
float Swarm::calcOrderparam(){
	sumxspeed = 0.0; sumyspeed = 0.0;
	for (int i = 0; i < nParticles; i++) {
		Particle particle = h_particles[i];
		sumxspeed += particle.vel.x;
		sumyspeed += particle.vel.y;
	}
	orderParam = (pow((pow(sumxspeed,2) + pow(sumyspeed,2)),0.5) / nParticles) / Particle::speed;
	return orderParam;
}
//function to calculate mean sqaure displacement
float Swarm::calcMSD(){
	msd = 0.0;
	for (int i = 0; i < nParticles; i++){
		msd += pow((h_particles[i].coord.x - 0.0), 2) + pow((h_particles[i].coord.y - 0.0), 2);
	}
	return msd;
}
//function to find number of independent groups
int Swarm::findgroups (){
	int c = 0;
	for (int i = 0; i < nParticles; i++){
		if (h_id[i] == i)
			c++;
	}
	return c;
}
//function to find the size of each independent group
void Swarm::calcgsd(int *gsd){
	int j = 0;
	for (int i = 0; i < nParticles; i++){
		if (h_id[i] == i){
			gsd[j] = h_sz[i];
			j++;
		}
	}
}

Swarm::~Swarm(){
	delete []h_particles;
	delete []h_predators;
	delete []h_id;
	delete []h_sz;
	delete []h_uniteIdx;
	delete []h_uniteIdy;
	cudaFree(d_particles);
	cudaFree(d_predators);
	cudaFree(d_sumdir);
	cudaFree(d_c);
	cudaFree(d_dist);
	cudaFree(d_state);
	cudaFree(randArray);
	cudaFree(d_id);
	cudaFree(d_sz);
	
}
