/*   Swarm.cpp
	Author: Jitesh
	Created on 25/July/2015
	*/

#include "Swarm.h"
#include <cuda_runtime.h>

Swarm::Swarm(int n, float L){
	systemSize = L;	
	nParticles = n;
	h_particles = new Particle[nParticles];
	size = sizeof(Particle) * nParticles;
}

void Swarm::init(float noise){
for (int i = 0; i < nParticles; i++){
		h_particles[i].init(systemSize, noise);
	}
}

int Swarm::allocate(){
	if (cudaMalloc(&d_particles, size) != cudaSuccess){
		cout << "unable to allocate memory on device" << endl;
		delete []h_particles;
		return 0;
	}
	if (cudaMalloc(&d_sumdir, (sizeof(float2) * nParticles)) != cudaSuccess){
		cout << "unable to allocate memory on device" << endl;
		delete []h_particles;
		cudaFree(d_particles);
		return 0;
	}
	if (cudaMalloc(&d_c, (sizeof(float) * nParticles)) != cudaSuccess){
		cout << "unable to allocate memory on device" << endl;
		delete []h_particles;
		cudaFree(d_particles);
		cudaFree(d_sumdir);
		return 0;
	}
	if (cudaMalloc(&d_dist, (sizeof(float) * nParticles)) != cudaSuccess){
		cout << "unable to allocate memory on device" << endl;
		delete []h_particles;
		cudaFree(d_particles);
		cudaFree(d_sumdir);
		return 0;
	}
	if (cudaMalloc(&d_state, nParticles * sizeof(curandState)) != cudaSuccess){
		cout << "unable to allocate memory on device" << endl;
		delete []h_particles;
		cudaFree(d_particles);
		cudaFree(d_sumdir);
		cudaFree(d_dist);
		return 0;
	}
	if (cudaMalloc(&randArray, nParticles * sizeof(float)) != cudaSuccess){
		cout << "unable to allocate memory on device" << endl;
		delete []h_particles;
		cudaFree(d_particles);
		cudaFree(d_sumdir);
		cudaFree(d_dist);
		cudaFree(d_state);
		return 0;
	}
}

int Swarm::cudaCopy(){
	checkCudaErrors(cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice));
}

int Swarm::update(){
	launchUpdateKernel(nParticles, systemSize);
}

int Swarm::cudaBackCopy(){
	checkCudaErrors(cudaMemcpy(h_particles, d_particles, size, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(h_sumdir, d_sumdir, (sizeof(float2) * nParticles), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(h_c, d_c (sizeof(float) * nParticles), cudaMemcpyDeviceToHost));	
		//cout << h_particles[0].coord.x << "\t" << h_particles[0].coord.y << "\n";
}

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

float Swarm::calcMSD(){
	msd = 0.0;
	for (int i = 0; i < nParticles; i++){
		msd += pow((h_particles[i].coord.x - 0.0), 2) + pow((h_particles[i].coord.y - 0.0), 2);
	}
	return msd;
}

Swarm::~Swarm(){
	delete []h_particles;
	cudaFree(d_particles);
	cudaFree(d_sumdir);
	cudaFree(d_c);
	cudaFree(d_dist);
	cudaFree(d_state);
	cudaFree(randArray);
	
}
