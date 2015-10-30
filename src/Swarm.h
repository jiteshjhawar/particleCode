/*   Swarm.h
	Author: Jitesh
	Created on 25/July/2015
	*/

#ifndef SWARM_H_
#define SWARM_H_

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "Particle.h"
#include "../utils/cuda_device.h"
#include <curand_kernel.h>
using namespace std;

class Swarm{

public:
	int nParticles;	
	int size;
	float systemSize;
	Particle *h_particles;
	Particle *d_particles;
	float2 *d_sumdir;
	float *d_c, *d_dist, *randArray;
	float sumxspeed;
	float sumyspeed;
	float orderParam, msd;
	curandState *d_state;
	
public:
	Swarm(int n, float L);
	void init(float noise);
	int allocate();
	int cudaCopy();
	int update();
	void launchUpdateKernel(int nParticles, float systemSize);
	void launchRandInit(unsigned long t);
	float calcOrderparam();
	float calcMSD();
	int cudaBackCopy();
	Particle const *returnParticles(){ return h_particles; };
	~Swarm();

};

#endif
