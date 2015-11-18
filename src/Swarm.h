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
#include "Predator.h"
#include "../utils/cuda_device.h"
#include <curand_kernel.h>
using namespace std;

class Swarm{

public:
	int nParticles;
	int nPredators;	
	int size;
	float systemSize;
	int *h_id;
	int *d_id;
	int *h_sz;
	int *d_sz;
	Particle *h_particles;
	Particle *d_particles;
	Predator *h_predators;
	Predator *d_predators;
	float2 *d_sumdir;
	float *h_uniteIdx, *h_uniteIdy;
	float *d_uniteIdx, *d_uniteIdy;
	float *d_c, *d_dist, *randArray;
	float sumxspeed;
	float sumyspeed;
	float orderParam, msd;
	curandState *d_state;
	
public:
	Swarm(int n, float L, int nPred);
	void init(float noise);
	void initPredator(float predNoise);
	void initid();
	int allocate();
	int cudaCopy();
	int cudaCopyPred();
	int cudaUniteIdCopy();
	int update();
	int grouping();
	void launchUpdateKernel(int nParticles, float systemSize, int nPredators);
	void launchRandInit(unsigned long t);
	void group(float *h_uniteIdx, float *h_uniteIdy, int nParticles, int *h_id, int *h_sz);
	//void predUpdater(Predator *d_predators, int nPredators, Particle *d_particles, int nParticles, float systemSize);
	int findgroups();
	void calcgsd(int *gsd);
	float calcOrderparam();
	float calcMSD();
	int cudaBackCopy();
	int cudaUniteIdBackCopy();
	Particle const *returnParticles(){ return h_particles; };
	Predator const *returnPredators(){ return h_predators; };
	~Swarm();

};

#endif
