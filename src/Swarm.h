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
	int *h_id;		//host group id array
	int *d_id;		//device group id array
	int *h_sz;		//host group size array
	int *d_sz;		//device group size array
	Particle *h_particles;		//host Particle type array
	Particle *d_particles;		//device Particle type array
	Predator *h_predators;		//host Predator type array 
	Predator *d_predators;		//device Predator type array
	float2 *d_sumdir;			//device pointer to store summation of directions of near neighbours
	float *h_uniteIdx, *h_uniteIdy;			//host array to store ids of individuals to be united
	float *d_uniteIdx, *d_uniteIdy;			//device array to store ids of individuals to be united 
	float *d_randNorm;			//device pointer to generate random number from a normal distribution
	float *d_c, *d_dist, *randArray;		//d_c stores the number of near neighbours, d_dist is a 2D array that stores distances between
											//each and every particle, randArray stores a random number genarated from a uniform 
											//distribution in its each element
	float sumxspeed;			//to store summation of x component of velocity
	float sumyspeed;			//to store summation of y component of velocity
	float orderParam, msd;		//orderParam is the average velocity of all the individuals
	curandState *d_state;
	float *d_preyPredDist, *d_preyPredDistNP;
	int *d_attack, *h_attack;
	
public:
	Swarm(int n, float L, int nPred);
	void init(float noise);
	void initPredator(float predNoise);
	void initid();
	void initAttack();
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
