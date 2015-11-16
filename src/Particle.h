/*   Particle.h
	Author: Jitesh
	Created on 25/July/2015
	*/
#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <stdlib.h>
#include "../utils/cuda_vector_math.cuh"

class Particle{

public: 
	float2 coord;	//coordinates vector of a particle
	float2 vel;	//velocity vector of a particle
	float2 dir;	//direction vector of a particle
	float theta;	//angle with x axis of the velocity vector of a particle
	const static float speed = 0.03;	//initialise speed for the particle
	const static float Rs = 1.0;		//initialise radius of interation of a particle
	float eta;	//parameter that controls noise of a particle
	
public:
	Particle();
	void init(float systemSize, float noise);
	~Particle();
};

#endif








