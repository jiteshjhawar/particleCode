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
	float2 coord;
	float2 vel;
	float2 dir;
	float theta;
	const static float speed = 0.03;
	const static float Rs = 1.0;
	float eta;
	
public:
	Particle();
	void init(float systemSize, float noise);
	~Particle();
};

#endif








