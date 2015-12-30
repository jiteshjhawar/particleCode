/*   Predator.h
	Author: Jitesh
	Created on 16/Nov/2015
	*/
#ifndef PREDATOR_H_
#define PREDATOR_H_ 

#include <stdlib.h>
#include "../utils/cuda_vector_math.cuh"

class Predator {
public:
	float2 coord;	//coordinates vector of a particle
	float2 vel;	//velocity vector of a particle
	float2 dir;	//direction vector of a particle
	float theta;	//angle with x axis of the velocity vector of a particle
	const static float speed = 0.04;
	const static float Rd = 1.0;
	const static float Ra = 0.1;
	float eta;
	
public:
	Predator();
	void init(float systemSize, float noise);
	~Predator();


};

#endif
