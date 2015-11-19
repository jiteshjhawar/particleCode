/*   Particle.cpp
	Author: Jitesh
	Created on 25/July/2015
	*/
#include "Particle.h"
#include <math.h>

Particle::Particle(){}
//This function initialises coordinate vectors and velocitiy vectors of one particle
void Particle::init(float systemSize, float noise){
	coord.x = (1.0 * rand() / RAND_MAX) * systemSize;	
	coord.y = (1.0 * rand() / RAND_MAX) * systemSize;
	theta = (2.0 * rand() / RAND_MAX) * M_PI;
	dir.x = cos(theta);
	dir.y = sin(theta);
	vel.x = speed * dir.x;
	vel.y = speed * dir.y;
	eta = (1.0 * rand() / RAND_MAX) * noise;
}

Particle::~Particle(){}
