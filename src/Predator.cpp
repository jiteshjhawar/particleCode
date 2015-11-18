/*   Predator.cpp
	Author: Jitesh
	Created on 16/Nov/2015
	*/

#include "Predator.h"
#include <math.h>

Predator::Predator(){}
//This function initialises coordinate vectors and velocitiy vectors of one particle
void Predator::init(float systemSize, float noise){
	coord.x = (1.0 * rand() / RAND_MAX) * systemSize;	
	coord.y = (1.0 * rand() / RAND_MAX) * systemSize;
	theta = (2.0 * rand() / RAND_MAX) * M_PI;
	dir.x = cos(theta);
	dir.y = sin(theta);
	vel.x = speed * dir.x;
	vel.y = speed * dir.y;
	eta = noise;
}

Predator::~Predator(){}
