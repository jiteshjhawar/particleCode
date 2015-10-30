/*   main.cpp
	Author: Jitesh
	Created on 25/July/2015
	*/

#include <iostream>
#include <SDL2/SDL.h>
#include "Screen.h"
#include "Swarm.h"
#include "Store.h"
using namespace std;

int main (int argc, char *argv[]){
	srand(time(NULL));
	time_t t;
	time(&t);
	const float systemSize = 5.0;
	const int particles = 100;
	const float maxeta = 5.0;
	const int realisations = 1;
	const int iterations = 2000;
	const int last = 100;
	int c;
	float theta;
	Store store(particles);
	store.fileOpen();
	Swarm swarm(particles, systemSize);
	swarm.allocate();
	swarm.launchRandInit((unsigned long) t);
	for (float eta = maxeta; eta <= maxeta; eta = eta + 1.0){
		store.orientationParam = 0.0;
		for (int rep = 0; rep < realisations; rep++){	
			swarm.init(eta);
			swarm.cudaCopy();
			/*Screen screen;
			if (screen.init() == false) {
				cout << "error initialising SDL." << endl;
			}*/
			c = 0;
			for (int i = 0; i < iterations; i++){
				//screen.clear();
				swarm.update();
				//swarm.cudaBackCopy();
				//store.msd[i] += swarm.calcMSD(); //store msd for each time step in an array also sum each value for many
				/*theta = swarm.returnRandTheta();
				store.print(theta, i);*/
				const Particle * const pParticles = swarm.returnParticles();
				if (i >= 1000){
					swarm.cudaBackCopy();
					for (int p = 0; p < particles; p++){
						Particle particle = pParticles[p];

						float x = particle.coord.x/* * Screen::SCREEN_WIDTH / systemSize*/;
						float y = particle.coord.y/* * Screen::SCREEN_HEIGHT / systemSize*/;
						store.printCoord(x,y);
						//screen.setPixel(x, y, 125, 255, 125);
					}
					//screen.update();
				
				 			//...many realisations at each time step
					
					store.orientationParam = swarm.calcOrderparam();
					store.print(eta);
					store.endl();
				}
				c++;
				/*if (screen.processEvents() == false || c == iterations){
					break;
				}*/
			}
			//screen.close();
			
		}
		//store.orientationParam = store.orientationParam / realisations / last;
		//store.print(eta);
		//store.endl();
		/*for (int i = 0; i < iterations; i++){
			store.msd[i] = store.msd[i] / realisations / last / particles; //calculate average msd for each time step
			store.print(eta, i);
		}*/
	}
	store.fileClose();
	
	return 0;
}
