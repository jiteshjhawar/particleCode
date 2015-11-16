/*   main.cpp
	Author: Jitesh
	Created on 25/July/2015
	*/

#include <iostream>
//#include <SDL2/SDL.h>
//#include "Screen.h"
#include "../utils/simple_timer.h"
#include "Swarm.h"
#include "Store.h"
using namespace std;

int main (int argc, char *argv[]){
	srand(time(NULL));		//seeding rand with time
	time_t t;				
	time(&t);
	const float systemSize = 32.0;		
	const int particles = 1024;
	const float maxeta = 1.5;		//maximum noise parameter
	const int realisations = 200;	//number of realisations
	const int iterations = 3000;	//number of time steps
	const int last = 100;			//number of last steps over which order parameter would be averaged
	int c;
	int *gsd;						//pointer to initialise array that stores different group size 
	float timeElapsed;
	Store store(particles);			//Store class object 
	store.fileOpen();
	Swarm swarm(particles, systemSize);
	swarm.allocate();
	swarm.launchRandInit((unsigned long) t);
	SimpleTimer time; time.reset();
	time.start();
	for (float eta = maxeta; eta <= maxeta; eta = eta + 0.2){		//loop to iterate over different noise values
		store.orientationParam = 0.0;				//initialize OP to zero before each round of replication
		for (int rep = 0; rep < realisations; rep++){		//loop to perform more number of realizations
			swarm.init(eta);
			swarm.initid();
			swarm.cudaCopy();
			/*Screen screen;
			if (screen.init() == false){
				cout << "error initialising SDL." << endl;
			}*/
			for (int i = 0; i < iterations; i++){		//loop to run the simulations for number of timesteps
				//screen.clear();
				swarm.update();
				/*const Particle * const pParticles = swarm.returnParticles();	//store the particle
				swarm.cudaBackCopy();
				for (int p = 0; p < particles; p++){
					Particle particle = pParticles[p];

					int x = particle.coord.x * Screen::SCREEN_WIDTH / systemSize;
					int y = particle.coord.y * Screen::SCREEN_HEIGHT / systemSize;
					//store.printCoord(x,y);
					screen.setPixel(x, y, 125, 255, 125);
					}
				screen.update();*/	
				/*if (i >= iterations - last){
					swarm.cudaBackCopy();
					store.orientationParam += swarm.calcOrderparam();
				}
				if (screen.processEvents() == false || c == iterations){
					break;
				}*/
			}
			//screen.close();
			if (cudaDeviceSynchronize() != cudaSuccess)
				cout << "Device synchronisation failed \n";
			swarm.cudaUniteIdBackCopy();
			swarm.grouping();
			c = swarm.findgroups();
			//cout << "number of independent groups are " << c << "\n";
			gsd = new int[c];
			swarm.calcgsd(gsd);
			for (int i = 0; i < c; i++){
				store.printGroupSize(gsd[i]);
			}
			store.endl();	
		}
		/*store.endl();
		store.orientationParam = store.orientationParam / realisations / last;
		//cout << store.orientationParam << "\n";
		store.print(eta);
		store.endl();*/
	}
	time.stop();
	timeElapsed = time.printTime();
	store.printTime(timeElapsed);
	store.fileClose();
	
	delete []gsd;
	return 0;
}
