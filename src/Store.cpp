/*
 * Swarm.cpp
 *
 *  Created on: 06-05-2015
 *      Author: jitesh
 */

using namespace std;
#include "Store.h"


Store::Store(int particles){
	/*for (int i = 0; i < iterations; i++){
		msd[i] = 0.0;
	}*/
}

void Store::fileOpen (){
	out = "/home/jiteshjhawar/Documents/c++/cuda/particleGroupDist/output/";
	name = "/orientationParamWithNoise";
	format = ".csv";
	ss << out << name << format;
	finalName = ss.str();
	fout.open(finalName.c_str());
}

void Store::print(float theta){
	fout << orientationParam << "," << theta;
}

void Store::printGroupSize(int groupSize){
	fout << groupSize << "\n";
}

void Store::printTime(float time){
	fout << "t = " << time << " ms.\n";
}

void Store::endl(){
	fout << "\n";
}

void Store::fileClose(){
	fout.close();
}

Store::~Store(){
}
