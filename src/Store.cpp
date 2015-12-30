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
	eta = new float[particles];
	noise = new float[particles];
}

void Store::fileOpen (){
	out = "/home/jitesh/Documents/c++/cuda/particleAlignment/output/";
	parseOut = "/home/jitesh/Documents/c++/cuda/output/";
	name = "noiseDistributionEvolution";
	parse = "parseNoiseVector";
	format = ".csv";
	ss << out << name << format;
	pf << parseOut << parse << format;
	finalName = ss.str();
	parseFile = pf.str();
	fout.open(finalName.c_str());
	fin.open(parseFile.c_str());
	
}

float* Store::parseAndReturn(){
	for (int i = 0; i < 256; i++){
		fin >> noise[i];
	}
	return noise;
}

void Store::print(int i){
	fout << eta[i] << "\n";
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
	delete []eta;
	delete []noise;
}
