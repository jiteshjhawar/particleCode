/*
 * Store.h
 *
 *  Created on: 06-05-2015
 *      Author: jitesh
 */

#ifndef STORE_H_
#define STORE_H_
#include <iostream>
#include <fstream>
#include <sstream>
#include<sys/stat.h>
#include<sys/types.h>
using namespace std;

class Store{

public:
	float orientationParam;
	float *eta;
	ofstream fout;
	ifstream fin;
	stringstream ss, pf;
	stringstream nf;
	string out, parseOut;
	string name, parse;	
	string format;
	string finalName, parseFile;
	float *noise;
	
public:
	Store(int particles);	
	void fileOpen();
	void print(int i);
	void printGroupSize(int groupSize);
	void printTime(float time);
	void fileClose();
	float *parseAndReturn();
	void endl();
	~Store();

};

#endif /* STORE_H_ */
