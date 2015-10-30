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
	ofstream fout;
	stringstream ss;
	stringstream nf;
	string out;
	string name;	
	string format;
	string finalName;
	
public:
	Store(int particles);	
	void fileOpen();
	void print(float theta);
	void printCoord(float x, float y);
	void fileClose();
	void endl();
	~Store();

};

#endif /* STORE_H_ */
