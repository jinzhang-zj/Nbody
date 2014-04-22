//
//  NBody_gendata.cpp
//  generate test data
//
//

#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<string>
#include<iostream>
#include<sstream>
#include<math.h>
#include<vector>

using namespace std;

void createTestCase(int x, int maxLevel, string file) {
    ofstream out;
    double fx = (double)x;
    out.open(file.c_str());
    out<< x*x <<" "<<maxLevel<<endl;
    for(int i = 0; i < x; i++)
        for(int j = 0; j < x; j++) {
            out <<(i+1)/(fx+1)<<" "<<(j+1)/(fx+1)<<" "<<1<<endl;
        }
    out.close();
}

void createHelper(ofstream &out, double corX, double corY, int level, int n, vector<double> &r){
	double loc = (double)rand()/ RAND_MAX;
	double scale = pow(2,level+1);
	double gap = 1.0/scale;
        double x = (double)rand()/RAND_MAX /scale;
        double y = (double)rand()/RAND_MAX /scale;
	

	if (loc <= 0.4){
		r.push_back(corX + x);
		r.push_back(corY + y);
	}else if(loc <= 0.7){
		r.push_back(corX + x);
		r.push_back(corY + y + gap);
	}else if(loc <= 0.9){
		r.push_back(corX + x + gap);
		r.push_back(corY + y);
	}else{ 
		r.push_back(corX + x + gap);
		r.push_back(corY + y + gap);
	}

	// if you want 2**(2n+2) points
	if (level >= n){
		return;
	}

	createHelper(out, corX, corY, level+1, n,r);
	createHelper(out, corX+gap, corY, level+1, n,r);
	createHelper(out, corX, corY+gap, level+1, n,r);
	createHelper(out, corX+gap, corY+gap, level+1, n,r);
}

void createTestCaseNU(int x, int maxLevel, string file) {
    ofstream out;
    // total number of Points
    int totalPoints = x*x;
    int n = (int)log2(totalPoints);  
    vector<double> r;
    double corX = 0;
    double corY = 0;
    int level = 0;
    for (int i=0; i<3; i++){
  	  createHelper(out, corX, corY, level, (n-2)/2, r);
    }
    r.push_back(0.45);
    r.push_back(0.55);
    cout << "rsize = 2**" << log2(r.size()/2) << endl;
    out.open(file.c_str());
    out<< x*x <<" "<<maxLevel<<endl;
    for (int i=0; i< r.size()/2; i++){
        out << r[2*i] << " " << r[2*i+1] << " " << 1 << endl;
    }
    out.close();
}

int main(int argc, char* argv[])
{   //string file = "/Users/xinyangyi/test.txt";
    
    // create 512*512, 1024*1024, 2048*2048 points
    createTestCase(512, 20, "NodeTest18");
    createTestCase(1024, 20, "NodeTest20");
    createTestCase(2048, 20, "NodeTest22");
    createTestCaseNU(512, 20, "NodeTest18nu");
    createTestCaseNU(1024, 20, "NodeTest20nu");
    createTestCaseNU(2048, 20, "NodeTest22nu");
}
