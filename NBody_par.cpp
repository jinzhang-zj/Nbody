//
//  main.cpp
//  test
//
//  Created by Xinyang Yi on 2/19/14.
//  Copyright (c) 2014 Xinyang Yi. All rights reserved.
//

#include<stdio.h>
#include<fstream>
#include<string>
#include<iostream>
#include<sstream>
#include<math.h>
#include<vector>
#include<omp.h>

using namespace std;


//interleave two integer x, y
unsigned int interleave(unsigned int x, unsigned int y) {
    unsigned int ans = 0;
    unsigned int mask = 1;
    bool flag = true;
    while(x != 0 || y !=0 ) {
        if(flag) {
            if(x & 1) ans = ans^mask;
            x = x >> 1;
        }
        else {
            if(y & 1) ans = ans^mask;
            y = y >> 1;
        }
        flag = !flag;
        mask = mask << 1;
    }
    return ans;
}

//attach level to mortonId, return the score of node
int attach(int mortonId, int level, int maxLevel) {
    return mortonId * pow(2,maxLevel) + level;
}

//sequential merge sort
void sortHelper(unsigned int* x, int startIndex, int endIndex, int * index) {
    if(startIndex >= endIndex) return;
    int mid = (startIndex + endIndex)/2;
    sortHelper(x, startIndex, mid, index);
    sortHelper(x, mid+1, endIndex, index);
    //merge
    int* temp = new int[endIndex - startIndex + 1];
    int* tempIndex = new int[endIndex - startIndex + 1];
    int j1, j2;
    j1 = startIndex;
    j2 = mid+1;
    for(int i = 0; i < endIndex - startIndex + 1; i++) {
        if(j1 <= mid && j2 <= endIndex){
            if(x[j1] < x[j2]) {
                temp[i] = x[j1];
                tempIndex[i] = index[j1++];
            }
            else {
                temp[i] = x[j2];
                tempIndex[i] = index[j2++];
            }
        }
        else if(j1 <= mid) {
            temp[i] = x[j1];
            tempIndex[i] = index[j1++];
        }
        else {
            temp[i] = x[j2];
            tempIndex[i] = index[j2++];
        }
    }
    for(int i = 0; i < endIndex - startIndex + 1; i++) {
        x[i+startIndex] = temp[i];
        index[i+startIndex] = tempIndex[i];
    }
    delete[] temp;
    delete[] tempIndex;
}

//sort integer array, index stores the index in the original array
//e.g. x = [1,10,9] after sorting x = [1,9,10], index = [1,3,2]
void sort(unsigned int* x, int lengthOfX, int* & index) {
    index = new int[lengthOfX];
    for(int i = 0; i < lengthOfX; i++)
        index[i] = i;
    sortHelper(x, 0, lengthOfX-1, index);
    //(1) parallel sorting
    //merge sort
    
}

class Node {
public:
    unsigned int mortonId;
    int level;
    int score;
    double anchor_x, anchor_y, squareSide;
    unsigned int* childScore;
    bool leaf;
    Node() {
        childScore = NULL;
        leaf = false;
    }
    Node(double anchor_x, double anchor_y, double squareSide, int level, int maxLevel) {
        double roughx, roughy;
        roughx = anchor_x*pow(2,maxLevel);
        roughy = anchor_y*pow(2,maxLevel);
        mortonId = interleave((unsigned int)roughx, (unsigned int)roughy);
        this->level = level;
        this->anchor_x = anchor_x;
        this->anchor_y = anchor_y;
        this->squareSide = squareSide;
        score = attach(mortonId, level, maxLevel);
        childScore = NULL;
        leaf = false;
    }
    ~Node() {
        if(childScore != NULL)
        delete[] childScore;
    }
    
    //contain: check if this node covers point with coordinate [x,y]
    bool contain(double x, double y) {
        if(x >= anchor_x && y >= anchor_y && x < anchor_x + squareSide && y < anchor_y + squareSide)
            return true;
        else return false;
    }
    //contain: check if node's children-t(t = 0,1,2,3) covers point with coordinate [x,y]
    bool contain(double x, double y, int t) {
        double anchor_x, anchor_y, ss;
        ss = this->squareSide/2;
        switch (t) {
            case 0: anchor_x = this->anchor_x; anchor_y = this->anchor_y; break;
            case 1: anchor_x = this->anchor_x + ss; anchor_y = this->anchor_y; break;
            case 2: anchor_x = this->anchor_x; anchor_y = this->anchor_y + ss; break;
            case 3: anchor_x = this->anchor_x + ss; anchor_y = this->anchor_y + ss;
        }
        if(x >= anchor_x && y >= anchor_y && x < anchor_x + ss && y < anchor_y + ss)
            return true;
        else return false;
    }
};


class NodeInTree:public Node {
public:
    NodeInTree *ch1;
    NodeInTree *ch2;
    NodeInTree *ch3;
    NodeInTree *ch4;
    NodeInTree(double anchor_x, double anchor_y, double squareSide, int level, int maxLevel):Node(anchor_x, anchor_y, squareSide, level, maxLevel) {
        ch1 = NULL;
        ch2 = NULL;
        ch3 = NULL;
        ch4 = NULL;
    }
};

class NodeInArray:public Node {
public:
    unsigned int* childrenIndex;
    vector<double> points_x;
    vector<double> points_y;
    vector<double> points_d;
    double ave_x;
    double ave_y;
    double ave_d;
    
    void copy(Node* p) {
        this->mortonId = p->mortonId;
        this->score = p->score;
        this->level = p->level;
        this->anchor_x  = p->anchor_x;
        this->anchor_y = p->anchor_y;
        this->squareSide = p->squareSide;
        if(p->childScore != NULL) {
            childScore = new unsigned int[4];
            for(int i = 0; i < 4; i++)
                childScore[i] = p->childScore[i];
        }
        else childScore = NULL;
        this->leaf = p->leaf;
        childrenIndex = NULL;
    }
    
    ~NodeInArray() {
        if(childrenIndex != NULL) {
            delete[] childrenIndex;
        }
        if(leaf) {
            points_x.clear();
            points_y.clear();
            points_d.clear();
        }
    }
    
    void searchChildren(NodeInArray* p, int L) {
        if(leaf == true) {
            childrenIndex = NULL;
            return;
        }
        else {
            childrenIndex = new unsigned int[4];
            int l = 0, u = L-1;
            while(l <= u) {
                if(p[(l+u)/2].score < childScore[0])
                    l = (l+u)/2 + 1;
                else if(childScore[3] < p[(l+u)/2].score)
                    u = (l+u)/2 - 1;
                else break;
            }
            int jl, ju;
            jl = l;
            for(int j = 0; j < 4; j++) {
                ju = u;
                while(jl <= ju) {
                    if(p[(jl+ju)/2].score < childScore[j])
                        jl = (jl+ju)/2 + 1;
                    else if(p[(jl+ju)/2].score > childScore[j])
                        ju = (jl+ju)/2 - 1;
                    else { childrenIndex[j] = jl = (jl+ju)/2;
                           break;
                    }
                }
            }
        }
    }
};


//merge two node arrays and delete the original ones. The length of new array is written int *L
NodeInArray *merge(NodeInArray* s1, NodeInArray* s2, int L1, int L2, int* L ) {
    NodeInArray * p = new NodeInArray[L1+L2];
    int length = 0;
    int i = 0, j = 0;
    while(i < L1 && j < L2) {
        if(s1[i].score < s2[j].score)
            p[length++].copy(s1+(i++));
        else if(s1[i].score > s2[j].score)
            p[length++].copy(s2+(j++));
        else {
            //if two nodes are identical but one is non-leaf, we should copy non-leaf node
            if(s1[i].leaf != true) {
                p[length++].copy(s1+i);
            }
            else p[length++].copy(s2+j);
            i++;
            j++;
        }
    }
    while(i < L1) p[length++].copy(s1+(i++));
    while(j < L2) p[length++].copy(s2+(j++));
    *L = length;
    if(s1 != NULL) delete[] s1;
    if(s2 != NULL) delete[] s2;
    return p;
}

class NBody {

public:
    double* points_x; //x coordinate of points, index indicates point id
    double* points_y; //y coordinate of points, index indicates point id
    double* points_d;
    int numOfPoint; //total number of points
    int maxLevel;
    double* ans;
    
    NodeInArray* tree; //quad tree stored in array in order of score ( mortonid + level)
    int numOfNode;
    
    NBody(string file) {
        ifstream input;
        input.open(file.c_str());
        input >> numOfPoint;
        input >> maxLevel;
        points_x = new double[numOfPoint];
        points_y = new double[numOfPoint];
        points_d = new double[numOfPoint];
        for(int i = 0; i < numOfPoint; i++) {
            input >> points_x[i];
            input >> points_y[i];
            input >> points_d[i];
        }
    }
    
    void constructTree() {
        unsigned int* pointMorton = new unsigned int[numOfPoint];
        int * index;
        double roughx, roughy;
     /*   #pragma omp parallel for */
        for(int i = 0; i < numOfPoint; i++) {
            roughx = points_x[i]*pow(2,maxLevel);
            roughy = points_y[i]*pow(2,maxLevel);
            pointMorton[i] = interleave( (unsigned int) roughx, (unsigned int) roughy);
        }
        sort(pointMorton, numOfPoint, index);
        
        int numOfThreads = omp_get_num_threads();
	
        NodeInArray** forest = new NodeInArray*[numOfThreads];
        int* num = new int[numOfThreads]; // indicate the size of each tree in forest
        //parallel construct forests
        #pragma omp parallel
        {
            int i = omp_get_thread_num();
            double anchor_x = 0;
            double anchor_y = 0;
            double squareSide = 1;
            int level = 0;
            int startIndex, endIndex;
            startIndex = numOfPoint/numOfThreads*i;
            endIndex = numOfPoint/numOfThreads*(i+1)-1;
            if(i == numOfThreads - 1) endIndex = numOfPoint - 1;
            
            NodeInTree* root;
            root = constructTreeSeq(pointMorton, startIndex, endIndex, index, anchor_x, anchor_y, squareSide, level);
            int pos = 0;
            num[i] = numOfNodes(root); // calculate the total number of nodes in tree
            forest[i] = new NodeInArray[num[i]];
            //The following function generate nodeArray that is naturally sorted by node's score(mortonId + level)
            tree2Array(root, forest[i], &pos);
            deleteTree(root); //delete original tree, just keep nodeArray
        }
        #pragma omp barrier
        int t = numOfThreads;
        int nt = t;

        //parallel merge forest
        while(t > 1) {
            nt = t/2 + t%2;
            NodeInArray** newForest = new NodeInArray*[nt];
            int* newNum = new int[nt];
            //parallel merge
	    #pragma omp parallel for
            for(int i = 0; i < nt; i++) {
                    newForest[i] = merge(forest[i], forest[t-1-i], num[i], num[t-1-i], newNum+i);
            }
	    #pragma omp barrier
            delete[] forest;
            delete[] num;
            num = newNum;
            forest = newForest;
            t = nt;
        }
        tree = forest[0];
        numOfNode = num[0];

        #pragma omp parallel for
	for(int i = 0; i < numOfNode; i++) {
            if(tree[i].leaf == false)
                tree[i].searchChildren(tree, numOfNode);
        }
        return;
    }
    
    NodeInTree* constructTreeSeq(unsigned int *pointMorton, int startIndex, int endIndex, int* index, double anchor_x, double anchor_y, double squareSide, int level) {
        //construct children
        int i = startIndex;
        NodeInTree* p = new NodeInTree(anchor_x, anchor_y, squareSide, level, maxLevel);
        NodeInTree* ch;
        if(level == maxLevel || startIndex > endIndex) { // no child is created in this case
            p->ch1 = NULL;
            p->ch2 = NULL;
            p->ch3 = NULL;
            p->ch4 = NULL;
            p->leaf = true;
            p->childScore = NULL;
            return p;
        }
        p->childScore = new unsigned int[4];
        double x;
        double y;
        squareSide = squareSide/2;
        //create subtree
        for(int t = 0; t < 4; t++) {
            switch(t) {
                case 0: x = anchor_x; y = anchor_y; break;
                case 1: x = anchor_x + squareSide; y = anchor_y; break;
                case 2: x = anchor_x; y = anchor_y + squareSide; break;
                case 3: x = anchor_x + squareSide; y = anchor_y + squareSide;
            }

            while( i <= endIndex && p->contain(points_x[index[i]], points_y[index[i]], t)) {
                i++;
            }
            ch = constructTreeSeq(pointMorton, startIndex, i-1, index, x, y, squareSide, level+1);
            startIndex = i;
            switch(t) {
                case 0: p->ch1 = ch; p->childScore[0] = ch->score; break;
                case 1: p->ch2 = ch; p->childScore[1] = ch->score; break;
                case 2: p->ch3 = ch; p->childScore[2] = ch->score; break;
                case 3: p->ch4 = ch; p->childScore[3] = ch->score;
            }
        }
        return p;
    }
    
    void deleteTree(NodeInTree* root) {
        if(root == NULL) return;
        else {
            deleteTree(root->ch1);
            deleteTree(root->ch2);
            deleteTree(root->ch3);
            deleteTree(root->ch4);
        }
        delete root;
        return;
    }
    
    void tree2Array(NodeInTree* p, NodeInArray* q, int* pos) {
        if(p == NULL) return;
        (q + *pos)->copy(p);
        *pos += 1;
        tree2Array(p->ch1, q, pos);
        tree2Array(p->ch2, q, pos);
        tree2Array(p->ch3, q, pos);
        tree2Array(p->ch4, q, pos);
    }
    
    int numOfNodes(NodeInTree* root) {
        int ans = 0;
        if(root == NULL) return ans;
        else {
            ans += numOfNodes(root->ch1);
            ans += numOfNodes(root->ch2);
            ans += numOfNodes(root->ch3);
            ans += numOfNodes(root->ch4);
        }
        return ans+1;
    }
    
    void outputPoints() {
        for(int i = 0; i < numOfPoint; i++)
            cout<<points_x[i]<<" "<<points_y[i]<<" "<<points_d[i]<<endl;
    }
    
    void insertPoints() {
    //(2) insert points (points_x, points_y, points_d) into tree;
    // fill in the vectors points_x, points_y, points_d in tree;
    }
    
    void average() {
    //(3) fill in ave_x, ave_y, ave_d in tree
    }
                 
    void evaluate(double x, double y) {
    //(4)
    //implement evaluate function based on tree
    }
};



int main(int argc, char* argv[])
{   string file = "/Users/xinyangyi/test.txt";
    NBody* sol  = new NBody(file);
    sol->constructTree();
    cout<<"Successfully constructing a tree with "<<sol->numOfNode<<" nodes"<<endl;
}