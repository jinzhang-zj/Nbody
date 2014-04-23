//  Collaborated project with Xinyang yi, Kai Zhong
//  main.cpp
//  implement a parallel alogrithm to solve N body problem 

#include<stdio.h>
#include<fstream>
#include<string>
#include<iostream>
#include<sstream>
#include<math.h>
#include<iomanip>
#include<vector>
#include<omp.h>
//#define numThreads 12
#define PI 3.1415926
using namespace std;

int numThreads;
//interleave two integer x, y
unsigned long interleave(unsigned long x, unsigned long y) {
    unsigned long ans = 0;
    unsigned long mask = 1;
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
unsigned long attach(unsigned long mortonId, int level, int maxLevel) {
    return mortonId * (unsigned long)pow(2,(int)log2((double)maxLevel) + 1) + level;
}

void subMerge(unsigned long* x, int *index, unsigned long* temp, int* tempIndex, int s1, int e1, int s2, int e2, int offset) {
	int j1, j2;
	j1 = s1;
	j2 = s2;
	int length = 0;
	length = e1 - s1 + 1 + e2 - s2 + 1;
	for(int i = offset; i < offset + length; i++) {
        if(j1 <= e1 && j2 <= e2) {
            if(x[j1] < x[j2]) {
                temp[i] = x[j1];
                tempIndex[i] = index[j1++];
            }
            else {
                temp[i] = x[j2];
                tempIndex[i] = index[j2++];
            }
        }
        else if(j1 <= e1) {
            temp[i] = x[j1];
			tempIndex[i] = index[j1++];
        }
        else {
			temp[i] = x[j2];
			tempIndex[i] = index[j2++];
        }
	}
}


int bsearch(unsigned long v, unsigned long* x, int s, int e) {
	int i = s, j = e;
	int mid = (i+j)/2;
	while(i < j) {
		mid = (i+j)/2;
		if(x[mid] == v) return mid;
		if(x[mid] < v) i = mid + 1;
		else j = mid - 1;
	}
	return mid;
}

//merge
void merge(unsigned long* x, int startIndex, int endIndex, int mid, int* index) {
    unsigned long* temp = new unsigned long[endIndex - startIndex + 1];
    int* tempIndex = new int[endIndex - startIndex + 1];
    int j1, j2;
    j1 = startIndex;
    j2 = mid+1;
    int p = numThreads;
    if(startIndex - endIndex + 1 < 2*p)
        subMerge(x,index, temp, tempIndex, startIndex, mid, mid+1, endIndex, 0);
    else {
        int* marker = new int[p+1];
        int* marker1 = new int[p+1];
        int* offset = new int[p];
#pragma omp parallel for
        for(int k = 0; k < p; k++)
            marker[k] = (mid - startIndex + 1)/p*k;
        marker[p] = mid+1;
        marker1[0] = mid+1;
        marker1[p] = endIndex + 1;
#pragma omp parallel for
        for(int k = 1; k <= p-1; k++) {
            marker1[k] = bsearch(x[marker[k]], x, mid+1, endIndex);
        }
        offset[0] = 0;
        for(int k = 1; k < p; k++)
            offset[k] = offset[k-1] + marker[k] - marker[k-1] + marker1[k] - marker1[k-1];
#pragma omp parallel for
        for(int k = 0; k < p; k++)
            subMerge(x, index, temp, tempIndex, marker[k], marker[k+1]-1, marker1[k], marker1[k+1]-1, offset[k]);
    	delete[] marker;
        delete[] marker1;
        delete[] offset;
    }
    if(startIndex - endIndex + 1 < 100) {
        for(int i = 0; i < endIndex - startIndex + 1; i++) {
            x[i+startIndex] = temp[i];
            index[i+startIndex] = tempIndex[i];
        }
    }
    else {
#pragma omp parallel for
        for(int i = 0; i < endIndex - startIndex + 1; i++) {
            x[i+startIndex] = temp[i];
            index[i+startIndex] = tempIndex[i];
        }
    }
    delete[] temp;
    delete[] tempIndex;
}


//parallel merge sort
void sortHelper(unsigned long* x, int startIndex, int endIndex, int * index) {
    if(startIndex >= endIndex) return;
    int mid = (startIndex + endIndex)/2;
    sortHelper(x, startIndex, mid, index);
    sortHelper(x, mid+1, endIndex, index);
    //merge
    merge(x, startIndex, endIndex, mid, index);
}


//sort integer array, index stores the index in the original array
//e.g. x = [1,10,9] after sorting x = [1,9,10], index = [1,3,2]
void sort(unsigned long* x, int lengthOfX, int* & index) {
    index = new int[lengthOfX];
    for(int i = 0; i < lengthOfX; i++)
        index[i] = i;
    sortHelper(x, 0, lengthOfX-1, index);
}

class Node {
public:
    unsigned long mortonId;
    int level;
    int score;
    double anchor_x, anchor_y, squareSide;
    unsigned long* childScore;
    bool leaf;
    Node() {
        childScore = NULL;
        leaf = false;
    }
    Node(double anchor_x, double anchor_y, double squareSide, int level, int maxLevel) {
        double roughx, roughy;
        roughx = anchor_x*pow(2,maxLevel);
        roughy = anchor_y*pow(2,maxLevel);
        mortonId = interleave((unsigned long)roughx, (unsigned long)roughy);
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
    vector<int> points_id;
    int pointNum; //number of points in this node
    double ave_x;
    double ave_y;
    double ave_d;
    NodeInArray(){
        childrenIndex = NULL;
    }
    void copy(Node* p) {
        this->mortonId = p->mortonId;
        this->score = p->score;
        this->level = p->level;
        this->anchor_x  = p->anchor_x;
        this->anchor_y = p->anchor_y;
        this->squareSide = p->squareSide;
        if(p->childScore != NULL) {
            childScore = new unsigned long[4];
            for(int i = 0; i < 4; i++)
                childScore[i] = p->childScore[i];
        }
        else childScore = NULL;
        this->leaf = p->leaf;
        childrenIndex = NULL;
        pointNum = 0;
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

//====================helper function for tree average===========================
// sequential sum of vector
double seqVsum(vector<double> & vec){
    double sum = 0;
    for (int i=0; i< vec.size(); i++)
        sum += vec[i];
    return sum;
};

// sequential sum of product of two vector
double seqV2sum(vector<double> & vec1, vector<double> & vec2){
    double sum = 0;
    for (int i=0; i<vec1.size(); i++)
        sum += vec1[i]*vec2[i];
    return sum;
};

// parallel scan for double array
double* parScan(double* array, int size){
    //printf("entering parScan\n");
    if (size == 1){
        return array;
    }
    
    double lastelement = array[size-1];
    int r=1;
    int t=(int) ceil(size*1.0/r);
    
    //printf ("upward sweep\n");
    do{
        //printf("%d,%d\n",r,t);
#pragma omp parallel for
        for (int i=1; i<t-1; i+=2){
            array[(i+1)*r-1] += array[i*r-1];
        }
        r*=2;
        t=(int)ceil(size*1.0/r);
    }while(t>2);
    
    //printf ("downward sweep\n");
    do{
        t=(int)ceil(size*1.0/r);
#pragma omp parallel for
        for (int i=2; i<t-1; i+=2){
            array[(i+1)*r-1] += array[i*r-1];
        }
        r/=2;
    }while(r>=1);
    
    array[size - 1] = array[size - 2] + lastelement;
    return array;
};

// find subsize of given node i in the tree
int subSize(NodeInArray *tree, int i){
    if (tree[i].leaf){
        return 0;
    }
    bool notleaf = true;
    int current = i;
    while(notleaf){
        current = tree[current].childrenIndex[3];
        notleaf = !tree[current].leaf;
    }
    return current - i;
};
//====================helper function for tree average===========================

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
    
    if(s1 == s2 && s1 != NULL) delete[] s1;
    else {
        if(s1 != NULL) delete[] s1;
        if(s2 != NULL) delete[] s2;
    }
    return p;
}

class NBody {
    
public:
    double* points_x; //x coordinate of points, index indicates point id
    double* points_y; //y coordinate of points, index indicates point id
    double* points_d;
    int numOfPoint; //total number of points
    int maxLevel;
    double* potential;
    
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
        unsigned long* pointMorton = new unsigned long[numOfPoint];
        int * index;
        
#pragma omp parallel for
        for(int i = 0; i < numOfPoint; i++) {
            double roughx = points_x[i]*pow(2,maxLevel);
            double roughy = points_y[i]*pow(2,maxLevel);
            pointMorton[i] = interleave( (unsigned long) roughx, (unsigned long) roughy);
        }
        
        sort(pointMorton, numOfPoint, index);
        int numOfThreads = numThreads;
        
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
        NodeInArray** newForest;
        int* newNum;
        //parallel merge forest
        while(t > 1) {
            
            nt = t/2 + t%2;
            newForest = new NodeInArray*[nt];
            newNum = new int[nt];
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
    
    NodeInTree* constructTreeSeq(unsigned long *pointMorton, int startIndex, int endIndex, int* index, double anchor_x, double anchor_y, double squareSide, int level) {
        //construct children
        int i = startIndex;
        NodeInTree* p = new NodeInTree(anchor_x, anchor_y, squareSide, level, maxLevel);
        NodeInTree* ch;
        if(level == maxLevel || startIndex >= endIndex) { // no child is created in this case
            p->ch1 = NULL;
            p->ch2 = NULL;
            p->ch3 = NULL;
            p->ch4 = NULL;
            p->leaf = true;
            p->childScore = NULL;
            return p;
        }
        p->childScore = new unsigned long[4];
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
    
    void insertAllPoints(){
#pragma omp parallel for
	    for (int i=0;i<numOfPoint;i++){
            //cerr<<"begin insert points"<<endl;
            insertPoints(i,tree[0]);
        }
        //printf("insert done\n");
    }
    
    void insertPoints(int id, NodeInArray& p) {
        //(2) insert points (points_x, points_y, points_d) into tree;
        // fill in the vectors points_x, points_y, points_d in tree;
        if (p.leaf==false){
            for (int c=0;c<4;c++){
                if (p.contain(points_x[id],points_y[id],c)){
                	insertPoints(id,tree[p.childrenIndex[c]]);
                    break;
                }
            }
        }
        else{
#pragma omp critical
            {
                p.points_x.push_back(points_x[id]);
                p.points_y.push_back(points_y[id]);
                p.points_d.push_back(points_d[id]);
                p.pointNum++;
                p.points_id.push_back(id);
            }
        }
    }
    
    void average() {
        //(3) fill in ave_x, ave_y, ave_d in tree
        
        //cout << "call average" << endl;
        //cout << numOfNode << endl;
        // Find subtree size for every node
        // subS: store size of subtree of every node
        int* subS = new int[numOfNode];
#pragma omp parallel for
        for (int i=0; i<numOfNode; i++)
            subS[i] = subSize(tree, i);
        
        //cout << "subtree size done" << endl;
        
        // Build Euler tour
        // I: rank of the first incidence of node in Euler tour
        // O: rank of the last incidence of node in Euler tour
        int* I = new int[numOfNode];
        int* O = new int[numOfNode];
        
#pragma omp parallel for
        for (int i=0; i<numOfNode; i++){
            I[i] = i * 2 - tree[i].level;
            O[i] = I[i] + 2*subS[i];
        }
        
        //cout << "euler tour done" << endl;
        
        // average all points within a node
        // NX: averaged x for all points in a node
        // NY: averaged y for all points in a node
        // ND: sum of d for all points in node
        double* NX = new double[numOfNode];
        double* NY = new double[numOfNode];
        double* ND = new double[numOfNode];
#pragma omp parallel for
        for (int i=0; i<numOfNode; i++){
            if (tree[i].pointNum){
        	    ND[i] = seqVsum(tree[i].points_d);
       		    NX[i] = seqV2sum(tree[i].points_x, tree[i].points_d)/ND[i];
                NY[i] = seqV2sum(tree[i].points_y, tree[i].points_d)/ND[i];
            }else{
                ND[i] = NX[i] = NY[i] = 0;
            }
        }
        
        //cout << "averge within node done" << endl;
        
        // use prefix sum to calculate and store average results.
        // SX: store data (x coordinate) to scan
        // SY: store data (x coordinate) to scan
        // SD: store data (d density) to scan
        double* SX = new double[numOfNode*2];
        double* SY = new double[numOfNode*2];
        double* SD = new double[numOfNode*2];
        
#pragma omp parallel for
        for (int i=0; i<numOfNode*2; i++){
            SX[i] = 0;
            SY[i] = 0;
            SD[i] = 0;
        }
        
        //cout << "initialize data for scan done" << endl;
        
#pragma omp parallel for
        for (int i=0; i<numOfNode; i++){
            SX[I[i]] = NX[i]*ND[i];
            SY[I[i]] = NY[i]*ND[i];
            SD[I[i]] = ND[i];
        }
        
        //cout << "parallel scan start" << endl;
        
        //for (int i=0; i<numOfNode*2; i++){
        //	cout << i << " " <<  SX[i] << " " << SY[i] << " " << SD[i] << endl;
        //}
        
        
        // parallel scan
        SX = parScan(SX,numOfNode*2);
        SY = parScan(SY,numOfNode*2);
        SD = parScan(SD,numOfNode*2);
        
        //for (int i=0; i<numOfNode*2; i++){
        //	cout << i << " " <<  SX[i] << " " << SY[i] << " " << SD[i] << endl;
        //}
        
        //cout << "parallel scan done" << endl;
        
#pragma omp parallel for
        for (int i=0; i<numOfNode; i++){
            tree[i].ave_d = SD[O[i]] - SD[I[i]] + ND[i];
            if (!tree[i].ave_d){
                tree[i].ave_x = 0;
                tree[i].ave_y = 0;
            }else{
                tree[i].ave_x = (SX[O[i]] - SX[I[i]] + NX[i]*ND[i])/tree[i].ave_d;
                tree[i].ave_y = (SY[O[i]] - SY[I[i]] + NY[i]*ND[i])/tree[i].ave_d;
            }
        }
        
        
        //for (int i=0; i<numOfNode; i++){
        //    cout << i << " " << tree[i].ave_x << " " << tree[i].ave_y << " " << tree[i].ave_d <<  endl;
        //}
        
        // cleaning up
        delete [] subS;
        delete [] I;
        delete [] O;
        delete [] NX;
        delete [] NY;
        delete [] ND;
        delete [] SX;
        delete [] SY;
        delete [] SD;
    }
    
    // O(m*N) complexity totally direct sum, where m is number of targets,N is number of sources
    double totalDirect(const double &x, const double &y){
        double bufx,bufy,r;
        double potential = 0.0;
        for (int j=0;j<numOfPoint;j++){
            bufx = x - points_x[j];
            bufy = y - points_y[j];
            r = bufx*bufx+bufy*bufy;
            if (r>10e-100)
                potential -= 1.0/4.0/PI*log(r)*points_d[j];
        }
        return potential;
        
    }
    
    void direct(const double &x, const double &y, const NodeInArray &src, double &potential){
        if (!src.leaf){
            cout<< "not a leaf. can't do direct calculation,please do approximation"<<endl; return;
        }
        double bufx,bufy,r;
        // for (int i=0;i<trg.pointNum;i++){
        for (int j=0;j<src.pointNum;j++){
            bufx = x - src.points_x[j];
            bufy = y - src.points_y[j];
            r = bufx*bufx+bufy*bufy;
            if (r>10e-100)
                potential -= 1.0/4.0/PI*log(r)*src.points_d[j];
        }
        //}
    }
    
    void approximate(const double &x, const double &y, const NodeInArray &src, double &potential){
        double bufx,bufy,r;
        // for (int i=0;i<trg.pointNum;i++){
        bufx = x - src.ave_x;
        bufy = y - src.ave_y;
        r = bufx*bufx+bufy*bufy;
        if (r>10e-100)
            potential -= 1.0/4.0/PI*log(r)*src.ave_d;
        // potential[i] = u;
        // }
    }
    
    
    bool wellSeparated(const double &x, const double &y, const NodeInArray &src){
        //double txmin,txmax,tymin,tymax;
        double sxmin,sxmax,symin,symax;
        /*txmin = trg.anchor_x;
         txmax = trg.anchor_x+trg.squareSide;
         tymin = trg.anchor_y;
         tymax = trg.anchor_y+trg.squareSide;
         */
        sxmin = src.anchor_x-src.squareSide;
        sxmax = src.anchor_x+2*src.squareSide;
        symin = src.anchor_y-src.squareSide;
        symax = src.anchor_y+2*src.squareSide;
        bool flagx, flagy;
        flagx = (x>sxmin) && (x<sxmax);
        flagy = (y>symin) && (y<symax);
        return ! (flagx && flagy);
    }
    
    void evaluate(const double &x, const double &y, const NodeInArray &src,//input
                  double &potential//output
                  ) {
        //(4)
        //implement evaluate function based on tree
        if (wellSeparated(x,y,src)){
            approximate(x,y,src,potential);
            return;
        }
        else{
            if (src.leaf==true){
                if (src.pointNum > 0){
                    direct(x,y,src,potential);
                }
                return;
            }
            else{
                for (int i=0;i<4;i++){
                    evaluate(x,y,tree[src.childrenIndex[i]],potential);
                }
            }
        }
    }
    
    void evaluateAll(){
	    potential = new double[numOfPoint];
#pragma omp parallel for
        for (int i = 0;i<numOfPoint;i++){
            potential[i] = 0.0;
            evaluate(points_x[i], points_y[i], tree[0], potential[i]);
            //            cerr<<"potential for id "<< i<<" is "<<potential[i]<<endl;;
        }
        
    }

    void speed_error(int n, double &err, double &speed){
        int *sample = new int[n];
        int id;
        double *accPot = new double[n];
        double *appPot = new double[n];
        for (int i=0;i<n;i++){
            sample[i] = rand() % numOfPoint;
            appPot[i] = 0.0;
            accPot[i] = 0.0;
        }
        double acctb = omp_get_wtime();
        for (int i = 0;i<n;i++){
            id = sample[i];
            accPot[i] = totalDirect(points_x[id],points_y[id]);
        }
        double accte = omp_get_wtime();
        double accT = accte - acctb;
        double apptb = omp_get_wtime();
        for (int i=0;i<n;i++){
            id = sample[i];
            evaluate(points_x[id],points_y[id],tree[0],appPot[i]);
        }
        double appte = omp_get_wtime();
        double appT = appte - apptb;
        //cout <<std::setprecision(8)<<" approximate Time: "<<appT<<endl<<"accurate Time: "<<accT<<endl;;
        speed = accT/appT;
        err = 0.0;
        double accP2 = 0.0;
        for (int i=0;i<n;i++){
            err += pow((accPot[i]-appPot[i]),2);
            accP2 += accPot[i]*accPot[i];
        }
        err = sqrt(err/accP2);
    }
};

void writeRes(NBody *sol, const char* output){
    FILE *fi = fopen(output,"w");
    for (int i=0;i<sol->numOfPoint;i++)
        fprintf(fi,"%d %f\n",i,sol->potential[i]);
    fclose(fi);
}

int main(int argc, char* argv[])
{  
    if(argc < 3){
        cerr << "Usage: ./test.exe [#threads] [input_file] [output_file]" << endl;
        exit(0);
    }
    numThreads = atoi(argv[1]);
    string file = argv[2]; //input file
    const char* output;
    if (argc < 4)
        output = "result.txt";
    else
        output = argv[3]; //output file
    

    omp_set_num_threads(numThreads);    
    NBody* sol  = new NBody(file);
    double start_time = omp_get_wtime();
    //main algorithm
    sol->constructTree();
    cout<<"Successfully constructing a tree with "<<sol->numOfNode<<" nodes"<<endl;
    sol->insertAllPoints();
    sol->average();
    sol->evaluateAll();
    cout<<endl<<"Total time: "<<omp_get_wtime() - start_time<<endl<<endl;

    //output potential
    writeRes(sol,output);
    cout<<"Potential results are writen in "<<output<<" file"<<endl<<endl;
    //compare Barnes-Hut scheme with direct calculation
    double err = 0.0;
    double speed = 0.0;
    int num_sam = 100;
    sol->speed_error(num_sam,err,speed);
    cout<<"******Compare Barnes-Hut scheme with direct calculation using "<<num_sam<<" sample points*****"<<endl;
    cout<<"relative error: "<<err<<endl<<"speed up: "<<speed<<endl<<endl;
}
