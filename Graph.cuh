#include "Combinatorics.cuh"

typedef class CompactGraph {
public:
	CompactGraph(unsigned int n);
	~CompactGraph();
	unsigned int getOrder();
private:
	/*
	adjMatrix to be viewed as an array of unsigned chars, where the number of rows and columns is 8*size. Each character is viewed as an array of bits, indicating the relevant adjacency data.

	For example, if size=1 then adjMatrix is viewed as the column vector [c_0,...,c_7]^T, and if we write c_i = e_{i0} + ... + e_{i7}*2^7 then we have (the data for) the matrix

	e_{00} ... e_{07}
	  .   .       .
	  .     .     .
	  .       .   .
	e_{70} ... e_{77}


	The adjacency data for vertices i->j is accessed by ( *(adjMatrix + size*i + floor(i/8)) >> i%8) & 1. Thus the data of the matrix at (row,col)=(i,j) is the data of whether there is an edge from i->j.
	*/
	unsigned char* adjMatrix;
	unsigned int order;
	bool simple;
	bool directed;
} CompactGraph;

typedef struct vertexData {
	unsigned int degree;
	unsigned int index;
	unsigned int* neighbors;
} vertexData;

typedef class Graph {
public:
	Graph(unsigned int n, unsigned int density);
	Graph(unsigned int n, double block_frequency, unsigned int block_size, double density);
	~Graph();
	unsigned int getOrder();
	uint64_t getSize();
	Permutation pack();
	//Pushes isolated vertices to the end of the vertex list and returns the index of the first isolated vertex.
	void swapVertices(unsigned int i, unsigned int j);
	void forceSimple();
	void relabel(Permutation* s);
	void print();
	void writeToFileJSON(char* fName);
private:
	unsigned int order;
	unsigned char* adjMatrix;
} Graph;

void heapsort(unsigned int* arr, unsigned int* weights, unsigned int len);