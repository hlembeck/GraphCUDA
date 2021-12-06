#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <random>
#include <stdint.h>
#include <limits.h>

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


typedef class Graph {
public:
	Graph(unsigned int n, unsigned int density);
	~Graph();
	unsigned int getOrder();
	uint64_t getSize();
	/*
	blockify() modifies adjMatrix so that, if C_1,...,C_k are adjacency matrices of the connected components of G, then adjMatrix is a diagonal block matrix with C_1 ... C_k as the diagonal. Thus blockify() alters G, but only up to isomorphism of graphs.

	**IMPORTANT**
	adjMatrix must represent a simple undirected graph.

	Proof of concept:
	Let G=({v_1,...,v_n},E) be a simple undirected graph, and let adj(G) be its adjacency matrix. Since switching two rows i,j and then switching the columns i,j produces an isomorphic graph (we just permuted the vertices while preserving the adjacencies), each pass through compactify produces an isomorphic graph.
	*/
	void blockify();
	void forceSimple();
	void print();
private:
	unsigned int order;
	unsigned char* adjMatrix;
	unsigned int compactify(unsigned int row, unsigned int start_col);
} Graph;