#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <random>
#include <stdint.h>
#include <limits.h>

typedef struct node_LL {
	struct node_LL* next;
	unsigned int a;
	unsigned int b;
} node_LL;

typedef struct node {
	struct node* next;
	unsigned int val;
} node;

typedef class FIFO {
public:
	FIFO(unsigned int val);
	~FIFO();
	void push(unsigned int val);
	void pop(unsigned int* res);
private:
	struct node {
		struct node* next;
		unsigned int val;
	};
	struct node* head;
	struct node* tail;
} FIFO;

typedef class Stack {
public:
	Stack(unsigned int val);
	~Stack();
	void push(unsigned int val);
	void pop(unsigned int *res);
private:
	struct node {
		struct node* next;
		unsigned int val;
	};
	struct node* head;
} Stack;

typedef class Permutation {
public:
	Permutation(unsigned int* mapping, unsigned int len);
	Permutation(unsigned int n);
	~Permutation();
	unsigned int* getMap();
	//If map represents the permutation s, then swap(i,j) corresponds to setting s=s(i j).
	void swap(unsigned int i, unsigned int j);
	void print();
	void printCycles();
	void randomize();
	void printTranspositions();
	unsigned int** getCycleDecomp(unsigned int** lengths);
	node_LL* getTranspositions();
	unsigned int getOrder();
private:
	unsigned int n;
	unsigned int* map;
	unsigned int** cycleDecomp;
	unsigned int* cycleLengths;
	unsigned int cycleCount;
	void generateCycles();
	void printTransposition(node_LL* v);
} Permutation;