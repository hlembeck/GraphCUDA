#include "Graph.cuh"

int main() {
	Graph g = Graph(10,20);
	g.forceSimple();
	g.print();
	printf("\nSize: %llu\n", g.getSize());
	g.blockify();
	g.print();
	printf("\nSize: %llu\n", g.getSize());
	return 0;
}