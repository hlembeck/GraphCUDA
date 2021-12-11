#include "Graph.cuh"

int main() {
	Graph g = Graph(100, .1, 5, .5);
	g.forceSimple();
	g.writeToFileJSON("C:\\Users\\Henry\\Desktop\\Coding\\Program_Graph\\graph_JSON\\initialGraph.json");
	g.print();

	Permutation s = g.pack();
	g.relabel(&s);

	g.print();
	g.writeToFileJSON("C:\\Users\\Henry\\Desktop\\Coding\\Program_Graph\\graph_JSON\\finalGraph.json");

	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}