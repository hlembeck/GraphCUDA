#include "Graph.cuh"

int main() {
	Graph g = Graph(100, .075, 5, .2);
	g.forceSimple();
	g.writeToFileJSON("C:\\Users\\Henry\\Desktop\\Coding\\Program_Graph\\graph_JSON\\initialGraph.json");
	g.print();
	printf("\nSize: %llu\n", g.getSize());
	//g.blockify();
	g.pack();
	printf("\nSize: %llu\n", g.getSize());
	g.print();
	g.writeToFileJSON("C:\\Users\\Henry\\Desktop\\Coding\\Program_Graph\\graph_JSON\\finalGraph.json");

	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}