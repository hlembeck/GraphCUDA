#include "Graph.cuh"

CompactGraph::CompactGraph(unsigned int n) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(0, 255);

	order = n;

	adjMatrix = new unsigned char[8*(uint64_t)n*(uint64_t)n];
	for (uint64_t i = 0; i < 8*(uint64_t)n*(uint64_t)n; i++) {
		adjMatrix[i] = (unsigned char)distrib(gen);
	}

	simple = false;
	directed = true;
}

CompactGraph::~CompactGraph(){
	delete adjMatrix;
}

unsigned int CompactGraph::getOrder() {
	return order;
}

Graph::Graph(unsigned int n, unsigned int density) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(1, n*n);

	order = n;
	unsigned int t = 0;
	adjMatrix = new unsigned char[(uint64_t)order * (uint64_t)order];
	for (uint64_t i = 0; i < (uint64_t)order * (uint64_t)order; i++) {
		t = (unsigned int)distrib(gen);
		if (t > n*n-density) {
			adjMatrix[i] = 1;
		}
		else {
			adjMatrix[i] = 0;
		}
	}
}

Graph::~Graph() {
	delete[] adjMatrix;
}

unsigned int Graph::getOrder() {
	return order;
}

uint64_t Graph::getSize() {
	uint64_t ret = 0;
	for (unsigned int i = 0; i < order; i++) {
		for (unsigned int j = i + 1; j < order; j++) {
			if (adjMatrix[i * order + j])
				ret++;
		}
	}
	return ret;
}

__global__ void MatMult_UChar(unsigned char* A, unsigned char* B, unsigned char* Out, unsigned int size) {
	unsigned int row = 16*blockIdx.x + threadIdx.x;
	unsigned int col = 16*blockIdx.y + threadIdx.y;
	if (row >= size || col >= size)
		return;

	unsigned char temp = 0;
	for (unsigned int i = 0; i < size; i++) {
		temp += A[row * size + i] * B[i * size + col];
	}
	Out[row * size + col] = temp;
	return;
}

unsigned int ceiling_div_16(unsigned int n) {
	return (n & 15 ? (n >> 4) + 1 : n >> 4);
}

void Graph::blockify() {
	unsigned int row = 0;
	unsigned int col = 1;
	while (row < order-1) {
		col = compactify(row,col);
		row++;
		//printf("row: %d, col: %d\n", row, col);
	}
	return;
}

unsigned int Graph::compactify(unsigned int row, unsigned int start_col) {
	unsigned char* dev_A = 0;
	unsigned char* dev_B = 0;
	unsigned char* dev_Out = 0;
	//unsigned int col = row+1;
	unsigned int col = start_col;
	bool trivial = true;
	unsigned char* switcher = new unsigned char[(uint64_t)order * (uint64_t)order];

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_A, order * order);
	cudaMalloc((void**)&dev_B, order * order);
	cudaMalloc((void**)&dev_Out, order * order);

	while (col < order-1) {
		while (col < order && adjMatrix[row * order + col]) {
			col++;
		}

		memset(switcher, 0, (uint64_t)order * (uint64_t)order);
		cudaMemset(dev_Out, 0, (uint64_t)order * (uint64_t)order);
		cudaMemcpy(dev_A, adjMatrix, (uint64_t)order * (uint64_t)order, cudaMemcpyHostToDevice);

		//Set switcher to elementary matrix that swaps first column possible.
		for (uint64_t i = 0; i < order; i++) {
			switcher[((uint64_t)order + 1) * i] = 1;
		}
		for (uint64_t i = col + 1; i < order; i++) {
			if (adjMatrix[(uint64_t)row * order + i]) {
				switcher[((uint64_t)order + 1) * col] = 0;
				switcher[((uint64_t)order + 1) * i] = 0;
				switcher[col * order + i] = 1;
				switcher[i * order + col] = 1;
				col++;
				trivial = false;
				break;
			}
		}
		/*for (unsigned int i = 0; i < order; i++) {
			for (unsigned int j = 0; j < order; j++) {
				printf(" %d", switcher[i * order + j]);
			}
			std::cout << "\n";
		}
		std::cout << "\n";*/

		if (trivial) {
			break;
		}

		cudaMemcpy(dev_B, switcher, (uint64_t)order * (uint64_t)order, cudaMemcpyHostToDevice);
		dim3 numBlocks(ceiling_div_16(order), ceiling_div_16(order));
		dim3 numThreadsPerBlock(16, 16);
		MatMult_UChar<<<numBlocks, numThreadsPerBlock>>>(dev_A, dev_B, dev_Out, order);
		cudaDeviceSynchronize();
		cudaMemcpy(dev_A, dev_Out, order * order, cudaMemcpyDeviceToDevice);
		MatMult_UChar<<<numBlocks, numThreadsPerBlock >>>(dev_B, dev_A, dev_Out, order);
		cudaDeviceSynchronize();
		cudaMemcpy(adjMatrix, dev_Out, (uint64_t)order * (uint64_t)order, cudaMemcpyDeviceToHost);
		trivial = true;
	}
	/*
	for (unsigned int i = 0; i < order; i++) {
		for (unsigned int j = 0; j < order; j++) {
			printf(" %d", switcher[i * order + j]);
		}
		std::cout << "\n";
	}
	*/

	/*for (unsigned int i = row + 1; i < order; i++) {
		if (adjMatrix[row * order + i] == 0)
			return i+1;
	}*/

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_Out);
	delete[] switcher;
	return col;
}

void Graph::forceSimple() {
	for (unsigned int i = 0; i < order; i++) {
		for (unsigned int j = i; j < order; j++) {
			if (i == j)
				adjMatrix[i * order + j] = 0;
			else
				adjMatrix[j * order + i] = adjMatrix[i * order + j];
		}
	}
	return;
}

void Graph::print() {
	std::cout << "\n---- Printing Graph ----\n\n";
	for (unsigned int i = 0; i < order; i++) {
		for (unsigned int j = 0; j < order; j++) {
			printf(" %d", adjMatrix[i * order + j]);
		}
		std::cout << "\n";
	}
	std::cout << "\n------------------------\n\n";
}