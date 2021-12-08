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


//fills mat with a matrix taking values 0,1 in such a way that mat takes the form of a block matrix with blocks of size block_size, each of which is a random matrix whose fraction of 1's is ~density.
void randomBlockMatrix(unsigned char *mat, double block_frequency, unsigned int block_size, double density, unsigned int size) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distrib(0.0,1.0);
	memset(mat, 0, size * size);
	for (unsigned int i = 0; i < size; i+=block_size) {
		for (unsigned int j = 0; j < size; j += block_size) {

			if (distrib(gen) < block_frequency) {
				for (unsigned int k = 0; k < block_size; k++) {
					for (unsigned int l = 0; l < block_size; l++) {
						if (distrib(gen) < density && i+k<size && j+l<size)
							mat[(i+k) * size + j + l] = 1;
					}
				}
			}

		}
	}
	return;
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

Graph::Graph(unsigned int n, double block_frequency, unsigned int block_size, double density) {
	order = n;
	unsigned int t = 0;
	adjMatrix = new unsigned char[(uint64_t)order * (uint64_t)order];
	randomBlockMatrix(adjMatrix, block_frequency, block_size, density, n);
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



//for (unsigned int k = 0; k < order; k++) {
//	for (unsigned int l = 0; l < order; l++) {
//		printf(" %d", swapper[k * order + l]);
//	}
//	printf("\n");
//}
//printf("\n");
void Graph::pack() {

	/*
	head to point to the head of a linked list that partitions the vertices of G so that the algorithm only swaps vertices in the same part. If the linked list describes an array [i_0,...,i_k], then the algorithm is to swap columns or rows of indices j_1,j_2 only if i_l < j_1 < j_2 < i_{l+1} for some 0<=l<k.
	*/
	node_LL* head = new node_LL;
	head->val = 1;
	node_LL* temp = new node_LL;
	head->next = temp;
	temp->next = NULL;
	temp->val = order;

	unsigned char* swapper = new unsigned char[order * order];
	unsigned char* dev_A = 0;
	unsigned char* dev_B = 0;
	unsigned char* dev_Out = 0;

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_A, order * order);
	cudaMalloc((void**)&dev_B, order * order);
	cudaMalloc((void**)&dev_Out, order * order);

	for (unsigned int row = 0; row < order; row++) {

		for (temp = head; temp->next; temp = temp->next) {
			if (temp->val < row + 1)
				continue;
			unsigned int stop = temp->next->val;
			for (unsigned int i = temp->val; i < stop; i++) {
				if (adjMatrix[row * order + i])
					continue;

				for (unsigned int j = i + 1; j < stop; j++) {
					if (adjMatrix[row * order + j]) {
						memset(swapper, 0, order * order);
						for (unsigned int i = 0; i < order; i++) {
							swapper[i * (order + 1)] = 1;
						}

						swapper[i * order + j] = 1;
						swapper[j * order + i] = 1;
						swapper[i * (order + 1)] = 0;
						swapper[j * (order + 1)] = 0;

						cudaMemcpy(dev_A, adjMatrix, order * order, cudaMemcpyHostToDevice);
						cudaMemcpy(dev_B, swapper, order * order, cudaMemcpyHostToDevice);

						dim3 numBlocks(ceiling_div_16(order), ceiling_div_16(order));
						dim3 numThreadsPerBlock(16, 16);
						MatMult_UChar <<<numBlocks, numThreadsPerBlock>>> (dev_A, dev_B, dev_Out, order);
						cudaDeviceSynchronize();
						cudaMemcpy(dev_A, dev_Out, order * order, cudaMemcpyDeviceToDevice);
						MatMult_UChar <<<numBlocks, numThreadsPerBlock>>> (dev_B, dev_A, dev_Out, order);
						cudaDeviceSynchronize();
						cudaMemcpy(adjMatrix, dev_Out, order * order, cudaMemcpyDeviceToHost);
						goto end;
					}
				}

				if (i != temp->val) {
					node_LL* temp2 = new node_LL;
					temp2->val = i;
					temp2->next = temp->next;
					temp->next = temp2;
					temp = temp->next;
					goto final;
				}

				end:
			}
			final:
		}
		//print();
	}


	/*
	//Set first row
	for (unsigned int i = 1; i < order-1; i++) {
		if (adjMatrix[i])
			continue;

		memset(swapper, 0, order*order);
		for (unsigned int i = 0; i < order; i++) {
			swapper[i * (order + 1)] = 1;
		}

		for (unsigned int j = i + 1; j < order; j++) {
			if (adjMatrix[j]) {
				swapper[i * order + j] = 1;
				swapper[j * order + i] = 1;
				swapper[i * (order + 1)] = 0;
				swapper[j * (order + 1)] = 0;

				cudaMemcpy(dev_A, adjMatrix, order * order, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_B, swapper, order * order, cudaMemcpyHostToDevice);

				dim3 numBlocks(ceiling_div_16(order), ceiling_div_16(order));
				dim3 numThreadsPerBlock(16, 16);
				MatMult_UChar << <numBlocks, numThreadsPerBlock >> > (dev_A, dev_B, dev_Out, order);
				cudaDeviceSynchronize();
				cudaMemcpy(dev_A, dev_Out, order * order, cudaMemcpyDeviceToDevice);
				MatMult_UChar << <numBlocks, numThreadsPerBlock >> > (dev_B, dev_A, dev_Out, order);
				cudaDeviceSynchronize();
				cudaMemcpy(adjMatrix, dev_Out, order * order, cudaMemcpyDeviceToHost);
				break;
			}
		}
	}
	*/

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_Out);
	
	while (head) {
		temp = head;
		head = head->next;
		delete temp;
	}
	delete[] swapper;
}


void Graph::blockify() {
	unsigned int row = 0;
	unsigned int col = 0;
	unsigned int temp;
	while (row < order-1) {
		col = compactify(row,col);
		row++;
	}
	return;
}

unsigned int Graph::compactify(unsigned int row, unsigned int start_col) {
	unsigned char* dev_A = 0;
	unsigned char* dev_B = 0;
	unsigned char* dev_Out = 0;
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

void Graph::writeToFileJSON(char* fName) {
	std::ofstream file;
	file.open(fName);
	file << "{\"vertices\":" << order << ",\"edges\":[";

	bool nonempty = false;
	for (unsigned int i = 0; i < order; i++) {
		for (unsigned int j = i + 1; j < order; j++) {
			if (adjMatrix[i * order + j]) {
				if (nonempty)
					file << ",";
				file << "[" << i << "," << j << "]";
				nonempty = true;
			}
		}
	}

	file << "]}";
	file.close();
}