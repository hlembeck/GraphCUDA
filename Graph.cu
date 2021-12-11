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

void Graph::swapVertices(unsigned int i, unsigned int j) {
	unsigned char* swapper = new unsigned char[order * order];
	unsigned char* dev_A = 0;
	unsigned char* dev_B = 0;
	unsigned char* dev_Out = 0;

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_A, order * order);
	cudaMalloc((void**)&dev_B, order * order);
	cudaMalloc((void**)&dev_Out, order * order);

	memset(swapper, 0, order * order);
	for (unsigned int row = 0; row < order; row++) {
		swapper[row * (order + 1)] = 1;
	}

	swapper[i * order + j] = 1;
	swapper[j * order + i] = 1;
	swapper[i * (order + 1)] = 0;
	swapper[j * (order + 1)] = 0;

	cudaMemcpy(dev_A, adjMatrix, order * order, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, swapper, order * order, cudaMemcpyHostToDevice);

	dim3 numBlocks(ceiling_div_16(order), ceiling_div_16(order));
	dim3 numThreadsPerBlock(16, 16);
	MatMult_UChar <<<numBlocks, numThreadsPerBlock >>> (dev_A, dev_B, dev_Out, order);
	cudaDeviceSynchronize();
	cudaMemcpy(dev_A, dev_Out, order * order, cudaMemcpyDeviceToDevice);

	memset(swapper, 0, order * order);
	for (unsigned int row = 0; row < order; row++) {
		swapper[row * (order + 1)] = 1;
	}

	swapper[i * order + j] = 1;
	swapper[j * order + i] = 1;
	swapper[i * (order + 1)] = 0;
	swapper[j * (order + 1)] = 0;
	cudaMemcpy(dev_B, swapper, order * order, cudaMemcpyHostToDevice);

	MatMult_UChar <<<numBlocks, numThreadsPerBlock >>> (dev_B, dev_A, dev_Out, order);
	cudaDeviceSynchronize();
	cudaMemcpy(adjMatrix, dev_Out, order * order, cudaMemcpyDeviceToHost);


	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_Out);
	delete[] swapper;
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

void min_heapify(unsigned int* arr, unsigned int* weights, unsigned int i, unsigned int heapsize){
	unsigned int left = (i << 1) + 1;
	unsigned int right = (i + 1) << 1;
	unsigned int min = i;
	if (left < heapsize && weights[arr[left]] < weights[arr[i]])
		min = left;
	if (right < heapsize && weights[arr[right]] < weights[arr[min]])
		min = right;
	if (min != i) {
		unsigned int temp = arr[i];
		arr[i] = arr[min];
		arr[min] = temp;

		min_heapify(arr, weights, min, heapsize);
	}
	return;
}

void build_min_heap(unsigned int* arr, unsigned int* weights, unsigned int len) {
	for (unsigned int i = (len >> 1); i > 0; i--) {
		min_heapify(arr, weights, i, len);
	}
	min_heapify(arr, weights, 0, len);
}

void heapsort(unsigned int* arr, unsigned int* weights, unsigned int len) {
	if (len == 0)
		return;
	build_min_heap(arr, weights, len);

	for (unsigned int i = len - 1; i > 0; i--) {
		unsigned int temp = arr[0];
		arr[0] = arr[i];
		arr[i] = temp;
		min_heapify(arr, weights, 0, i);
	}
}

void fill_map(unsigned int* map, unsigned int *curr, unsigned int i, unsigned char* visited, unsigned int* degrees, unsigned int** adjList) {
	node* head = new node;
	node* tail = head;
	node* temp;
	head->val = i;
	head->next = NULL;
	while (head) {
		i = head->val;
		map[*curr] = i;
		(*curr)++;
		for (unsigned int j = 0; j < degrees[i]; j++) {
			if (visited[adjList[i][j]] == 0) {
				visited[adjList[i][j]] = 1;
				temp = new node;
				temp->val = adjList[i][j];
				temp->next = NULL;
				tail->next = temp;
				tail = temp;
			}
		}
		temp = head;
		head = head->next;
		delete temp;
	}
}

Permutation Graph::pack() {
	unsigned int* degrees = new unsigned int[order];
	unsigned int** adjList = new unsigned int* [order];
	unsigned int* sortedVertices = new unsigned int[order];
	unsigned int* map = new unsigned int[order];
	unsigned int temp;
	unsigned char* visited = new unsigned char[order];
	unsigned int curr = 0;
	memset(visited, 0, order);

	for (unsigned int i = 0; i < order; i++) {
		temp = 0;
		for (unsigned int j = 0; j < order; j++) {
			temp += adjMatrix[i * order + j];
		}
		adjList[i] = new unsigned int[temp];
		degrees[i] = temp;
		temp = 0;
		for (unsigned int j = 0; j < order; j++) {
			if (adjMatrix[i * order + j])
				adjList[i][temp++] = j;
		}
		sortedVertices[i] = i;
	}
	heapsort(sortedVertices, degrees, order);

	for (unsigned int i = 0; i < order; i++) {
		heapsort(adjList[i], degrees, degrees[i]);
	}

	node* head = NULL;
	while (curr<order) {
		unsigned int i = 0;
		for (i; i < order; i++) {
			if (visited[sortedVertices[i]] == 0)
				break;
		}
		if (i == order)
			break;
		visited[sortedVertices[i]] = 1;
		fill_map(map, &curr, sortedVertices[i], visited, degrees, adjList);
	}
	delete[] degrees;
	return Permutation(map, order);
}

void Graph::relabel(Permutation* s) {
	unsigned int* map = new unsigned int[order];
	for (unsigned int i = 0; i < order; i++) {
		map[i] = i;
	}

	node_LL* head = s->getTranspositions();
	node_LL* temp = head;

	while (temp) {
		swapVertices(map[temp->a], map[temp->b]);
		unsigned int t = map[temp->a];
		map[temp->a] = map[temp->b];
		map[temp->b] = t;
		temp = temp->next;
	}

	while (head) {
		temp = head;
		head = head->next;
		delete temp;
	}
}