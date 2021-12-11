#include "Combinatorics.cuh"

Stack::Stack(unsigned int val) {
	struct node* head = new struct node;
	head->next = NULL;
	head->val = val;
}

Stack::~Stack() {
	node* temp;
	while (head) {
		temp = head;
		head = head->next;
		delete temp;
	}
}

void Stack::push(unsigned int val) {
	struct node* temp = new struct node;
	temp->val = val;
	temp->next = head;
	head = temp;
}

void Stack::pop(unsigned int *res) {
	struct node* temp = head;
	head = head->next;
	*res = temp->val;
	delete temp;
}

FIFO::FIFO(unsigned int val) {
	struct node* head = new struct node;
	head->val = val;
	head->next = NULL;
	tail = head;
}

FIFO::~FIFO() {
	while (head) {
		tail = head;
		head = head->next;
		delete tail;
	}
}

void FIFO::push(unsigned int val) {
	struct node* temp = new struct node;
	temp->next = NULL;
	temp->val = val;
	if (tail)
		tail->next = temp;
	else
		head = temp;
	tail = temp;
}

void FIFO::pop(unsigned int* res) {
	struct node* temp = head;
	head = head->next;
	*res = temp->val;
	delete temp;
}

Permutation::Permutation(unsigned int* mapping, unsigned int len) {
	map = mapping;
	n = len;
	generateCycles();
}

Permutation::Permutation(unsigned int len) {
	n = len;
	map = new unsigned int[len];
	for (unsigned int i = 0; i < len; i++) {
		map[i] = i;
	}
	generateCycles();
}

void Permutation::randomize() {
	delete[] cycleLengths;
	for (unsigned int i = 0; i < cycleCount; i++) {
		delete[] cycleDecomp[i];
	}
	delete[] cycleDecomp;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(0, n - 1);
	unsigned int j, temp;
	for (unsigned int i = 0; i < n - 1; i++) {
		j = (unsigned int)distrib(gen);
		temp = map[i];
		map[i] = map[j];
		map[j] = temp;
	}
	generateCycles();
}

Permutation::~Permutation() {
	for (unsigned int i = 0; i < cycleCount; i++) {
		delete[] cycleDecomp[i];
	}
	delete[] cycleDecomp;
	delete[] cycleLengths;
	delete[] map;
}

unsigned int* Permutation::getMap() {
	return map;
}

unsigned int** Permutation::getCycleDecomp(unsigned int** lengths) {
	*lengths = cycleLengths;
	return cycleDecomp;
}

unsigned int Permutation::getOrder() {
	unsigned int order = 1;
	for (unsigned int i = 0; i < cycleCount; i++) {
		order = (order < cycleLengths[i] ? cycleLengths[i] : order);
	}
	return order;
}

void Permutation::swap(unsigned int i, unsigned int j) {
	unsigned int temp = map[i];
	map[i] = map[j];
	map[j] = temp;
}

void Permutation::print() {
	printf("\n---- Printing Permutation ----\n\n");
	for (unsigned int i = 0; i < n; i++) {
		printf(" %d", i);
	}
	printf("\n");
	for (unsigned int i = 0; i < n; i++) {
		printf(" %d", map[i]);
	}
	printf("\n\n------------------------------\n");
}

void Permutation::generateCycles() {
	unsigned int numCycles = 0;
	unsigned char* visited = new unsigned char[n];

	memset(visited, 0, n);
	for (unsigned int i = 0; i < n; i++) {
		if (visited[i])
			continue;
		visited[i] = 1;
		if (map[i] == i)
			continue;
		unsigned int curr = map[i];
		numCycles++;
		while (curr != i) {
			visited[curr] = 1;
			curr = map[curr];
		}
	}

	cycleDecomp = new unsigned int* [numCycles];
	cycleLengths = new unsigned int[numCycles];
	cycleCount = numCycles;

	memset(visited, 0, n);
	//numCycles now is the index used to write into cycleDecomp
	numCycles = 0;
	for (unsigned int i = 0; i < n; i++) {
		if (visited[i])
			continue;
		visited[i] = 1;
		if (map[i] == i)
			continue;
		unsigned int curr = map[i];
		unsigned int temp = 1;

		while (curr != i) {
			visited[curr] = 1;
			curr = map[curr];
			temp++;
		}
		cycleDecomp[numCycles] = new unsigned int[temp];
		cycleLengths[numCycles] = temp;
		temp = 1;
		curr = map[i];
		cycleDecomp[numCycles][0] = i;
		while (curr != i) {
			cycleDecomp[numCycles][temp] = curr;
			temp++;
			curr = map[curr];
		}
		numCycles++;
	}
	delete[] visited;
}

void Permutation::printCycles() {
	printf("\n---- Printing Cycle Decomposition ----\n\n");
	for (unsigned int i = 0; i < cycleCount; i++) {
		printf(" ( ");
		for (unsigned int j = 0; j < cycleLengths[i]; j++) {
			printf("%u ", cycleDecomp[i][j]);
		}
		printf(")");
	}
	printf("\n\n--------------------------------------\n");
}

node_LL* Permutation::getTranspositions() {
	if (cycleCount == 0)
		return NULL;
	node_LL* head = new node_LL;
	node_LL* temp = head;
	head->a = cycleDecomp[0][0];
	head->b = cycleDecomp[0][1];
	node_LL* prev = NULL;
	for (unsigned int j = 2; j < cycleLengths[0]; j++) {
		prev = temp;
		temp = new node_LL;
		prev->next = temp;
		temp->a = cycleDecomp[0][0];
		temp->b = cycleDecomp[0][j];
		temp->next = NULL;
	}
	for (unsigned int i = 1; i < cycleCount; i++) {
		for (unsigned int j = 1; j < cycleLengths[i]; j++) {
			prev = temp;
			temp = new node_LL;
			prev->next = temp;
			temp->a = cycleDecomp[i][0];
			temp->b = cycleDecomp[i][j];
			temp->next = NULL;
		}
	}
	return head;
}

void Permutation::printTransposition(node_LL* v) {
	if (v->next)
		printTransposition(v->next);
	printf(" ( %d %d )", v->a, v->b);
}

void Permutation::printTranspositions() {
	node_LL* head = getTranspositions();

	printf("\n---- Printing Transpositions ----\n\n");
	printTransposition(head);
	printf("\n\n---------------------------------\n");

	node_LL* temp;
	while (head) {
		temp = head->next;
		delete head;
		head = temp;
	}
}