#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <memory>

void addWithCuda(int *c, int *a, int *b, int size);

int main()
{
	int size = 5;

	int* a = new int[size] {1, 2, 3, 4, 5};
	int* b = new int[size] {10, 20, 30, 40, 50};
	int* c = new int[size];

	//int* a = (int*)malloc(size * sizeof(int));
	//int* b = (int*)malloc(size * sizeof(int));
	//int* c = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < 0; i++) {
		std::cout << a[i] << std::endl;
		//printf("%d\n", a[i]);
	}

	for (int i = 0; i < 0; i++) {
		std::cout << b[i] << std::endl;
		//printf("%d\n", b[i]);
	}

	int* d_a;
	int* d_b;
	int* d_c;

	cudaMalloc(&d_a, size * sizeof(int));
	cudaMalloc(&d_b, size * sizeof(int));
	cudaMalloc(&d_c, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	// Add vectors in parallel.
	addWithCuda(d_c, d_a, d_b, size);
	//cudaThreadSynchronize();

	cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	for (int i = 0; i < size; i++) {
		printf("%d\n", c[i]);
	}

	//for (int i = 0; i < 5; i++) {
	//	std::cout << c[i] << std::endl;
	//}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//free(a);
	//free(b);
	//free(c);

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}
