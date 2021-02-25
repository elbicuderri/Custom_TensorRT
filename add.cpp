#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>

void addWithCuda(int *c, int *a, int *b, int size);

int main()
{

	int size = 5;

	////int* a = new int[size] {1, 2, 3, 4, 5};
	////int* b = new int[size] {10, 20, 30, 40, 50};
	////int* c = new int[size];

	int* a = (int*)std::malloc(size * sizeof(int));
	int* b = (int*)std::malloc(size * sizeof(int));
	int* c = (int*)std::malloc(size * sizeof(int));

	for (int i = 0; i < size; i++)
	{
		a[i] = i + 1;
	}

	for (int i = 0; i < size; i++)
	{
		b[i] = 10 * (i + 1);
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

	cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::free(a);
	std::free(b);
	std::free(c);

	////delete[] a;
	////delete[] b;
	////delete[] c;

	return 0;
}
