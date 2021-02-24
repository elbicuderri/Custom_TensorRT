#include <math.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "NvInfer.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

__global__ void my_convolution_kernel(float *output, float *input, float *weight, float *bias,
	int batch, int in_channel, int out_channel, int in_height, int in_width,
	int kernel_height, int kernel_width, int pad_height, int pad_width,
	int stride_height, int stride_width, int total_size, cudaStream_t stream)
{
	//int N = batch;
	int C = in_channel;
	int K = out_channel;
	int H = in_height;
	int W = in_width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;
	int t_size = total_size;

	int P = ((H + 2 * pH - kH) / sH) + 1;
	int Q = ((W + 2 * pW - kW) / sW) + 1;

	//tid : thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= t_size)
		return;

	//q_idx : output w-index
	int q_idx = tid % Q;
	int idx = tid / Q;

	//p_idx : output h-index
	int p_idx = idx % P;
	idx /= P;

	//k_idx : output channel-index
	int k_idx = idx % K;

	//n_idx : output batch-index
	int n_idx = idx / K;

	//output(n_idx, k_idx, p_idx, q_idx)

	float sum = 0.0f;
	for (int c_idx = 0; c_idx < C; c_idx++)
	{
		for (int kh_idx = 0; kh_idx < kH; kh_idx++)
		{
			int h_idx = p_idx * sH + kh_idx - pH;
			if (h_idx >= 0 && h_idx < H)
			{
				for (int kw_idx = 0; kw_idx < kW; kw_idx++)
				{
					int w_idx = q_idx * sW + kw_idx - pW;
					if (w_idx >= 0 && w_idx < W)
					{
						int input_index = n_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx;
						int weight_index = k_idx * C * kH * kW + c_idx * kH * kW + kh_idx * kW + kw_idx;
						sum += input[input_index] * weight[weight_index];
					}
				}
			}
		}
	}
	sum += bias[k_idx];
	output[tid] = sum;

}

void my_convolution_func(float* output, float* input, float *weight, float *bias,
	int batch, int out_channel, Dims dim_input, Dims dim_kernel,
	Dims dim_pad, Dims dim_stride, cudaStream_t stream)
{

	int N = batch;
	int C = dim_input.d[0];
	int H = dim_input.d[1];
	int W = dim_input.d[2];
	int K = out_channel;
	int kH = dim_kernel.d[0];
	int kW = dim_kernel.d[1];
	int pH = dim_pad.d[0];
	int pW = dim_pad.d[1];
	int sH = dim_stride.d[0];
	int sW = dim_stride.d[1];

	int P = ((H + 2 * pH - kH) / sH) + 1;
	int Q = ((W + 2 * pW - kW) / sW) + 1;

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * K * P * Q;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	//std::cout << "N of d_inputs is " << N << std::endl;
	//std::cout << "C of d_inputs is " << C << std::endl;
	//std::cout << "H of d_inputs is " << H << std::endl;
	//std::cout << "W of d_inputs is " << W << std::endl;

	//std::cout << "K  is " << K << std::endl;
	//std::cout << "P  is " << Q << std::endl;
	//std::cout << "Q  is " << Q << std::endl;

	//std::cout << "TOTAL_SIZE  is " << TOTAL_SIZE << std::endl;
	//std::cout << "THREADS_PER_BLOCK  is " << THREADS_PER_BLOCK << std::endl;
	//std::cout << "NUMBER_OF_BLOCKS  is " << NUMBER_OF_BLOCKS << std::endl;

	my_convolution_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > ((float*)output, (float*)input, (float*)weight, (float*)bias,
		N, C, K, H, W,
		kH, kW, pH, pW, sH, sW, TOTAL_SIZE, stream);

}

