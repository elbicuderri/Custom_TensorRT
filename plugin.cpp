/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <cassert>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "EntropyCalibrator.h"

#include <cstdlib>
#include <time.h>
#include <string>
#include <algorithm>
#include <vector>
#include <limits.h>
#include <float.h>

void my_convolution_func(float* output, float* input, float *weight, float *bias,
	int batch, int out_channel, Dims dim_input, Dims dim_kernel,
	Dims dim_pad, Dims dim_stride, cudaStream_t stream);

class my_convolution : public nvinfer1::IPlugin
{
public:
	my_convolution() {}

	my_convolution(float* input, size_t size,
		int batchSize,
		int out_channel,
		float* weight,
		float* bias,
		Dims dim_input,
		Dims dim_kernel,
		Dims dim_pad,
		Dims dim_stride)
	{
		N = batchSize;
		K = out_channel;

		C = dim_input.d[0];
		H = dim_input.d[1];
		W = dim_input.d[2];
		int kH = dim_kernel.d[0];
		int kW = dim_kernel.d[1];
		int pH = dim_pad.d[0];
		int pW = dim_pad.d[1];
		int sH = dim_stride.d[0];
		int sW = dim_stride.d[1];

		//calculate the dims of outputs
		P = ((H + 2 * pH - kH) / sH) + 1;
		Q = ((W + 2 * pW - kW) / sW) + 1;

		//allocate memory for data on CPU
		WEIGHT = (float*)malloc(K * C * kH * kW * sizeof(float));
		BIAS = (float*)malloc(K * sizeof(float));

		WEIGHT = weight;
		BIAS = bias;

		d_input = dim_input;
		d_kernel = dim_kernel;
		d_pad = dim_pad;
		d_stride = dim_stride;

		outputDims = { 3, {K, P, Q}, {} };

		std::cout << "N  is " << N << std::endl;
		std::cout << "C  is " << C << std::endl;
		std::cout << "H  is " << H << std::endl;
		std::cout << "W  is " << W << std::endl;

		std::cout << "K  is " << K << std::endl;
		std::cout << "P  is " << P << std::endl;
		std::cout << "Q  is " << Q << std::endl;

		std::cout << "kH  is " << kH << std::endl;
		std::cout << "kW  is " << kW << std::endl;

		std::cout << "pH  is " << pH << std::endl;
		std::cout << "pW  is " << pW << std::endl;

		std::cout << "sH  is " << sH << std::endl;
		std::cout << "sW  is " << sW << std::endl;

	}

	~my_convolution() {}

	int getNbOutputs() const override { return 1; } // Number of outputs from the layer

	Dims getOutputDimensions(int index, const Dims* inputs, int npInputDims) override { // Dims of outputs
		assert(index == 0);
		assert(npInputDims == 1);
		assert(inputs[index].nbDims == 3);
		return outputDims;
	}

	void configure(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
	{
		assert(nbInputs == 1);
		assert(inputs[0].nbDims == 3);
		assert(outputDims[0].nbDims == 3);
		assert(nbOutputs == 1);
	}

	int initialize() override { return 0; }

	void terminate() override {}

	size_t getWorkspaceSize(int maxBatchSize) const override { return 0; /*return maxBatchSize * sizeof(float);*/ }

	int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override // inputs and outputs on GPU
	{
		float* d_WEIGHT;
		cudaMalloc((void**)&d_WEIGHT, K * C * kH * kW * sizeof(float));
		cudaMemcpyAsync((void*)d_WEIGHT, (const void*)WEIGHT, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice, stream);
		//cudaMemcpy((void*)d_WEIGHT, (const void*)WEIGHT, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice);

		float* d_BIAS;
		cudaMalloc((void**)&d_BIAS, K * sizeof(float));
		cudaMemcpyAsync((void*)d_BIAS, (const void*)BIAS, K * sizeof(float), cudaMemcpyHostToDevice, stream);
		//cudaMemcpy(d_BIAS, (const void*)BIAS, sizeof(BIAS), cudaMemcpyHostToDevice);

		std::cout << "Device Data Ready" << std::endl;

		my_convolution_func((float*)outputs[0], (float*)inputs[0],
			(float*)d_WEIGHT, (float*)d_BIAS, N, K, d_input, d_kernel, d_pad, d_stride, stream);

		std::cout << "CUDA kernel completed" << std::endl;

		return 0;
	}

	size_t getSerializationSize() override { return 0; }

	void serialize(void* buffer) override { }

public:
	int N{ 1 };
	int K{ 5 };
	int C{ 1 };
	int H{ 28 };
	int W{ 28 };
	int kH{ 5 };
	int kW{ 5 };
	int pH{ 0 };
	int pW{ 0 };
	int sH{ 1 };
	int sW{ 1 };
	float* WEIGHT;
	float* BIAS;
	int P;
	int Q;
	Dims d_input;
	Dims d_kernel;
	Dims d_pad;
	Dims d_stride;
	Dims outputDims;

};

bool load_data(float* output, const char* name, int size)
{
	std::ifstream pfile(name, std::ios::in | std::ios::binary);
	if (pfile.bad()) {
		std::cout << "Cannot find the file!" << std::endl;
		return false;
	}

	pfile.read((char*)output, size * sizeof(float));
	if (pfile.bad()) {
		std::cout << "Read Error!" << std::endl;
		return false;
	}

	return true;
}

bool load_data_int(int* output, const char* name, int size)
{
	std::ifstream pfile(name, std::ios::in | std::ios::binary);
	if (pfile.bad()) {
		std::cout << "Cannot find the file!" << std::endl;
		return false;
	}

	pfile.read((char*)output, size * sizeof(int));
	if (pfile.bad()) {
		std::cout << "Read Error!" << std::endl;
		return false;
	}

	return true;
}

class MyInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
	const int m_Total;
	const int m_Batch;
	const int m_InputC;
	const int m_InputH;
	const int m_InputW;
	const int m_TotalSize;
	const int m_BatchSize;
	const std::string m_InputBlobName;
	const std::string m_CalibTableFilePath;
	int m_ImageIndex;
	bool m_ReadCache;
	void* m_DeviceInput{ nullptr };
	std::vector<char> m_CalibrationCache;
	float* m_Data{ nullptr };

	MyInt8Calibrator(const int& Total, const int& Batch, const int& InputC, const int& InputH, const int& InputW,
		const std::string& InputBlobName, const std::string& CalibTableFilePath,
		float* Data, bool ReadCache = true) :
		m_Total(Total),
		m_Batch(Batch),
		m_InputC(InputC),
		m_InputH(InputH),
		m_InputW(InputW),
		m_TotalSize(Total * InputC * InputH * InputW),
		m_BatchSize(Batch * InputC * InputH * InputW),
		m_InputBlobName(InputBlobName.c_str()),
		m_CalibTableFilePath(CalibTableFilePath.c_str()),
		m_ImageIndex{ 0 },
		m_Data(Data),
		m_ReadCache(ReadCache)
	{
		cudaMalloc((void**)&m_DeviceInput, m_BatchSize * sizeof(float));
	}

	virtual ~MyInt8Calibrator() { cudaFree(m_DeviceInput); }

	int getBatchSize() const override { return m_Batch; }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		std::cout << m_ImageIndex << std::endl;

		float* BatchData = (float*)malloc(m_BatchSize * sizeof(float));

		for (int i = 0; i < m_BatchSize; i++) {
			int index = m_BatchSize * m_ImageIndex + i;
			if (index >= m_TotalSize) { std::cout << "calibration finished" << std::endl; return false; }
			else {
				BatchData[i] = m_Data[index];
			}
		}

		m_ImageIndex += m_Batch;

		cudaMemcpy(m_DeviceInput, (const void*)BatchData, m_BatchSize * sizeof(float), cudaMemcpyHostToDevice);

		assert(!strcmp(names[0], m_InputBlobName.c_str()));
		bindings[0] = m_DeviceInput;

		std::free(BatchData);

		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		void* output{ nullptr };
		m_CalibrationCache.clear();
		assert(!m_CalibTableFilePath.empty());
		std::ifstream input(m_CalibTableFilePath, std::ios::binary | std::ios::in);
		input >> std::noskipws;
		if (m_ReadCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
				std::back_inserter(m_CalibrationCache));

		length = m_CalibrationCache.size();
		if (length)
		{
			std::cout << "Using cached calibration table to build the engine" << std::endl;
			output = &m_CalibrationCache[0];
		}

		else
		{
			std::cout << "New calibration table will be created to build the engine" << std::endl;
			output = nullptr;
		}

		return output;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		assert(!m_CalibTableFilePath.empty());
		std::ofstream output(m_CalibTableFilePath, std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
		std::cout << "Write New calibration table" << std::endl;
	}

};

int main()
{
	clock_t start = clock();

	int status{ 0 };

	const int Total = 10000;
	const int BatchSize = 10000;
	const int InputC = 1;
	const int InputH = 28;
	const int InputW = 28;
	const int OutputSize = 10;

	const char* InputName = "data";
	const char* OutputName = "prob";

	Logger gLogger{ Logger::Severity::kVERBOSE };

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
	if (!builder)
	{
		status = -1;
		return status;
	}

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	if (!config)
	{
		status = -1;
		return status;
	}

	//// Build engine
	builder->setMaxBatchSize(BatchSize);
	config->setMaxWorkspaceSize(1_GiB);
	config->setFlag(BuilderFlag::kDEBUG);	
	config->setAvgTimingIterations(1);
	config->setMinTimingIterations(1);
	//config->setFlag(BuilderFlag::kGPU_FALLBACK);
	//builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION));

	//// FP16
	//config->setFlag(BuilderFlag::kFP16);
	//config->setFlag(BuilderFlag::kSTRICT_TYPES);

	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	if (!network)
	{
		status = -1;
		return status;
	}

	float* conv1filter_data = new float[125];
	float* conv1bias_data = new float[5];
	float* ip1filter_data = new float[120 * 720];
	float* ip1bias_data = new float[120];
	float* ip2filter_data = new float[10 * 120];
	float* ip2bias_data = new float[10];

	if (!load_data(conv1filter_data, "weights/conv1filter_torch_float32.wts", 125)) { status = -1;  return status; }
	if (!load_data(conv1bias_data, "weights/conv1bias_torch_float32.wts", 5)) { status = -1;  return status; }
	if (!load_data(ip1filter_data, "weights/ip1filter_torch_float32.wts", 120 * 720)) { status = -1;  return status; }
	if (!load_data(ip1bias_data, "weights/ip1bias_torch_float32.wts", 120)) { status = -1;  return status; }
	if (!load_data(ip2filter_data, "weights/ip2filter_torch_float32.wts", 10 * 120)) { status = -1;  return status; }
	if (!load_data(ip2bias_data, "weights/ip2bias_torch_float32.wts", 10)) { status = -1;  return status; }

	//if (!load_data(conv1filter_data, "weights/conv1filter_tf_float32.wts", 125)) { status = -1;  return status; }
	//if (!load_data(conv1bias_data, "weights/conv1bias_tf_float32.wts", 5)) { status = -1;  return status; }
	//if (!load_data(ip1filter_data, "weights/ip1filter_tf_float32.wts", 120 * 720)) { status = -1;  return status; }
	//if (!load_data(ip1bias_data, "weights/ip1bias_tf_float32.wts", 120)) { status = -1;  return status; }
	//if (!load_data(ip2filter_data, "weights/ip2filter_tf_float32.wts", 10 * 120)) { status = -1;  return status; }
	//if (!load_data(ip2bias_data, "weights/ip2bias_tf_float32.wts", 10)) { status = -1;  return status; }

	Weights conv1filter{ DataType::kFLOAT,  (const void*)conv1filter_data, (int64_t)125 };
	Weights conv1bias{ DataType::kFLOAT,   (const void*)conv1bias_data , (int64_t)5 };
	Weights ip1filter{ DataType::kFLOAT,  (const void*)ip1filter_data, (int64_t)(120 * 720) };
	Weights ip1bias{ DataType::kFLOAT, (const void*)ip1bias_data, (int64_t)120 };
	Weights ip2filter{ DataType::kFLOAT, (const void*)ip2filter_data, (int64_t)(10 * 120) };
	Weights ip2bias{ DataType::kFLOAT, (const void*)ip2bias_data, (int64_t)10 };

	// Create input tensor of shape { 1, 1, 28, 28 }
	ITensor* data = network->addInput(
		InputName, DataType::kFLOAT, Dims3{ 1, InputH, InputW });
	assert(data);

	std::cout << data->getDimensions() << std::endl;
	std::cout << data->getName() << std::endl;
	std::cout << (int)(data->getType()) << std::endl;
	std::cout << data->getDynamicRange() << std::endl;
	std::cout << data->isNetworkInput() << std::endl;
	std::cout << data->isNetworkOutput() << std::endl;

	std::cout << "=====================================================" << std::endl;

	// Plugin convolution layer
	IPlugin* my_conv = new my_convolution((float*)data, 1 * 28 * 28 * sizeof(float),
		BatchSize, 5, (float*)conv1filter.values, (float*)conv1bias.values,
		Dims{ 3, {1, 28, 28} }, Dims{ 2, {5, 5} }, Dims{ 2, {0, 0} }, Dims{ 2, {1, 1} });

	IPluginLayer *conv_plugin = network->addPlugin(&data, 1, *my_conv);

	std::cout << conv_plugin->getName() << std::endl;
	std::cout << (int)(conv_plugin->getType()) << std::endl;

	std::cout << "IpluginLayer made." << std::endl;
	conv_plugin->getOutput(0)->setName("conv_plugin");

	//conv_plugin->getOutput(0)->setName(OutputName);
	//network->markOutput(*conv_plugin->getOutput(0));

	// Add convolution layer with 5 outputs and a 5x5 filter.
	//IConvolutionLayer* conv1 = network->addConvolutionNd(
	//	*data, 5, Dims{ 2, {5, 5}, {} }, conv1filter, conv1bias);
	//assert(conv1);
	//conv1->setPadding(DimsHW{ 0, 0 });
	//conv1->setStride(DimsHW{ 1, 1 });
	//conv1->getOutput(0)->setName("conv1");

	////conv1->getOutput(0)->setName(output);
	////network->markOutput(*conv1->getOutput(0));

	//////Add max pooling layer with stride of 2x2 and kernel size of 2x2.
	IPoolingLayer* pool1 = network->addPoolingNd(*conv_plugin->getOutput(0), PoolingType::kMAX, Dims{ 2, {2, 2}, {} });
	assert(pool1);
	pool1->setStride(DimsHW{ 2, 2 });
	pool1->getOutput(0)->setName("maxpool1");

	//pool1->getOutput(0)->setName(output);
	//network->markOutput(*pool1->getOutput(0));

	//// Add fully connected layer with 120 outputs.
	IFullyConnectedLayer* ip1
		= network->addFullyConnected(*pool1->getOutput(0), 120, ip1filter, ip1bias);
	assert(ip1);
	ip1->getOutput(0)->setName("dense1");

	//ip1->getOutput(0)->setName(output);
	//network->markOutput(*ip1->getOutput(0));

	//// Add activation layer using the ReLU algorithm.
	IActivationLayer* relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
	assert(relu1);
	relu1->getOutput(0)->setName("relu_dense1");

	// Add second fully connected layer with 10 outputs.
	IFullyConnectedLayer* ip2 = network->addFullyConnected(
		*relu1->getOutput(0), OutputSize, ip2filter, ip2bias);
	assert(ip2);
	ip2->getOutput(0)->setName("dense2");

	//ip2->getOutput(0)->setName(OutputName);
	//network->markOutput(*ip2->getOutput(0));

	////Add softmax layer to determine the probability.
	ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob);
	prob->getOutput(0)->setName(OutputName);
	network->markOutput(*prob->getOutput(0));

	////Loda the data
	float* h_data = (float*)malloc(Total * InputH * InputW * sizeof(float));

	if (!load_data(h_data, "data/mnist_test_images_float32.bin", Total * InputH * InputW))
	{
		status = -1; return status;
	}

	for (int i = 0; i < Total * InputH * InputW; i++)
	{
		h_data[i] /= 255.0f;
	}

	std::cout << "Test Data Ready" << std::endl;

	//////===================================================================================================
	//INT8
	//config->setFlag(BuilderFlag::kINT8);

	//const int calibration_number = 1000;

	//const std::string CalibrationFile = "CalibrationTableSample";

	//MyInt8Calibrator calibrator(Total, BatchSize, InputC, InputH, InputW, InputName, CalibrationFile, h_data);

	//config->setInt8Calibrator(&calibrator);

	//////===================================================================================================

	nvinfer1::ICudaEngine* mEngine = builder->buildEngineWithConfig(*network, *config);
	if (!mEngine)
	{
		status = -1;
		return status;
	}

	std::cout << "Engine Ready" << std::endl;

	////===================================================================================================
	IExecutionContext* context = mEngine->createExecutionContext();
	if (!context)
	{
		status = -1;
		return status;
	}

	std::cout << "Context Ready" << std::endl;

	int m_InputBindingIndex = mEngine->getBindingIndex(InputName);
	int m_OutputBindingIndex = mEngine->getBindingIndex(OutputName);

	////========================================================================================================================

	std::vector<void*> Buffers;

	Buffers.resize(mEngine->getNbBindings(), nullptr);

	cudaMalloc((void**)&Buffers[m_InputBindingIndex], Total * InputH * InputW * sizeof(float));

	cudaMalloc((void**)&Buffers.at(m_OutputBindingIndex), Total * OutputSize * sizeof(float));

	//cudaMalloc(&Buffers[m_OutputBindingIndex], BatchSize * 5 * 24 * 24 * sizeof(float));

	float* h_output = (float*)malloc(BatchSize * OutputSize * sizeof(float));

	//std::unique_ptr<float> h_output = (float*)malloc(BatchSize * OutputSize * sizeof(float));

	//float* h_output = (float*)malloc(BatchSize * 5 * 24 * 24 * sizeof(float));

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaMemcpyAsync(Buffers[m_InputBindingIndex], h_data,
		Total * InputH * InputW * sizeof(float),
		cudaMemcpyHostToDevice, stream);

	//cudaMemcpy(Buffers.at(m_InputBindingIndex), h_data,
	//	Total * InputH * InputW * sizeof(float),
	//	cudaMemcpyHostToDevice);

	std::cout << "HostToDevice" << std::endl;

	bool stat = context->enqueue(Total, Buffers.data(), stream, nullptr);
	if (!stat)
	{
		std::cout << "context ERROR!" << std::endl;
		status = -1;
		return status;
	}

	cudaMemcpyAsync(h_output, Buffers[m_OutputBindingIndex],
		BatchSize * OutputSize * sizeof(float),
		cudaMemcpyDeviceToHost, stream);

	//cudaMemcpy(h_output, Buffers.at(m_OutputBindingIndex),
	//	BatchSize * OutputSize * sizeof(float),
	//	cudaMemcpyDeviceToHost);

	std::cout << "DeviceToHost" << std::endl;

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	std::cout << "Stream Destroyed" << std::endl;

	//// layer confirm
	//original answer
	//float* origin = (float*)malloc(BatchSize * OutputSize * sizeof(float));
	//if (!load_data(origin, "value/result_torch_float32.bin", BatchSize * OutputSize)) { status = -1; return status; }

	//// compare
	//for (int i = 0; i < BatchSize * OutputSize; ++i) {
	//	printf("Index: %d, PyTorch: %f,  TensorRT: %f\n", i, origin[i], h_output[i]);
	//}

	int* label = (int*)malloc(Total * sizeof(int));

	if (!(load_data_int(label, "data/mnist_test_labels_int32.bin", Total))) { status = -1; return status; }

	int count = 0;
	for (int i = 0; i < Total; i++) {
		int answer = label[i];
		int MyAnswer;
		float max = -0.01f;
		for (int j = 0; j < OutputSize; j++)
		{
			int index = OutputSize * i + j;
			if (h_output[index] > max) { max = h_output[index]; MyAnswer = j; }
		}
		if (MyAnswer == answer) { count += 1;}
	}

	std::cout << "The number of correct is " << count << std::endl;
	std::cout << ((float)count / (float)(Total)) * 100.0f << "%" << std::endl;

	////========================================================================================================================

	std::cout << "Finished!\n" << std::endl;

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;

}
