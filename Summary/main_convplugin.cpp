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
#include <cstring>
#include <time.h>
#include <string>
#include <algorithm>
#include <vector>
#include <limits.h>
#include <float.h>

#include "utils.h"
#include "ConvPlugin.h"

int main()
{
	clock_t start = clock();

	int status{ 0 };

	int Total = 10000;
	int BatchSize = 1;
	int InputC = 1;
	int InputH = 28;
	int InputW = 28;
	int OutputSize = 10;

	const char* InputName = "data";
	const char* OutputName = "prob";

	std::vector<float> h_data = load_data_vector<float>("C:\\Users\\muger\\Desktop\\data/mnist_test_images_float32.bin");

	std::cout << "Test Data Ready" << std::endl;

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
	config->setFlag(BuilderFlag::kGPU_FALLBACK);

	//======================================================================================================

	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	if (!network)
	{
		status = -1;
		return status;
	}

	std::vector<float> conv1filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights/conv1filter_torch_float32.wts");
	std::vector<float> conv1biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights/conv1bias_torch_float32.wts");
	std::vector<float> ip1filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights/ip1filter_torch_float32.wts");
	std::vector<float> ip1biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights/ip1bias_torch_float32.wts");
	std::vector<float> ip2filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights/ip2filter_torch_float32.wts");
	std::vector<float> ip2biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights/ip2bias_torch_float32.wts");

	Weights conv1filter{ DataType::kFLOAT,  (const void*)&conv1filterData[0], (int64_t)125 };
	Weights conv1bias{ DataType::kFLOAT,  (const void*)&conv1biasData[0], (int64_t)5 };
	Weights ip1filter{ DataType::kFLOAT,  (const void*)&ip1filterData[0], (int64_t)(120 * 720) };
	Weights ip1bias{ DataType::kFLOAT,  (const void*)&ip1biasData[0], (int64_t)120 };
	Weights ip2filter{ DataType::kFLOAT,  (const void*)&ip2filterData[0], (int64_t)(10 * 120) };
	Weights ip2bias{ DataType::kFLOAT,  (const void*)&ip2biasData[0], (int64_t)10 };

	//// Create input tensor of shape { 1, 1, 28, 28 }
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

	//// Create scale layer with default power/shift and specified scale parameter.
	const float scaleParam = 1.0f / 255.0f;
	const Weights power{ DataType::kFLOAT, nullptr, 0 };
	const Weights shift{ DataType::kFLOAT, nullptr, 0 };
	const Weights scale{ DataType::kFLOAT, &scaleParam, 1 };
	IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
	assert(scale_1);
	scale_1->getOutput(0)->setName("scale1");

	//////===================================================================================================
	//////===================================================================================================
	std::unique_ptr<IPluginV2Ext> MyConvPlugin = std::make_unique<ConvPluginV2>(
		scale_1->getOutput(0),
		BatchSize,
		5,
		InputC,
		InputH,
		InputW, 5, 5, 0, 0, 1, 1, conv1filter, conv1bias);

	ITensor* const OutputOfScale = scale_1->getOutput(0);
	IPluginV2Layer* conv1 = network->addPluginV2(&OutputOfScale, 1, *MyConvPlugin);
	assert(conv1);
	std::cout << "Conv_Plugin_Layer made." << std::endl;
	conv1->getOutput(0)->setName("conv1");

	//////===================================================================================================
	//////===================================================================================================
	//IConvolutionLayer* conv1 = network->addConvolutionNd(
	//	*scale_1->getOutput(0), 5, Dims{ 2, {5, 5}, {} }, conv1filter, conv1bias);
	//assert(conv1);
	//conv1->setPadding(DimsHW{ 0, 0 });
	//conv1->setStride(DimsHW{ 1, 1 });
	//conv1->getOutput(0)->setName("conv1");

	//conv1->getOutput(0)->setName(OutputName); //// if you want to see result of each layer, you have to set output to this layer.
	//network->markOutput(*conv1->getOutput(0));
	//////===================================================================================================
	//////===================================================================================================
	//////Add max pooling layer with stride of 2x2 and kernel size of 2x2.
	IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0),
		PoolingType::kMAX, Dims{ 2, {2, 2}, {} });
	assert(pool1);
	pool1->setStride(DimsHW{ 2, 2 });
	pool1->getOutput(0)->setName("maxpool1");

	//pool1->getOutput(0)->setName(output);
	//network->markOutput(*pool1->getOutput(0));
	//////===================================================================================================
	//////===================================================================================================
	//// Add fully connected layer with 120 outputs.
	IFullyConnectedLayer* ip1
		= network->addFullyConnected(*pool1->getOutput(0), 120, ip1filter, ip1bias);
	assert(ip1);
	ip1->getOutput(0)->setName("dense1");

	//ip1->getOutput(0)->setName(output);
	//network->markOutput(*ip1->getOutput(0));
	//////===================================================================================================

	//// Add activation layer using the ReLU algorithm.
	IActivationLayer* relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
	assert(relu1);
	relu1->getOutput(0)->setName("relu_dense1");

	// Add second fully connected layer with 10 outputs.
	IFullyConnectedLayer* ip2 = network->addFullyConnected(*relu1->getOutput(0), OutputSize, ip2filter, ip2bias);
	assert(ip2);
	ip2->getOutput(0)->setName("dense2");

	//ip2->getOutput(0)->setName(OutputName);
	//network->markOutput(*ip2->getOutput(0));

	////Add softmax layer to determine the probability.
	ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob);

	prob->getOutput(0)->setName(OutputName);
	network->markOutput(*prob->getOutput(0));

	//// NETWORK created ////

	//////===================================================================================================
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
	//// inference loop
	int epochs = (Total + BatchSize - 1) / BatchSize;

	int OutSize = OutputSize;

	const int ResultSize = Total * OutSize;

	const int BatchResultSize = BatchSize * OutSize;

	std::vector<std::vector<float>> h_total_output;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{

		std::vector<float> h_batchdata;

		auto start_index = h_data.begin() + (epoch * BatchSize * InputH * InputW);

		auto end_index = h_data.begin() + (epoch * BatchSize * InputH * InputW + BatchSize * InputH * InputW);

		h_batchdata = std::vector<float>(start_index, end_index);

		std::vector<void*> Buffers;

		Buffers.resize(mEngine->getNbBindings(), nullptr);

		cudaMalloc((void**)&Buffers[m_InputBindingIndex], BatchSize * InputH * InputW * sizeof(float));

		cudaMalloc((void**)&Buffers[m_OutputBindingIndex], BatchResultSize * sizeof(float));

		std::vector<float> h_output(BatchResultSize);

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cudaMemcpyAsync(
			Buffers[m_InputBindingIndex],
			(const void*)&h_batchdata[0],
			BatchSize * InputH * InputW * sizeof(float),
			cudaMemcpyHostToDevice,
			stream);

		bool stat = context->enqueue(BatchSize, Buffers.data(), stream, nullptr);
		if (!stat)
		{
			std::cout << "context ERROR!" << std::endl;
			status = -1;
			return status;
		}

		cudaMemcpyAsync(
			(void*)&h_output[0],
			Buffers[m_OutputBindingIndex],
			BatchResultSize * sizeof(float),
			cudaMemcpyDeviceToHost,
			stream);

		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);

		h_total_output.emplace_back(h_output);
	}

	////========================================================================================================================

	auto label = load_data_vector<int>("C:\\Users\\muger\\Desktop\\data/mnist_test_labels_int32.bin");

	int count = 0;
	for (int i = 0; i < Total; i++) {
		int answer = label[i];
		int MyAnswer;
		float max = -10.0f;
		for (int j = 0; j < OutputSize; j++)
		{
			if (h_total_output[i][j] > max) { max = h_total_output[i][j]; MyAnswer = j; }
		}
		if (MyAnswer == answer) { count += 1; }
	}

	std::cout << "The number of correct is " << count << std::endl;
	std::cout << ((float)count / (float)(Total)) * 100.0f << "%" << std::endl;

	//////========================================================================================================================
	//////========================================================================================================================

	/*
	생성한 CudaEngine을 저장하는 방법.
	*/
	auto ModelEngine = mEngine->serialize();

	std::ofstream pout;
	pout.open("trtengine.engine", std::ios::binary | std::ios::out);

	if (pout.is_open())
	{
		pout.write((const char*)ModelEngine->data(), ModelEngine->size());
		pout.close();
		std::cout << "trtengine made.\n";
	}

	ModelEngine->destroy();
//////========================================================================================================================
//////========================================================================================================================
	std::cout << "Finished!\n" << std::endl;

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return status; //// if not error, return 0;

}