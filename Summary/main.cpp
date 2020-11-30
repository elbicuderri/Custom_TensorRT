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
#include "PoolingPlugin.h"
#include "ReluPlugin.h"
#include "Int8Calibrator.h"
#include "Utils.h"

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

	int calibration_number = 1000;

	const std::string InputName = "data";
	const std::string OutputName = "prob";

	std::vector<float> h_data = load_data_vector<float>("C:\\Users\\muger\\Desktop\\data\\mnist_test_images_float32.bin");

	std::cout << "Test Data Ready" << std::endl;

	//======================================================================================================
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
	config->setFlag(BuilderFlag::kSTRICT_TYPES);
	config->setFlag(BuilderFlag::kGPU_FALLBACK);

	//// FP16
	//config->setFlag(BuilderFlag::kFP16);

	////INT8
	config->setFlag(BuilderFlag::kINT8);

	auto calib_data = std::vector<float>(h_data.begin(),
		h_data.begin() + (calibration_number * InputH * InputW));

	const std::string CalibrationFile = "CalibrationTableSample";

	MyInt8Calibrator* calibrator = new MyInt8Calibrator(calibration_number, 1,
		InputC, InputH, InputW, InputName, CalibrationFile, (float*)&calib_data[0]);

	config->setInt8Calibrator(calibrator);

	//======================================================================================================
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	if (!network)
	{
		status = -1;
		return status;
	}

	std::vector<float> conv1filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\conv1filter_torch_float32.wts");
	std::vector<float> conv1biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\conv1bias_torch_float32.wts");
	std::vector<float> ip1filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip1filter_torch_float32.wts");
	std::vector<float> ip1biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip1bias_torch_float32.wts");
	std::vector<float> ip2filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip2filter_torch_float32.wts");
	std::vector<float> ip2biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip2bias_torch_float32.wts");

	Weights conv1filter{ DataType::kFLOAT,  (const void*)&conv1filterData[0], (int64_t)125 };
	Weights conv1bias{ DataType::kFLOAT,  (const void*)&conv1biasData[0], (int64_t)5 };
	Weights ip1filter{ DataType::kFLOAT,  (const void*)&ip1filterData[0], (int64_t)(120 * 720) };
	Weights ip1bias{ DataType::kFLOAT,  (const void*)&ip1biasData[0], (int64_t)120 };
	Weights ip2filter{ DataType::kFLOAT,  (const void*)&ip2filterData[0], (int64_t)(10 * 120) };
	Weights ip2bias{ DataType::kFLOAT,  (const void*)&ip2biasData[0], (int64_t)10 };

	//// Create input tensor of shape { 1, 1, 28, 28 }
	ITensor* data = network->addInput(
		InputName.c_str(), DataType::kFLOAT, Dims3{ 1, InputH, InputW });
	assert(data);

	std::cout << data->getDimensions() << std::endl;
	std::cout << data->getName() << std::endl;
	std::cout << (int)(data->getType()) << std::endl;
	std::cout << data->getDynamicRange() << std::endl;
	std::cout << data->isNetworkInput() << std::endl;
	std::cout << data->isNetworkOutput() << std::endl;

	std::cout << "=====================================================" << std::endl;
	//////===================================================================================================
	//// Create scale layer with default power/shift and specified scale parameter.
	const float scaleParam = 1.0f / 255.0f;
	const Weights power{ DataType::kFLOAT, nullptr, 0 };
	const Weights shift{ DataType::kFLOAT, nullptr, 0 };
	const Weights scale{ DataType::kFLOAT, &scaleParam, 1 };
	IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
	assert(scale_1);
	scale_1->getOutput(0)->setName("scale1");

	//////===================================================================================================
	//Convolution layer
	IConvolutionLayer* conv1 = network->addConvolutionNd(
		*scale_1->getOutput(0), 5, Dims{ 2, {5, 5}, {} }, conv1filter, conv1bias);
	assert(conv1);
	conv1->setPadding(DimsHW{ 0, 0 });
	conv1->setStride(DimsHW{ 1, 1 });
	conv1->getOutput(0)->setName("conv1");

	//////===================================================================================================
	IPluginV2IOExt* PoolPlugin = new PoolingPluginV2IO(*conv1->getOutput(0), BatchSize, 5, 24, 24,
		2, 2, 0, 0, 2, 2);

	ITensor* const OutputOfConv = conv1->getOutput(0);

	IPluginV2Layer* pool_plugin = network->addPluginV2(&OutputOfConv, 1, *PoolPlugin);

	pool_plugin->getOutput(0)->setName("pool_pluginV2");
	//////===================================================================================================
	//////Add max pooling layer with stride of 2x2 and kernel size of 2x2.
	//IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0),
	//	PoolingType::kMAX, Dims{ 2, {2, 2}, {} });
	//assert(pool1);
	//pool1->setStride(DimsHW{ 2, 2 });
	//pool1->getOutput(0)->setName("maxpool1");

	//pool1->getOutput(0)->setName(output);
	//network->markOutput(*pool1->getOutput(0));

	//////===================================================================================================
	//// Add fully connected layer with 120 outputs.
	IFullyConnectedLayer* ip1
		= network->addFullyConnected(*pool_plugin->getOutput(0), 120, ip1filter, ip1bias);
	assert(ip1);
	ip1->getOutput(0)->setName("dense1");

	//////===================================================================================================
	IPluginV2IOExt* ReluPlugin = new ReluPluginV2IO(*ip1->getOutput(0), BatchSize,
		120, 1, 1);

	ITensor* const OutputOfIP1 = ip1->getOutput(0);

	IPluginV2Layer* relu1_plugin = network->addPluginV2(&OutputOfIP1, 1, *ReluPlugin);

	relu1_plugin->getOutput(0)->setName("relu1_pluginV2");
	//////===================================================================================================
	//// Add activation layer using the ReLU algorithm.
	//IActivationLayer* relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
	//assert(relu1);
	//relu1->getOutput(0)->setName("relu_dense1");

	//////===================================================================================================
	// Add second fully connected layer with 10 outputs.
	IFullyConnectedLayer* ip2 = network->addFullyConnected(
		*relu1_plugin->getOutput(0), OutputSize, ip2filter, ip2bias);
	assert(ip2);
	ip2->getOutput(0)->setName("dense2");

	//////===================================================================================================
	////Add softmax layer to determine the probability.
	ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob);

	prob->getOutput(0)->setName(OutputName.c_str());
	network->markOutput(*prob->getOutput(0));

	//////===================================================================================================
	//set INT8 type
	scale_1->setPrecision(DataType::kINT8);
	scale_1->setOutputType(0, DataType::kINT8);

	conv1->setPrecision(DataType::kINT8);
	conv1->setOutputType(0, DataType::kINT8);

	pool_plugin->setOutputType(0, DataType::kINT8);
	pool_plugin->setPrecision(DataType::kINT8);

	ip1->setPrecision(DataType::kINT8);
	ip1->setOutputType(0, DataType::kINT8);

	//relu1->setPrecision(DataType::kINT8);
	//relu1->setOutputType(0, DataType::kINT8);

	relu1_plugin->setPrecision(DataType::kINT8);
	relu1_plugin->setOutputType(0, DataType::kINT8);
	//////===================================================================================================

	//network->getInput(0)->setAllowedFormats(TensorFormat::kLINEAR | TensorFormat::kCHW32);
	//network->getOutput(0)->setAllowedFormats(TensorFormat::kLINEAR | TensorFormat::kCHW32);

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

	int m_InputBindingIndex = mEngine->getBindingIndex(InputName.c_str());
	int m_OutputBindingIndex = mEngine->getBindingIndex(OutputName.c_str());

	////========================================================================================================================

	int epochs = ((Total - calibration_number) + BatchSize - 1) / BatchSize;

	int OutSize = OutputSize;

	const int ResultSize = Total * OutSize;

	const int BatchResultSize = BatchSize * OutSize;

	std::vector<std::vector<float>> h_total_output;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{

		std::vector<float> h_batchdata;

		auto start_index = h_data.begin() + ((epoch + calibration_number) * BatchSize * InputH * InputW);

		auto end_index = h_data.begin() + ((epoch + calibration_number) * BatchSize * InputH * InputW + BatchSize * InputH * InputW);

		h_batchdata = std::vector<float>(start_index, end_index);

		std::vector<void*> Buffers;

		Buffers.resize(mEngine->getNbBindings(), nullptr);

		cudaMalloc((void**)&Buffers[m_InputBindingIndex], BatchSize * InputH * InputW * sizeof(float));

		cudaMalloc((void**)&Buffers[m_OutputBindingIndex], BatchResultSize * sizeof(float));

		std::vector<float> h_output(BatchResultSize);

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cudaMemcpyAsync(Buffers[m_InputBindingIndex], (const void*)&h_batchdata[0],
			BatchSize * InputH * InputW * sizeof(float),
			cudaMemcpyHostToDevice, stream);

		bool stat = context->enqueue(BatchSize, Buffers.data(), stream, nullptr);
		if (!stat)
		{
			std::cout << "context ERROR!" << std::endl;
			status = -1;
			return status;
		}

		cudaMemcpyAsync((void*)&h_output[0], Buffers[m_OutputBindingIndex],
			BatchResultSize * sizeof(float),
			cudaMemcpyDeviceToHost, stream);

		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);

		h_total_output.push_back(h_output);

		h_output.clear();

	}

	////========================================================================================================================
	auto label = load_data_vector<int>("C:\\Users\\muger\\Desktop\\data\\mnist_test_labels_int32.bin");

	int count = 0;
	for (int i = 0; i < Total - calibration_number; i++) {
		int answer = label[i + calibration_number];
		int MyAnswer;
		float max = -10.0f;
		for (int j = 0; j < OutputSize; j++)
		{
			if (h_total_output[i][j] > max) { max = h_total_output[i][j]; MyAnswer = j; }
		}
		if (MyAnswer == answer) { count += 1; }
	}

	std::cout << "The total number is " << Total - calibration_number << std::endl;
	std::cout << "The number of correct is " << count << std::endl;
	std::cout << ((float)count / (float)(Total - calibration_number)) * 100.0f << "%" << std::endl;

	//////========================================================================================================================
	//std::cout << (int)data->getAllowedFormats() << std::endl;
	//std::cout << (int)scale_1->getAllowedFormats() << std::endl;
	//std::cout << (int)conv1->getAllowedFormats() << std::endl;
	//std::cout << (int)data->getAllowedFormats() << std::endl;


	std::cout << "Finished!\n" << std::endl;

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;

}