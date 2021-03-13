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

/*
TensorRT C++ API documentation
https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html

TensorRT Best Practices Guide 
https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html

여기에 부족한 설명은 위 두 링크를 참고하면 된다. 본인도 위 두 곳에서 퍼온 것 뿐이다.

작성자: 백승환
*/

int main()
{
	clock_t start = clock();

	int status{ 0 };

	/* 필요한 값들을 미리 정해둔다.
	Total : 사용된 총 이미지수. mnist test image 수(10000장)
	BatchSize : inference 단계니까 BatchSize=1 대입.
	InputC, InputH, InputW = (1, 28, 28)
	OuputSize = 10 (mnist class 개수)
	calibration_number: int8 calibrator에 사용할 이미지 수. 1000이라면 처음 1000장을 이용해 
	calibration을 하고 calibrationtable 파일을 생성한다. 이후 이 파일을 이용해 8bit inference가
	진행된다. 이 코드에서는 나머지 9000장에 대해서 inference가 진행되고 이 9000장에 대한 accuracy가
	출력된다. calibration_number을 N으로 변경하면 (10000-N)장에 대한 accuray가 나온다. 
	*/
	int Total = 10000;
	int BatchSize = 1;
	const int InputC = 1;
	const int InputH = 28;
	const int InputW = 28;
	const int OutputSize = 10;

	int calibration_number = 1000;

	/*
	TensorRT에서는 각 layer마다 name을 줄 수 있다. 특히 input layer name은 input data를 넣는 곳,
	output layer name은 output data를 얻을 수 있는 곳을 특정시킬 수 있기에 중요하다.
	*/
	const std::string InputName = "data";
	const std::string OutputName = "prob";

	/*
	load_data_vector<data_type> : 받고 싶은 data type을 지정하면 binary파일로부터 지정한 type으로 data를 읽어들여
									std::vector 형태로 받는다.		
	*/
	auto h_data = load_data_vector<float>("C:\\Users\\muger\\Desktop\\data\\mnist_test_images_float32.bin");

	std::cout << "Test Data Ready" << std::endl;

	//======================================================================================================
	/*
	TensorRT에서 engine을 생성하기 위해서는 Logger 객체를 생성해야한다.
	engine: Neural Network를 최적화된 graph로 만든 실행 파일로 생각하자.
	kVERBOSE : 이 option을 사용하면 TensorRT 최적화 과정이 자세히 출력된다. 한 번 읽어보면 도움된다.
	*/
	Logger gLogger{ Logger::Severity::kVERBOSE };

	/*
	IBuilder 객체를 생성한다.
	*/
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
	if (!builder)
	{
		status = -1;
		return status;
	}

	
	/*
	Config 객체를 생성한다. (IBuilder로부터)
	*/
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	if (!config)
	{
		status = -1;
		return status;
	}

	/*
	IBuilder 와 Config를 setting한다.
	*/
	builder->setMaxBatchSize(BatchSize); // MaxBatchSize를 set한다.
	config->setMaxWorkspaceSize(1_GiB);  // MaxWorkspaceSize를 set한다.
	config->setFlag(BuilderFlag::kDEBUG); // DEBUG 옵션을 활성화한다.
	config->setAvgTimingIterations(1); // Set the number of averaging iterations used when timing layers.
	config->setMinTimingIterations(1); // setMinTimingIterations
	config->setFlag(BuilderFlag::kSTRICT_TYPES); // layer의 activation(==data), weight type을 고정시킨다.
	config->setFlag(BuilderFlag::kGPU_FALLBACK); // Enable layers marked to execute on GPU if layer cannot execute on DLA.

	//// FP16
	//config->setFlag(BuilderFlag::kFP16); // 16bit 사용시

	////INT8
	config->setFlag(BuilderFlag::kINT8); // 8bit 사용시

	/*
	vector slicing 하는 법.
	h_data.begin() == 0
	h_data.begin() + (calibration_number * InputH * InputW) == calibration_number * InputH * InputW
	h_data[0:calibration_number * InputH * InputW] 이런 느낌이라 생각하면 된다.
	*/
	auto calib_data = std::vector<float>(h_data.begin(),
		h_data.begin() + (calibration_number * InputH * InputW)); 

	const std::string CalibrationFile = "CalibrationTableSample";

	//======================================================================================================
	//MyInt8Calibrator* calibrator = new MyInt8Calibrator(calibration_number, 1,
	//	InputC, InputH, InputW, InputName, CalibrationFile, (float*)&calib_data[0]);

	////std::unique_ptr<MyInt8Calibrator> Copy_calibrator(calibrator); // (*)
	//config->setInt8Calibrator(calibrator);
	//======================================================================================================

	/*
	Int8 calibrator 객체를 생성한다.
	unique_ptr를 사용한 방식.
	밑의 방식대신 위의 단락을 이용하고 밑에서 delete[] calibrator; 을 사용해도 된다.
	또한 위의 단락을 사용하면서 (*)을 같이 사용하면 delete을 사용하지 않아도 된다.
	자세한 사용방법은 unique_ptr을 설명해야 하므로 따로 검색.
	*/
	std::unique_ptr<MyInt8Calibrator> calibrator = std::make_unique<MyInt8Calibrator>(calibration_number, BatchSize,
		InputC, InputH, InputW, InputName, CalibrationFile, calib_data);

	config->setInt8Calibrator(calibrator.get());
	//======================================================================================================
	/*
	IBuilder로부터 network 객체를 생성한다.
	network는 Neural Network를 graph를 그리는 객체이므로 network구조와 관련되어 있다.
	*/
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	if (!network)
	{
		status = -1;
		return status;
	}

	/*
	network에 사용된 weights들을 std::vector 형태로 load한다.
	*/
	std::vector<float> conv1filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\conv1filter_torch_float32.wts");
	std::vector<float> conv1biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\conv1bias_torch_float32.wts");
	std::vector<float> ip1filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip1filter_torch_float32.wts");
	std::vector<float> ip1biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip1bias_torch_float32.wts");
	std::vector<float> ip2filterData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip2filter_torch_float32.wts");
	std::vector<float> ip2biasData = load_data_vector<float>("C:\\Users\\muger\\Desktop\\weights\\ip2bias_torch_float32.wts");

	/*
	TensoRT에서 사용하는 Weights라는 class 형식에 맞춰서 객체를 생성해준다.
	위 과정과 아래 과정을 함께 하는 function을 만들 수 있으나, 이해를 위하여 두 과정으로 나누었다.
	알아두면 좋은 것이 vector의 경우, float* float_array = &conv1filterData[0];
	&conv1filterData[0] 자체가 float* 형태로 사용될 수 있다. 
	vector에서 array를 만들어야 하는 경우가 아주 많이 있으므로 꼭 기억하자.
	*/
	Weights conv1filter{ DataType::kFLOAT,  (const void*)&conv1filterData[0], (int64_t)125 };
	Weights conv1bias{ DataType::kFLOAT,  (const void*)&conv1biasData[0], (int64_t)5 };
	Weights ip1filter{ DataType::kFLOAT,  (const void*)&ip1filterData[0], (int64_t)(120 * 720) };
	Weights ip1bias{ DataType::kFLOAT,  (const void*)&ip1biasData[0], (int64_t)120 };
	Weights ip2filter{ DataType::kFLOAT,  (const void*)&ip2filterData[0], (int64_t)(10 * 120) };
	Weights ip2bias{ DataType::kFLOAT,  (const void*)&ip2biasData[0], (int64_t)10 };

	/*
	Create input tensor. shape: { 1, 1, 28, 28 }
	InputName.c_str() == (const char*)"data"
	*/
	ITensor* data = network->addInput(
		InputName.c_str(), DataType::kFLOAT, Dims3{ 1, InputH, InputW });
	assert(data);

	/*
	input data의 각종 정보를 확인하는 방법.
	*/
	std::cout << data->getDimensions() << std::endl;
	std::cout << data->getName() << std::endl;
	std::cout << (int)(data->getType()) << std::endl;
	std::cout << data->getDynamicRange() << std::endl;
	std::cout << data->isNetworkInput() << std::endl;
	std::cout << data->isNetworkOutput() << std::endl;

	std::cout << "=====================================================" << std::endl;
	//////===================================================================================================
	/*
	Create scale layer with default power/shift and specified scale parameter.
	*/
	const float scaleParam = 1.0f / 255.0f;
	const Weights power{ DataType::kFLOAT, nullptr, 0 };
	const Weights shift{ DataType::kFLOAT, nullptr, 0 };
	const Weights scale{ DataType::kFLOAT, &scaleParam, 1 };
	IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
	assert(scale_1);
	scale_1->getOutput(0)->setName("scale1");

	//////===================================================================================================
	/*
	Convolution layer
	*/
	IConvolutionLayer* conv1 = network->addConvolutionNd(
		*scale_1->getOutput(0), 5, Dims{ 2, {5, 5}, {} }, conv1filter, conv1bias);
	assert(conv1);
	conv1->setPadding(DimsHW{ 0, 0 });
	conv1->setStride(DimsHW{ 1, 1 });
	conv1->getOutput(0)->setName("conv1");

	//////===================================================================================================
	//IPluginV2IOExt* PoolPlugin = new PoolingPluginV2IO(*conv1->getOutput(0), BatchSize, 5, 24, 24,
	//	2, 2, 0, 0, 2, 2);
	/*
	평범한 new 방식을 사용할려면 윗 줄을 사용하고 밑에서 delete[] PoolPlugin; 을 하면 된다.
	addPluginV2 함수의 첫 번째 argument type이 ITensor* const * 으로 되어있다.
	한 번에 이를 전해주는 방식을 찾지 못 했다. 
	혹시 방법을 찾은 분은 알려주길 바란다.
	*/
	std::unique_ptr<PoolingPluginV2IO> PoolPlugin = std::make_unique<PoolingPluginV2IO>(conv1->getOutput(0), BatchSize, 5, 24, 24,
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
	/*
	TensorFormat이 DataFormat을 의미한다.
	아직 이 부분까지는 손을 대지 못 했는데, 이 부분까지 설정을 하면 Plugin Layer를 더 최적화시킬 수 있는 걸로 보인다.
	시간이 나면 해보겠다. 이런 것이 있다는 걸 기억해두자.
	*/
	//network->getInput(0)->setAllowedFormats(TensorFormat::kLINEAR | TensorFormat::kCHW32);
	//network->getOutput(0)->setAllowedFormats(TensorFormat::kLINEAR | TensorFormat::kCHW32);

	//////===================================================================================================
	/*
	CudaEngine 객체를 IBuilder와 config, network로부터 생성한다.
	** 만약 그 전에 생성된 calibrationtable이 없다면 여기서 calibration이 실행되고 calibrationtable이 생성된다.
	그 calibration이 반영된 CudaEngine이 생성된다.
	*/
	nvinfer1::ICudaEngine* mEngine = builder->buildEngineWithConfig(*network, *config);
	if (!mEngine)
	{
		status = -1;
		return status;
	}

	std::cout << "Engine Ready" << std::endl;
	////===================================================================================================
	/*
	Contexte 객체를 생성한다. 이런 순서를 지켜야 된다는 것만 기억하자.
	*/
	IExecutionContext* context = mEngine->createExecutionContext();
	if (!context)
	{
		status = -1;
		return status;
	}

	std::cout << "Context Ready" << std::endl;

	/*
	이후에 input, output data를 위해 index를 CudaEngine에 기록해둔다.
	*/
	int m_InputBindingIndex = mEngine->getBindingIndex(InputName.c_str());
	int m_OutputBindingIndex = mEngine->getBindingIndex(OutputName.c_str());

	////========================================================================================================================
	/*
	이 과정은 CudaEngine을 이용해 testdata를 batch=1인 경우에 한 장씩 data를 보내 inference를
	하는 과정을 풀어서 쓴 것이다.
	중요 과정에 코멘트를 한다.
	*/
	// 전체 epoch를 구한다. (calibration에 사용한 image 제외)
	int epochs = ((Total - calibration_number) + BatchSize - 1) / BatchSize;

	// OutSize == 10
	int OutSize = OutputSize;

	const int BatchResultSize = BatchSize * OutSize;

	// 이해하기 쉽게 vector를 담는 vector를 생성한다.
	std::vector<std::vector<float>> h_total_output;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		// batchdata만 담는 vector 생성.
		std::vector<float> h_batchdata;

		// 각 batchdata의 시작 index
		auto start_index = h_data.begin() + ((epoch + calibration_number) * BatchSize * InputH * InputW);

		// 각 batchdata의 종료 index
		auto end_index = h_data.begin() + ((epoch + calibration_number) * BatchSize * InputH * InputW + BatchSize * InputH * InputW);

		// batchdata[시작 index: 종료 index]
		h_batchdata = std::vector<float>(start_index, end_index);

		// context->enqueue 함수 argument type이 void**이 필요해서 void* 형태의 vector를 생성한다.
		std::vector<void*> Buffers;
		
		/*
		Buffers.size = 2 -> 이경우는 input, output 1개씩인 단순한 경우
		좀 더 복잡한 network인 경우...
		*/
		Buffers.resize(mEngine->getNbBindings(), nullptr);

		/*
		input, output 자리에 gpu memory를 할당해놓는다.
		*/
		cudaMalloc((void**)&Buffers[m_InputBindingIndex], BatchSize * InputH * InputW * sizeof(float));

		cudaMalloc((void**)&Buffers[m_OutputBindingIndex], BatchResultSize * sizeof(float));

		/*
		output data를 받을 cpu memory를 할당해놓는다.
		*/
		std::vector<float> h_output(BatchResultSize);

		/*
		async를 사용하기 위해 cudastream 객체를 생성한다.
		*/
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		/*
		input data를 gpu에 async하게 memcpy한다.
		*/
		cudaMemcpyAsync(Buffers[m_InputBindingIndex], (const void*)&h_batchdata[0],
			BatchSize * InputH * InputW * sizeof(float),
			cudaMemcpyHostToDevice, stream);

		/*
		context를 enqueue한다.
		-> network의 연산을 한다.
		-> inference를 한다.
		*/
		bool stat = context->enqueue(BatchSize, Buffers.data(), stream, nullptr);
		if (!stat)
		{
			std::cout << "context ERROR!" << std::endl;
			status = -1;
			return status;
		}

		/*
		output data를 gpu에서 cpu로 async하게 memcpy한다.
		*/
		cudaMemcpyAsync((void*)&h_output[0], Buffers[m_OutputBindingIndex],
			BatchResultSize * sizeof(float),
			cudaMemcpyDeviceToHost, stream);

		/*
		그냥 순서를 외우자.
		*/
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);

		/*
		하나의 epoch이 끝났으니 나중에 전체 accuracy를 계산하기 위해 저장해놓는다.
		*/
		h_total_output.emplace_back(h_output);

	}

	////========================================================================================================================
	auto label = load_data_vector<int>("C:\\Users\\muger\\Desktop\\data\\mnist_test_labels_int32.bin");

	/*
	calibration data를 제외한 test data에 대해 inference를 진행한 accuracy를 확인.
	*/
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

	/*
	생성한 CudaEngine을 저장하는 방법.
	지금은 작은 모델이라 생성과 최적화 시간이 매우 적게 걸렸지만, 복잡한 모델의 경우 시간이 많이 걸리기 때문에
	시작할 때마다 모델 생성을 기다리는 건 시간 낭비다. 그렇기에 생성한 CudaEngine을 저장해놓고 실행파일 형태로
	사용하면 좋다.
	*/
	auto ModelEngine = mEngine->serialize();

	std::ofstream pout;
	pout.open("trtengine.engine", std::ios::binary | std::ios::out);

	if (pout.is_open())
	{
		pout.write((const char*)ModelEngine->data(), ModelEngine->size());
		pout.close();
	}

	ModelEngine->destroy();

	std::cout << "Finished!\n" << std::endl;

	//std::free(builder);

	//delete builder;

	//builder->destroy();

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;

}
