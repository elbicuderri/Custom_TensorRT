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

/*
convolution을 cuda kernel로 작성하고 이를 tensorrt pluginV2Ext를 사용하여
custom layer를 implement 한 연습을 한 코드다.

놀라운 사실을 발견했는데.. 16bit dtype 돌아가는 kernel이 구현
안되어있는데 16비트로 engine이 생성되었다. engine 파일 저장은 안된다...
8bit 경우에는 convplugin도 table자체는 생성되지만 weight를 직접 넣는 방법을 구현
하지 못했다. calibrationtable을 어떻게 읽는지도 아직 잘 모르겠다..

작성자: 백승환

*/

/*
CUDA함수 error 체크 함수
*/
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

void my_convolution_func(float* output, float* input, float *weight, float *bias,
	int batch, int out_channel, Dims dim_input, Dims dim_kernel,
	Dims dim_pad, Dims dim_stride, cudaStream_t stream);

/*
calibrator writeCalibrationCache함수에서 쓰이는 함수.
*/
template<typename T> void write(char*& buffer, const T& val)
{
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}


class ConvPluginV2 : public nvinfer1::IPluginV2Ext
{
/*
class에서 쓰일 멤버 변수 정의.
*/
public:
	DataType iType;
	int N;
	int iC, iH, iW;
	int oC, oH, oW;
	int kH, kW;
	std::vector<float> h_weight;
	std::vector<float> h_bias;
	const char* mPluginNamespace;
	std::string mNamespace;

public:
	ConvPluginV2() = delete;

	ConvPluginV2(ITensor* data, int BatchSize, int oC, int iC, int iH, int iW, 
		int kH, int kW, int pH, int pW, int sH, int sW,
		 Weights Weight, Weights Bias)
		: iType(data->getType())
		, N(BatchSize)
		, iC(iC)
		, iH(iH)
		, iW(iW)
		, oC(oC)
		, oH(((iH - 2 * pH - kH) / sH) +1)
		, oW(((iW - 2 * pW - kW) / sW) + 1)
		, kH(kH)
		, kW(kW)
	{
		h_weight.insert(h_weight.end(), &((float*)Weight.values)[0], &((float*)Weight.values)[oC*iC*kH*kW]); // Weights 타입을 직접 이용한 방법.
		h_bias.insert(h_bias.end(), &((float*)Bias.values)[0], &((float*)Bias.values)[oC]);
	}

	~ConvPluginV2() override
	{
	}

	int getNbOutputs() const override { return 1; }

	int initialize() override { return 0; }

	void terminate() override {}

	Dims getOutputDimensions(int index,
		const Dims* inputs, int npInputDims) override
	{
		return Dims{ 3, {oC, oH, oW} };
	}

	size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

	/*
	cudastream을 이용한 cudamemcpyasync 사용가능 구현.
	*/
	int enqueue(int batchSize, const void* const* inputs,
		void** outputs, void* workspace, cudaStream_t stream) override
	{
		float* d_Weight{ nullptr };

		CHECK(cudaMalloc((void**)&d_Weight, oC * iC * kH * kW * sizeof(float)));

		cudaMemcpyAsync((void*)d_Weight, (const void*)&h_weight[0],
			oC * iC * kH * kW * sizeof(float), cudaMemcpyHostToDevice, stream);

		float* d_Bias{ nullptr };

		CHECK(cudaMalloc((void**)&d_Bias, oC * sizeof(float)));

		cudaMemcpyAsync((void*)d_Bias, (const void*)&h_bias[0], oC * sizeof(float), cudaMemcpyHostToDevice, stream);

		my_convolution_func((float*)outputs[0], (float*)inputs[0],
			(float*)d_Weight, (float*)d_Bias, N, oC,
			Dims{ 3, {iC, iH, iW} },
			Dims{ 2, {kH, kW} },
			Dims{ 2, {0,0} },
			Dims{ 2, {1,1} },
			stream);

		return 0;
	}

	size_t getSerializationSize() const override
	{
		// iC, iH, iW, oC, oH, oW
		return 6 * sizeof(int);
	}

	void serialize(void* buffer) const override
	{
		char *d = reinterpret_cast<char*>(buffer), *a = d;
		write(d, iC);
		write(d, iH);
		write(d, iW);
		write(d, oC);
		write(d, oH);
		write(d, oW);
		assert(d == a + getSerializationSize());
	}

	void configurePlugin(const Dims* inputDims, int nbInputs,
		const Dims* outputDims, int nbOutputs,
		const DataType* inputTypes, const DataType* outputTypes,
		const bool* inputIsBroadcast, const bool* outputIsBroadcast,
		PluginFormat floatFormat, int maxBatchSize) override
	{
		assert(nbInputs == 1);
		assert(nbOutputs == 1);

		iC = inputDims->d[0];
		iH = inputDims->d[1];
		iW = inputDims->d[2];

		oC = outputDims->d[0];
		oH = outputDims->d[1];
		oW = outputDims->d[2];

		iType = inputTypes[0];
	}

	bool supportsFormat(DataType type, PluginFormat format) const override
	{
		return ((type == DataType::kFLOAT || type == DataType::kHALF)
			&& format == PluginFormat::kNCHW);
	}

	const char* getPluginType() const override
	{
		return "ConvPluginV2";
	}

	const char* getPluginVersion() const override
	{
		return "2";
	}

	void destroy() override
	{
		delete this;
	}

	IPluginV2Ext* clone() const override
	{
		auto* plugin = new ConvPluginV2(*this);
		return plugin;
	}

	DataType getOutputDataType(int index, const DataType* inputTypes,
		int nbInputs) const override
	{
		return inputTypes[0];
	}

	void setPluginNamespace(const char* pluginNamespace) override
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mPluginNamespace;
	}

	bool isOutputBroadcastAcrossBatch(int outputIndex,
		const bool* inputIsBroadcasted, int nbInputs) const override
	{
		return false;
	}

	bool canBroadcastInputAcrossBatch(int inputIndex) const override
	{
		return false;
	}

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
	std::vector<float> m_data;
	std::vector<float> h_batchdata;

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
		CHECK(cudaMalloc((void**)&m_DeviceInput, m_BatchSize * sizeof(float)));
	}

	MyInt8Calibrator(const int& Total, const int& Batch, const int& InputC, const int& InputH, const int& InputW,
		const std::string& InputBlobName, const std::string& CalibTableFilePath,
		std::vector<float>& Data, bool ReadCache = true) :
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
		m_Data((float*)&Data[0]),
		m_ReadCache(ReadCache),
		m_data(Data)
	{
		CHECK(cudaMalloc((void**)&m_DeviceInput, m_BatchSize * sizeof(float)));
	}

	~MyInt8Calibrator() { cudaFree(m_DeviceInput); }

	int getBatchSize() const override { return m_Batch; }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{

		for (int i = 0; i < m_BatchSize; i++) {
			int index = m_BatchSize * m_ImageIndex + i;
			if (index >= m_TotalSize) { std::cout << "calibration finished" << std::endl; return false; }
			else {
				h_batchdata.emplace_back(m_Data[index]);
			}
		}

		m_ImageIndex += m_Batch;

		cudaMemcpy(m_DeviceInput, (const void*)&h_batchdata[0], m_BatchSize * sizeof(float), cudaMemcpyHostToDevice);

		assert(!strcmp(names[0], m_InputBlobName.c_str()));
		bindings[0] = m_DeviceInput;

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

template<typename T>
std::vector<T> load_data_vector(std::string name)
{
	std::ifstream input(name, std::ios::in | std::ios::binary);
	if (!(input.is_open()))
	{
		std::cout << "Cannot open the file!" << std::endl;
		exit(-1);
	}

	std::vector<T> data;
	input.seekg(0, std::ios::end);
	int size = input.tellg();
	input.seekg(0, std::ios::beg);

	for (int i = 0; i < size / sizeof(T); ++i) {
		T value;
		input.read((char*)&value, sizeof(T));
		data.push_back(value);
	}

	return data;
}

template<typename T>
void PrintVector(std::vector<T>& vector)
{
	for (auto &e : vector)
	{
		std::cout << e << std::endl;
	}
}

void print_DataType(DataType datatype)
{
	switch (datatype)
	{
	case DataType::kFLOAT:
		std::cout << "FLOAT" << std::endl;
		break;
	case DataType::kHALF:
		std::cout << "HALF" << std::endl;
		break;
	case DataType::kINT8:
		std::cout << "INT8" << std::endl;
		break;
	default:
		std::cout << "Unknown" << std::endl;
		break;
	}
}

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

	int calibration_number = 0;

	const char* InputName = "data";
	const char* OutputName = "prob";

	std::vector<float> h_data = load_data_vector<float>("C:\\Users\\muger\\Desktop\\data\\mnist_test_images_float32.bin");

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
	//config->setFlag(BuilderFlag::kSTRICT_TYPES);
	config->setFlag(BuilderFlag::kGPU_FALLBACK);

	//// FP16
	//config->setFlag(BuilderFlag::kFP16);

	////INT8
	//config->setFlag(BuilderFlag::kINT8);

	//auto calib_data = std::vector<float>(h_data.begin(),
	//	h_data.begin() + (calibration_number * InputH * InputW));

	//const std::string CalibrationFile = "CalibrationTableSample";

	//MyInt8Calibrator* calibrator = new MyInt8Calibrator(calibration_number, 1,
	//	InputC, InputH, InputW, InputName, CalibrationFile, (float*)&calib_data[0]);

	//config->setInt8Calibrator(calibrator);

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

	//scale_1->setPrecision(DataType::kINT8);
	//scale_1->setOutputType(0, DataType::kINT8);

	//////===================================================================================================
	//custom convolution layer
	//IPluginV2Ext* my_conv2 = new ConvPluginV2(DataType::kFLOAT, BatchSize,
	//	InputC, InputH, InputW, 5, 24, 24, 5, 5, conv1filter, conv1bias);

	//ConvPluginV2(ITensor* data, int BatchSize, int oC, int iC, int iH, int iW,
	//	int kH, int kW, int pH, int pW, int sH, int sW,
	//	Weights Weight, Weights Bias)

	//IPluginV2Ext* MyConvPlugin = new ConvPluginV2(scale_1->getOutput(0), BatchSize,
	//	5, InputC, InputH, InputW, 5, 5, 0, 0, 1, 1, conv1filter, conv1bias);

	std::unique_ptr<IPluginV2Ext> MyConvPlugin = std::make_unique<ConvPluginV2>(scale_1->getOutput(0), BatchSize,
		5, InputC, InputH, InputW, 5, 5, 0, 0, 1, 1, conv1filter, conv1bias);

	ITensor* const OutputOfScale = scale_1->getOutput(0);

	//IPluginV2Layer* conv1 = network->addPluginV2(&OutputOfScale, 1, *MyConvPlugin);
	IPluginV2Layer* conv1 = network->addPluginV2(&OutputOfScale, 1, *MyConvPlugin);

	std::cout << "IpluginLayer made." << std::endl;
	conv1->getOutput(0)->setName("Conv_PluginV2");

	//conv_plugin->getOutput(0)->setName(OutputName);
	//network->markOutput(*conv_plugin->getOutput(0));
	//////===================================================================================================

	//IConvolutionLayer* conv1 = network->addConvolutionNd(
	//	*scale_1->getOutput(0), 5, Dims{ 2, {5, 5}, {} }, conv1filter, conv1bias);
	//assert(conv1);
	//conv1->setPadding(DimsHW{ 0, 0 });
	//conv1->setStride(DimsHW{ 1, 1 });
	//conv1->getOutput(0)->setName("conv1");

	//////===================================================================================================

	//////Add max pooling layer with stride of 2x2 and kernel size of 2x2.
	IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0),
		PoolingType::kMAX, Dims{ 2, {2, 2}, {} });
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

	//print_DataType(ip2->getOutputType(0));

	////Add softmax layer to determine the probability.
	ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob);

	prob->getOutput(0)->setName(OutputName);
	network->markOutput(*prob->getOutput(0));

	//////===================================================================================================
	//set INT8 type
	//conv1->setOutputType(0, DataType::kFLOAT);
	//conv1->setPrecision(DataType::kFLOAT);

	//pool1->setPrecision(DataType::kINT8);
	//pool1->setOutputType(0, DataType::kINT8);
	//
	//ip1->setPrecision(DataType::kINT8);
	//ip1->setOutputType(0, DataType::kINT8);

	//relu1->setPrecision(DataType::kINT8);
	//relu1->setOutputType(0, DataType::kINT8);

	//ip2->setPrecision(DataType::kINT8);
	//ip2->setOutputType(0, DataType::kINT8);

	//prob->setPrecision(DataType::kINT8);
	//prob->setOutputType(0, DataType::kINT8);
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

	int epochs = ((Total - calibration_number) + BatchSize - 1) / BatchSize;

	const int ResultSize = Total * OutputSize;

	const int BatchResultSize = BatchSize * OutputSize;

	std::vector<std::vector<float>> h_total_output;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{

		//std::cout << epoch + 1 << "th image " << std::endl;

		std::vector<float> h_batchdata;

		auto start_index = h_data.begin() + ((epoch + calibration_number) * BatchSize * InputH * InputW);

		auto end_index = h_data.begin() + ((epoch + calibration_number) * BatchSize * InputH * InputW + BatchSize * InputH * InputW);

		h_batchdata = std::vector<float>(start_index, end_index);

		std::vector<void*> Buffers;

		Buffers.resize(mEngine->getNbBindings(), nullptr);

		CHECK(cudaMalloc((void**)&Buffers[m_InputBindingIndex], BatchSize * InputH * InputW * sizeof(float)));

		CHECK(cudaMalloc((void**)&Buffers[m_OutputBindingIndex], BatchResultSize * sizeof(float)));

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

		h_total_output.emplace_back(h_output);

	}

	////========================================================================================================================
	// layer confirm
	//auto origin = load_data_vector<float>("C:\\Users\\muger\\Desktop\\value\\dense2_torch_float32.bin");

	//for (int n = 0; n < Total - calibration_number; ++n) {
	//	std::cout << n + 1 << "th image: " << std::endl;
	//	for (int c = 0; c < 10; ++c) {
	//		printf("PyTorch: %f, TensorRT: %f\n", origin[(n + calibration_number) * 10 + c], h_total_output[n][c]);
	//	}
	//}

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
	/*
	이 코드는 현재 engine이 저장이 안 되고 있다. 
	*/

	auto ModelEngine = mEngine->serialize();

	std::ofstream pout;
	pout.open("trtengine.engine", std::ios::binary | std::ios::out);

	if (pout.is_open())
	{
		pout.write((const char*)ModelEngine->data(), ModelEngine->size());
		pout.close();
	}

	//ModelEngine->destroy();

	//delete[] calibrator;
	//delete[] MyConvPlugin;

	std::cout << "Finished!\n" << std::endl;

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;

}
