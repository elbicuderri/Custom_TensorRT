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

void my_convolution_func(float* output, float* input, float *weight, float *bias,
	int batch, int out_channel, Dims dim_input, Dims dim_kernel,
	Dims dim_pad, Dims dim_stride, cudaStream_t stream);

template<typename T> void write(char*& buffer, const T& val)
{
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

class ConvPluginV2 : public nvinfer1::IPluginV2Ext
{
public:
	DataType iType;
	int N;
	int iC, iH, iW;
	int oC, oH, oW;
	int kH, kW;
	float* h_Weight;
	float* h_Bias;
	std::vector<float> h_weight;
	std::vector<float> h_bias;
	const char* mPluginNamespace;
	std::string mNamespace;
	//const int NUM_COORDCONV_CHANNELS = 2;

public:
	ConvPluginV2() {}

	ConvPluginV2(DataType iType, int BatchSize, int iC, int iH, int iW,
		int oC, int oH, int oW, int kH, int kW, Weights Weight, Weights Bias)
		: iType(iType)
		, N(BatchSize)
		, iC(iC)
		, iH(iH)
		, iW(iW)
		, oC(oC)
		, oH(oH)
		, oW(oW)
		, kH(kH)
		, kW(kW)
		, h_Weight((float*)Weight.values)
		, h_Bias((float*)Bias.values)
	{
	}

	ConvPluginV2(ITensor* data, int BatchSize, int iC, int iH, int iW,
		int oC, int oH, int oW, int kH, int kW, Weights Weight, Weights Bias)
		: iType(data->getType())
		, N(BatchSize)
		, iC(iC)
		, iH(iH)
		, iW(iW)
		, oC(oC)
		, oH(oH)
		, oW(oW)
		, kH(kH)
		, kW(kW)
		, h_Weight((float*)Weight.values)
		, h_Bias((float*)Bias.values)
	{
	}

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
		, h_Weight((float*)Weight.values)
		, h_Bias((float*)Bias.values)
	{
	}

	//ConvPluginV2(DataType iType, int BatchSize, int iC, int iH, int iW,
	//	int oC, int oH, int oW, int kH, int kW, float* Weight, float* Bias)
	//	: iType(iType)
	//	, N(BatchSize)
	//	, iC(iC)
	//	, iH(iH)
	//	, iW(iW)
	//	, oC(oC)
	//	, oH(oH)
	//	, oW(oW)
	//	, kH(kH)
	//	, kW(kW)
	//{
	//	h_Weight = (float*)malloc(oC * iC * kH * kW * sizeof(float));
	//	h_Bias = (float*)malloc(oC * sizeof(float));
	//	h_Weight = Weight;
	//	h_Bias = Bias;
	//}

	//ConvPluginV2(DataType iType, int BatchSize, int iC, int iH, int iW,
	//	int oC, int oH, int oW, int kH, int kW, std::vector<float>& Weight, std::vector<float>& Bias)
	//	: iType(iType)
	//	, N(BatchSize)
	//	, iC(iC)
	//	, iH(iH)
	//	, iW(iW)
	//	, oC(oC)
	//	, oH(oH)
	//	, oW(oW)
	//	, kH(kH)
	//	, kW(kW)
	//	, h_Weight((float*)&Weight[0])
	//	, h_Bias((float*)&Bias[0])
	//{
	//}

	~ConvPluginV2() override
	{
	}

	int getNbOutputs() const override { return 1; }

	int initialize() override { return 0; }

	void terminate() override {}

	Dims getOutputDimensions(int index,
		const Dims* inputs, int npInputDims) override
	{
		//Dims dimsOutput;
		//dimsOutput.nbDims = inputs->nbDims;
		//dimsOutput.d[0] = inputs->d[0]
		//dimsOutput.d[1] = inputs->d[1];
		//dimsOutput.d[2] = inputs->d[2];
		//dimsOutput.d[3] = inputs->d[3];
		//return dimsOutput;

		return Dims{ 3, {oC, oH, oW} };
	}

	size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

	int enqueue(int batchSize, const void* const* inputs,
		void** outputs, void* workspace, cudaStream_t stream) override
	{
		float* d_Weight;
		cudaMalloc((void**)&d_Weight, oC * iC * kH * kW * sizeof(float));
		cudaMemcpyAsync((void*)d_Weight, (const void*)h_Weight,
			oC * iC * kH * kW * sizeof(float), cudaMemcpyHostToDevice, stream);
		//cudaMemcpyAsync((void*)d_Weight, (const void*)h_Weight,
		//	oC * iC * kH * kW * sizeof(float), cudaMemcpyHostToDevice, stream);
		//cudaMemcpy((void*)d_Weight, (const void*)WEIGHT, K * C * kH * kW * sizeof(float), cudaMemcpyHostToDevice);

		float* d_Bias;
		cudaMalloc((void**)&d_Bias, oC * sizeof(float));
		cudaMemcpyAsync((void*)d_Bias, (const void*)h_Bias, oC * sizeof(float), cudaMemcpyHostToDevice, stream);
		//cudaMemcpy(d_Bias, (const void*)BIAS, sizeof(BIAS), cudaMemcpyHostToDevice);

		//std::cout << "Device Data Ready" << std::endl;

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
		return "1";
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
		m_Data = Data;
		cudaMalloc((void**)&m_DeviceInput, m_BatchSize * sizeof(float));
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
		cudaMalloc((void**)&m_DeviceInput, m_BatchSize * sizeof(float));
	}

	virtual ~MyInt8Calibrator() { cudaFree(m_DeviceInput); }

	int getBatchSize() const override { return m_Batch; }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		std::cout << m_ImageIndex << std::endl;

		//float* BatchData = (float*)malloc(m_BatchSize * sizeof(float));

		//for (int i = 0; i < m_BatchSize; i++) {
		//	int index = m_BatchSize * m_ImageIndex + i;
		//	if (index >= m_TotalSize) { std::cout << "calibration finished" << std::endl; return false; }
		//	else {
		//		BatchData[i] = m_data[index];
		//	}
		//}

		for (int i = 0; i < m_BatchSize; i++) {
			int index = m_BatchSize * m_ImageIndex + i;
			if (index >= m_TotalSize) { std::cout << "calibration finished" << std::endl; return false; }
			else {
				h_batchdata.push_back(m_Data[index]);
			}
		}

		m_ImageIndex += m_Batch;

		//cudaMemcpy(m_DeviceInput, (const void*)BatchData, m_BatchSize * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(m_DeviceInput, (const void*)&h_batchdata[0], m_BatchSize * sizeof(float), cudaMemcpyHostToDevice);

		assert(!strcmp(names[0], m_InputBlobName.c_str()));
		bindings[0] = m_DeviceInput;

		h_batchdata.clear();

		//std::free(BatchData);

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

//template < typename T>
//std::vector<T> load_data_vector(std::string name)
//{
//	
//}

//template<typename T>
//std::vector<T> load_data_vector(std::string name)
//{
//	std::ifstream input(name, std::ios::in | std::ios::binary);
//	if (!(input.is_open())) 
//	{
//		std::cout << "Cannot open the file!" << std::endl;
//		exit(-1);
//	}
//	std::vector<T> data;
//	input.seekg(0, std::ios::end);
//	int size = input.tellg();
//	input.seekg(0, std::ios::beg);
//
//	for (int i = 0; i < size / sizeof(T); ++i) {
//		T value;
//		input.read((char*)&value, sizeof(T));
//		data.push_back(value);
//	}
//
//	return data;
//}

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

//template<typename T>
//void load_data_vector(std::vector<T>& data, std::string name)
//{
//	std::ifstream input(name, std::ios::in | std::ios::binary);
//	if (!(input.is_open()))
//	{
//		std::cout << "Cannot open the file!" << std::endl;
//		exit(-1);
//	}
//
//	input.seekg(0, std::ios::end);
//	int size = input.tellg();
//	input.seekg(0, std::ios::beg);
//
//	for (int i = 0; i < size / sizeof(T); ++i) {
//		T value;
//		input.read((char*)&value, sizeof(T));
//		data.push_back(value);
//	}
//}

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

	int Total = 9000;
	int BatchSize = 1;
	int InputC = 1;
	int InputH = 28;
	int InputW = 28;
	int OutputSize = 10;

	int calibration_number = 1000;

	const char* InputName = "data";
	const char* OutputName = "prob";

	std::vector<float> h_data = load_data_vector<float>("data/mnist_test_images_float32.bin");

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
	config->setFlag(BuilderFlag::kSTRICT_TYPES);
	//config->setFlag(BuilderFlag::kGPU_FALLBACK);

	//// FP16
	//config->setFlag(BuilderFlag::kFP16);

	////INT8
	config->setFlag(BuilderFlag::kINT8);

	auto calib_data = std::vector<float>(h_data.begin(),
		h_data.begin() + (calibration_number * InputH * InputW));

	const std::string CalibrationFile = "CalibrationTableSample";

	//MyInt8Calibrator calibrator(calibration_number, 1,
	//	InputC, InputH, InputW, InputName, CalibrationFile, (float*)&calib_data[0]);

	MyInt8Calibrator* calibrator2 = new MyInt8Calibrator(calibration_number, 1,
		InputC, InputH, InputW, InputName, CalibrationFile, (float*)&calib_data[0]);

	config->setInt8Calibrator(calibrator2);

	//======================================================================================================

	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	if (!network)
	{
		status = -1;
		return status;
	}

	std::vector<float> conv1filterData = load_data_vector<float>("weights/conv1filter_torch_float32.wts");
	std::vector<float> conv1biasData = load_data_vector<float>("weights/conv1bias_torch_float32.wts");
	std::vector<float> ip1filterData = load_data_vector<float>("weights/ip1filter_torch_float32.wts");
	std::vector<float> ip1biasData = load_data_vector<float>("weights/ip1bias_torch_float32.wts");
	std::vector<float> ip2filterData = load_data_vector<float>("weights/ip2filter_torch_float32.wts");
	std::vector<float> ip2biasData = load_data_vector<float>("weights/ip2bias_torch_float32.wts");

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

	IPluginV2Ext* my_conv2 = new ConvPluginV2(scale_1->getOutput(0), BatchSize,
		5, InputC, InputH, InputW, 5, 5, 0, 0, 1, 1, conv1filter, conv1bias);

	//IPluginV2Ext* my_conv2 = new ConvPluginV2(scale_1->getOutput(0), BatchSize,
	//	InputC, InputH, InputW, 5, 24, 24, 5, 5, conv1filter, conv1bias);

	ITensor* const OutputOfScale = scale_1->getOutput(0);

	IPluginV2Layer* conv_plugin = network->addPluginV2(&OutputOfScale, 1, *my_conv2);
	std::cout << "IpluginLayer made." << std::endl;
	conv_plugin->getOutput(0)->setName("Conv_PluginV2");

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
	IPoolingLayer* pool1 = network->addPoolingNd(*conv_plugin->getOutput(0),
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
	//conv_plugin->setOutputType(0, DataType::kINT8);
	//conv_plugin->setPrecision(DataType::kINT8);

	//conv1->setPrecision(DataType::kINT8);
	//conv1->setOutputType(0, DataType::kINT8);

	pool1->setPrecision(DataType::kINT8);
	pool1->setOutputType(0, DataType::kINT8);
	
	ip1->setPrecision(DataType::kINT8);
	ip1->setOutputType(0, DataType::kINT8);

	relu1->setPrecision(DataType::kINT8);
	relu1->setOutputType(0, DataType::kINT8);

	ip2->setPrecision(DataType::kINT8);
	ip2->setOutputType(0, DataType::kINT8);

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
	// layer confirm
	//auto origin = load_data_vector<float>("value/dense2_torch_float32.bin");

	//for (int n = 0; n < Total - calibration_number; ++n) {
	//	std::cout << n + 1 << "th image: " << std::endl;
	//	for (int c = 0; c < 10; ++c) {
	//		printf("PyTorch: %f, TensorRT: %f\n", origin[(n + calibration_number) * 10 + c], h_total_output[n][c]);
	//	}
	//}


	auto label = load_data_vector<int>("data/mnist_test_labels_int32.bin");

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

	std::cout << "The number of correct is " << count << std::endl;
	std::cout << ((float)count / (float)(Total - calibration_number)) * 100.0f << "%" << std::endl;

	//////========================================================================================================================

	std::cout << "Finished!\n" << std::endl;

	clock_t end = clock();

	printf("Time: %.6f\n", (float)(end - start) / CLOCKS_PER_SEC);

	return 0;

}
