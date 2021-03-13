//#pragma once
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
Int8Calibrator를 TensorRT에서 쓰라는 방법대로 하려면 BatchStream이라는 class를 사용해야 한다.
좀 더 편리하게 사용하기 위해 MyInt8Calibrator class라는 새로 만들었다. 
작성자: 백승환
*/

class MyInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
/*
이 class 안에서 내가 사용하고 싶은 variable들을 정의한다.
숫자나 string 경우에는 const를 사용해도 좋으나 array나 vector에 const를 붙일 경우(멤버 변수에)
error가 자주 난다. (이유를 설명하기 좀 복잡하다...) 그러니까 숫자랑 string정도에만 const를 붙이자.(붙이고 싶으면)
*/
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

	MyInt8Calibrator() = delete; // 혹시나 변수 없는 객체가 생기면 지우겠다는 constructor다. 없어도 된다.

	/*
	Data가 float* 형태로 들어왔을 때의 constructor(생성자)
	*/
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

	/*
	Data가 std::vector<float> 형태로 들어왔을 때의 constructor(생성자)
	*/
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

	~MyInt8Calibrator() { cudaFree(m_DeviceInput); }

	/*
	calibration의 batch size를 확인하는 함수다. 대부분의 경우 1이라고 생각하면 될 거 같다.
	키워드에 짧게 설명해보자면,
	const: 이 멤버 함수 내에서 멤버 변수는 const하다. 어떠한 경우에도 변하지 않는다.
	override: 이 멤버 함수는 부모 클래스에서 상속받은 함수이고 그 함수를 override한다.
	*/
	int getBatchSize() const override { return m_Batch; }

	/*
	가장 중요한 함수이다. 
	batch data를 gpu에 넘겨주는 함수이다.
	bindings: 우리가 원하는 data(calibration 하고 싶은 batch data)를 넘겨주고 싶은 gpu pointer.
	names: network input layer의 name
	nbBindings: 현재는 쓰이지 않았다. 아마 input이 여러개인 경우 필요할 것으로 예측된다..
				bindings[nbBindings] = m_DeviceInput[nbBindings] // 이런 식으로 쓰이지 않을까?
	*/
	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		/*
		debugging을 위해 index를 출력한다.
		*/
		std::cout << m_ImageIndex << std::endl;

		for (int i = 0; i < m_BatchSize; i++) {
			int index = m_BatchSize * m_ImageIndex + i;
			/*
			calibration할 index가 넘어가면 getBatch함수를 종료한다.
			*/
			if (index >= m_TotalSize) { std::cout << "calibration finished" << std::endl; return false; }
			/*
			h_batchdata (CPU_memory)에 내가 보내고 싶은 data를 넘긴다.
			*/
			else {
				h_batchdata.emplace_back(m_Data[index]);
			}
		}

		/*
		index 올린다.
		*/
		m_ImageIndex += m_Batch;
		//++m_ImageIndex; //// m_Batch == 1인 경우

		/*
		gpu memory에 data를 복사한다. async하게 복사하는 방법은 stream을 어디서 create, destroy해야 되는 지 헷갈려서 아직 못 했다.
		*/
		cudaMemcpy(m_DeviceInput, (const void*)&h_batchdata[0], m_BatchSize * sizeof(float), cudaMemcpyHostToDevice);

		/*
		input layer name 확인.
		*/
		assert(!strcmp(names[0], m_InputBlobName.c_str()));

		/*
		원하는 위치의 memory에 data를 보낸다. 데이터 복사가 일어나서 비효율적인 것 같다. C++ 공부를 더 해야된다...
		*/
		bindings[0] = m_DeviceInput;

		return true;
	}


	/*
	밑의 두 함수는 어딘가에서 그대로 복사해서 돌리니 잘 돌아가고 있으니, 일단은 건드리지 말자.
	calibrationtable 파일이 있으면 있어서 8bit inference를 하고 없으면 table파일을 작성해주는
	함수다.
	*/
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
