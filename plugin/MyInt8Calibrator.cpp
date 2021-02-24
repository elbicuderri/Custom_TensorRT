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

#include "MyInt8Calibrator.h"

MyInt8Calibrator::MyInt8Calibrator(int Total, int Batch, int InputC, int InputH, int InputW,
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

MyInt8Calibrator::MyInt8Calibrator(int Total, int Batch, int InputC, int InputH, int InputW,
    const std::string& InputBlobName, const std::string& CalibTableFilePath,
    const std::vector<float>& Data, bool ReadCache = true) :
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

MyInt8Calibrator::~MyInt8Calibrator() { cudaFree(m_DeviceInput); }

int MyInt8Calibrator::getBatchSize() const override { return m_Batch; }

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

const void* MyInt8Calibrator::readCalibrationCache(size_t& length) override
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

void MyInt8Calibrator::writeCalibrationCache(const void* cache, size_t length) override
{
    assert(!m_CalibTableFilePath.empty());
    std::ofstream output(m_CalibTableFilePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    std::cout << "Write New calibration table" << std::endl;
}