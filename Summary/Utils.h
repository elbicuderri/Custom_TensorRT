//#pragma once

template<typename T> void write(char*& buffer, const T& val)
{
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

template<typename T>
std::vector<T> load_data_vector(const std::string& name)
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
		data.emplace_back(value);
	}

	return data;
}

template<typename T>
void PrintVector(const std::vector<T>& vector)
{
	for (auto &e : vector)
	{
		std::cout << e << std::endl;
	}
}

//void print_DataType(DataType datatype)
//{
//	switch (datatype)
//	{
//	case DataType::kFLOAT:
//		std::cout << "FLOAT" << std::endl;
//		break;
//	case DataType::kHALF:
//		std::cout << "HALF" << std::endl;
//		break;
//	case DataType::kINT8:
//		std::cout << "INT8" << std::endl;
//		break;
//	default:
//		std::cout << "Unknown" << std::endl;
//		break;
//	}
//}
