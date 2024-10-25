#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Service/LogicOperationService.cuh"
#include "Tools/StringLogicOperations.cuh"

LogicOperationService& LogicOperationService::get_instance() {
    static LogicOperationService instance;
    return instance;
}

std::vector<int> LogicOperationService::splitStringToVector(const std::string& input) {
	return ::splitStringToVector(input);
}
std::string LogicOperationService::logicalOrStrings(const std::string& str1, const std::string& str2) {
	return ::logicalOrStrings(str1, str2);
}
std::string LogicOperationService::logicalAndStrings(const std::string& str1, const std::string& str2) {
	return ::logicalAndStrings(str1, str2);
}
std::string LogicOperationService::logicalXorStrings(const std::string& str1, const std::string& str2) {
	return ::logicalXorStrings(str1, str2);
}
std::string LogicOperationService::logicalNotString(const std::string& input) {
	return ::logicalNotString(input);
}
std::vector<std::u32string> LogicOperationService::splitU32StringToWords(const std::u32string& input) {
	return ::splitU32StringToWords(input);
}
std::vector<std::u32string> LogicOperationService::filterWordsByBoolean(const std::u32string& utf32Str, const std::string& boolCsv) {
	return ::filterWordsByBoolean(utf32Str, boolCsv);
}

// pybind11 module definition
namespace py = pybind11;

PYBIND11_MODULE(logic_operations_service, m) {
    py::class_<LogicOperationService>(m, "LogicOperationService")
        .def_static("get_instance", &LogicOperationService::get_instance, py::return_value_policy::reference)
        .def("splitStringToVector", &LogicOperationService::splitStringToVector)
        .def("logicalOrStrings", &LogicOperationService::logicalOrStrings)
        .def("logicalAndStrings", &LogicOperationService::logicalAndStrings)
        .def("logicalXorStrings", &LogicOperationService::logicalXorStrings)
        .def("logicalNotString", &LogicOperationService::logicalNotString)
        .def("splitU32StringToWords", &LogicOperationService::splitU32StringToWords)
        .def("filterWordsByBoolean", &LogicOperationService::filterWordsByBoolean);
}