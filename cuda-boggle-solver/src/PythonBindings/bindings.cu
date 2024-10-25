#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Tools/UnicodeTools.cuh"
#include "Service/DictionaryService.cuh"  // Include the header for DictionaryService
#include "Service/LogicOperationService.cuh"  // Include the header for LogicOperationService

namespace py = pybind11;

// PYBIND11_MODULE definition
PYBIND11_MODULE(dictionary_service, m) {
    m.doc() = "Dictionary and LogicOperation services for CUDA-based Boggle Solver";

    // Expose DictionaryService
    py::class_<DictionaryService>(m, "DictionaryService")
        .def_static("get_instance", &DictionaryService::get_instance, py::return_value_policy::reference)
        .def("check_words",
            [](DictionaryService& self, const std::string& language, const py::str& input) {
                std::string utf8input = input.cast<std::string>();
                std::u32string u32input = utf8ToUtf32(utf8input);
                return self.checkWords(language, u32input);
            },
            py::arg("language"),
            py::arg("input"),
            "Check words in the specified language's dictionary");

    // Expose LogicOperationService
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