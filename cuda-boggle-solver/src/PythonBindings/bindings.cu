#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Tools/UnicodeTools.cuh"
#include "Service/DictionaryService.cuh"
#include "Service/LogicOperationService.cuh"


namespace py = pybind11;

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
            "Check words in the specified language's dictionary")
        .def("create_custom_dictionary", &DictionaryService::createDictionary, py::arg("dictionaryName"), py::arg("maxWordLength"),
            "Create a new dictionary with the specified name and maximum word length")
        .def("destroy_custom_dictionary", &DictionaryService::deleteDictionary, py::arg("dictionaryName"),
            "Delete the dictionary with the specified name")
        .def("obtain_words",
            [](DictionaryService& self, const py::str& text) {
                std::string utf8text = text.cast<std::string>();
                std::u32string u32text = utf8ToUtf32(utf8text);
                return self.obtainWords(u32text);
            },
            py::arg("text"),
            "Obtain words from a given text")
        .def("populate_dictionary",
            [](DictionaryService& self, const std::string& dictionaryName, const py::str& spaceSeparatedWords) {
                std::string utf8Words = spaceSeparatedWords.cast<std::string>();
                std::u32string u32Words = utf8ToUtf32(utf8Words);
                self.populateDictionary(dictionaryName, u32Words);
            },
            py::arg("dictionaryName"),
            py::arg("spaceSeparatedWords"),
            "Populate a dictionary with space-separated words")
        .def("collect_words", &DictionaryService::collectWords, py::arg("dictionary"),
            "Collect words from a specified dictionary")
        .def("similarity_check_of_two_dictionaries", &DictionaryService::similarityByCommonWords, py::arg("dictionary1"), py::arg("dictionary2"),
            "Compute similarity based on common words between two dictionaries")
        .def("similarity_check_of_two_texts",
            [](DictionaryService& self, const py::str& text1, const py::str& text2) {
                std::u32string u32text1 = utf8ToUtf32(text1.cast<std::string>());
                std::u32string u32text2 = utf8ToUtf32(text2.cast<std::string>());
                return self.similarityByCommonSymbols(u32text1, u32text2);
            },
            py::arg("text1"), py::arg("text2"),
            "Compute similarity based on common symbols between two pieces of text");

    // Expose LogicOperationService
    py::class_<LogicOperationService>(m, "LogicOperationService")
        .def_static("get_instance", &LogicOperationService::get_instance, py::return_value_policy::reference)
        .def("splitStringToVector", &LogicOperationService::splitStringToVector)
        .def("logicalOrStrings", &LogicOperationService::logicalOrStrings)
        .def("logicalAndStrings", &LogicOperationService::logicalAndStrings)
        .def("logicalXorStrings", &LogicOperationService::logicalXorStrings)
        .def("logicalNotString", &LogicOperationService::logicalNotString)
        .def("splitU32StringToWords", &LogicOperationService::splitU32StringToWords)
        .def("filterWordsByBoolean", &LogicOperationService::filterWordsByBoolean)
        .def("calculate_bitmask_relevance", &LogicOperationService::getAccuracyFromBooleanMask);
}