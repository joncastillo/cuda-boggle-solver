#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Service\DictionaryService.cuh"
#include "Tools\UnicodeTools.cuh"

DictionaryService::DictionaryService() {
	initDictionary("English", &dictionaries["English"], "./words.txt");
	initDictionary("French", &dictionaries["French"], "./french.txt");
	initDictionary("Italian", &dictionaries["Italian"], "./italian.txt");
	initDictionary("Spanish", &dictionaries["Spanish"], "./spanish.txt");
}

void DictionaryService::initDictionary(const std::string& dictionaryName, HostTrie* dictionary, const std::string& dictionaryFile) {
    std::ifstream file(dictionaryFile);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << dictionaryFile << std::endl;
        return;
    }

    // Ensure file is read as UTF-8
    file.imbue(std::locale("en_US.UTF-8"));  // For Linux/macOS

    std::string line;  // Keep line as std::string because the file is in UTF-8
    std::wcout << L"Caching the " << std::wstring(dictionaryName.begin(), dictionaryName.end()) << L" dictionary..." << std::endl;

    while (getline(file, line)) {
        //printRawBytes(line);
        // Convert each line from UTF-8 to UTF-32
        std::u32string utf32Line = utf8ToUtf32(line);
        if (!utf32Line.empty()) {
            dictionary->addWord(utf32Line);  // Add the converted word to the trie
        }
    }
    file.close();
}

std::string DictionaryService::checkWords(std::string language, std::u32string input) {
	auto it = dictionaries.find(language);

	if (it != dictionaries.end()) {
		return it->second.searchFromHostParagraph(input);
	}
	else {
        return "";
	}
}

// pybind11 module definition
namespace py = pybind11;

PYBIND11_MODULE(dictionary_service, m) {
    py::class_<HostTrie>(m, "HostTrie");

    // Expose DictionaryService as a singleton with the get_instance method
    py::class_<DictionaryService>(m, "DictionaryService")
        .def_static("get_instance", &DictionaryService::get_instance, py::return_value_policy::reference)
        .def("check_words",
            [](DictionaryService& self, const std::string& language, const py::str& input) {
                std::u32string u32input = input.cast<std::u32string>();
                return self.checkWords(language, u32input);
            },
            py::arg("language"),
            py::arg("input"),
            "Check words in the specified language's dictionary");
}
