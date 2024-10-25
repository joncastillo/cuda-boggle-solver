#include <fstream>

#include "Service\DictionaryService.cuh"
#include "Tools\UnicodeTools.cuh"

DictionaryService::DictionaryService() {
	initDictionary("English", &dictionaries["English"], "./words.txt", 45);
	initDictionary("French", &dictionaries["French"], "./french.txt", 29);
	initDictionary("Italian", &dictionaries["Italian"], "./italian.txt", 26);
	initDictionary("Spanish", &dictionaries["Spanish"], "./spanish.txt", 29);
}

void DictionaryService::initDictionary(const std::string& dictionaryName, HostTrie* dictionary, const std::string& dictionaryFile, int maxWordLength) {
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

    dictionary->buildTrie(maxWordLength);
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
