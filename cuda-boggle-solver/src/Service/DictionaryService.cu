#include <sstream>
#include <fstream>
#include <unordered_set>
#include <cmath> // for log
#include "Service\DictionaryService.cuh"
#include "Tools\UnicodeTools.cuh"

DictionaryService::DictionaryService() {
	initDictionary("English", &dictionaries["English"], "./words.txt", 45);   //pneumonoultramicroscopicsilicovolcanoconiosis
	initDictionary("French", &dictionaries["French"], "./french.txt", 29);    //anticonstitutionnellement
	initDictionary("Italian", &dictionaries["Italian"], "./italian.txt", 26); //precipitevolissimevolmente
	initDictionary("Spanish", &dictionaries["Spanish"], "./spanish.txt", 22); //esternocleidomastoideo
    //initDictionary("Russian", &dictionaries["Russian"], "./russian.txt", 31); //превысокомногорассмотрительствующий
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

void DictionaryService::createDictionary(const std::string& dictionaryName, int maxWordLength) {
    dictionaries.emplace(dictionaryName, HostTrie());
    dictionaries[dictionaryName].maxWordSize = maxWordLength;
}

void DictionaryService::populateDictionary(const std::string& dictionaryName, const std::u32string& spaceSeparatedWords) {
    auto it = dictionaries.find(dictionaryName);
    if (it != dictionaries.end()) {
        std::u32string currentWord;
        for (char32_t ch : spaceSeparatedWords) {
            if (ch == U' ') {
                if (!currentWord.empty()) {
                    it->second.addWord(currentWord);
                    currentWord.clear();
                }
            }
            else {
                currentWord.push_back(ch);
            }
        }

        if (!currentWord.empty()) {
            it->second.addWord(currentWord);
        }
    }
    it->second.buildTrie(it->second.maxWordSize);
}

void DictionaryService::buildTrie(const std::string& dictionaryName) {
    auto it = dictionaries.find(dictionaryName);
    if (it != dictionaries.end()) {
        it->second.buildTrie(it->second.maxWordSize);
    }
}


void DictionaryService::deleteDictionary(const std::string& dictionaryName) {
    auto it = dictionaries.find(dictionaryName);
    if (it != dictionaries.end()) {
        it->second.destroyTrie();
    }
    dictionaries.erase(dictionaryName);
}

bool DictionaryService::isPunctuation(char32_t ch) {
    return ch == U'/' || ch == U'!' || ch == U'?' || ch == U'"' || ch == U'\'' ||
        ch == U'(' || ch == U')' || ch == U'[' || ch == U']' || ch == U'{' ||
        ch == U'}' || ch == U'%' || ch == U'&' || ch == U'#' || ch == U'*' ||
        ch == U'@' || ch == U'$' || ch == U'^' || ch == U'`' || ch == U'~' ||
        ch == U'\\' || ch == U'|' || ch == U';' || ch == U':' || ch == U',';
}

std::u32string DictionaryService::obtainWords(const std::u32string& text) {
    std::u32string processed;

    for (char32_t ch : text) {
        if (ch >= U'A' && ch <= U'Z') {
            // convert to lower case
            processed += ch + 32;
        }
        else if (!isPunctuation(ch)) {
            processed += ch;
        }
    }

    std::u32string result;
    std::u32string word;
    std::basic_stringstream<char32_t> stream(processed);

    while (stream >> word) {
        if (!result.empty()) result += U' ';
        result += word;
    }

    return result;
}

std::unordered_set<std::u32string> DictionaryService::collectWords(std::string dictionary) {
    std::unordered_set<std::u32string> words;
    if (dictionaries.find(dictionary) != dictionaries.end()) {
        dictionaries[dictionary].collectWords(words);
    }

    return words;
}

void displayCollectedWords(const std::unordered_set<std::u32string>& words) {
    for (const auto& word : words) {
        std::wcout << utf32ToWstring(word) << L" ";
    }
    std::wcout << std::endl;
}

double DictionaryService::similarityByCommonWords(const std::string& dictionary1, const std::string& dictionary2) {
    if (dictionaries.find(dictionary1) == dictionaries.end() || dictionaries.find(dictionary2) == dictionaries.end()) {
        std::cerr << "Dictionary not found!" << std::endl;
        return -1;
    }
    else {
        auto wordsDictionary1 = collectWords(dictionary1);
        auto wordsDictionary2 = collectWords(dictionary2);

        displayCollectedWords(wordsDictionary1);

        auto uniqueWordsDictionary = wordsDictionary2;

        size_t countCommonWord = 0;

        for (auto word : wordsDictionary1) {
            uniqueWordsDictionary.insert(word);
            if (wordsDictionary2.find(word) != wordsDictionary2.end()) {
                countCommonWord++;
            }
        }

        size_t countUniqueWord = uniqueWordsDictionary.size();

        double probability = log(countCommonWord + 1) / log(countUniqueWord + 1);
        return probability;
    }
}


double DictionaryService::similarityByCommonSymbols(const std::u32string& text1, const std::u32string& text2) {
    std::unordered_set<int> unicodeCharactersText1;
    std::unordered_set<int> unicodeCharactersText2;
    std::unordered_set<int> unicodeCharactersUniqueBothTexts;


    for (auto unicodeChar : text1) {
        unicodeCharactersText1.insert(unicodeChar);
        unicodeCharactersUniqueBothTexts.insert(unicodeChar);
    }

    for (auto unicodeChar : text2) {
        unicodeCharactersText2.insert(unicodeChar);
        unicodeCharactersUniqueBothTexts.insert(unicodeChar);
    }

    size_t countCommonChar = 0;

    for (auto unicodeChar : unicodeCharactersText1) {
        if (unicodeCharactersText2.find(unicodeChar) != unicodeCharactersText2.end()) {
            countCommonChar++;
        }
    }

    size_t countUniqueChar = unicodeCharactersUniqueBothTexts.size();

    double probability = log(countCommonChar + 1) / log(countUniqueChar + 1);
    return probability;
}
