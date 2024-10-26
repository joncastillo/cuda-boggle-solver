#include <iostream>
#include <map>
#include <string>
#include <unordered_set>

#include "CudaTrie/trie_host.cuh"

class DictionaryService {
    std::map<std::string, HostTrie> dictionaries;
    DictionaryService();
    void initDictionary(const std::string& dictionaryName, HostTrie* dictionary, const std::string& dictionaryFile, int maxWordLength);
    bool isPunctuation(char32_t ch);


public:
    // Delete copy constructor and assignment operator to prevent copying
    DictionaryService(const DictionaryService&) = delete;
    DictionaryService& operator=(const DictionaryService&) = delete;

    // Public method to access the single instance
    static DictionaryService& get_instance() {
        static DictionaryService instance; // Initialized on first use, destroyed on program exit
        return instance;
    }

    std::string checkWords(std::string language, std::u32string input);
    void createDictionary(const std::string& dictionaryName, int maxWordLength);
    void populateDictionary(const std::string& dictionaryName, const std::u32string& spaceSeparatedWords);
    void buildTrie(const std::string& dictionaryName);
    void deleteDictionary(const std::string& dictionaryName);
    std::u32string obtainWords(const std::u32string& text);
    std::unordered_set<std::u32string> DictionaryService::collectWords(std::string dictionary);
    double similarityByCommonWords(const std::string& dictionary1, const std::string& dictionary2);
    double similarityByCommonSymbols(const std::u32string& text1, const std::u32string& text2);
};
