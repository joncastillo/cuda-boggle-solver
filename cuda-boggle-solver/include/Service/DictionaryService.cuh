#include <map>
#include <string>
#include <iostream>

#include "CudaTrie/trie_host.cuh"

class DictionaryService {
    std::map<std::string, HostTrie> dictionaries;
    DictionaryService();
    void initDictionary(const std::string& dictionaryName, HostTrie* dictionary, const std::string& dictionaryFile);


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
};
