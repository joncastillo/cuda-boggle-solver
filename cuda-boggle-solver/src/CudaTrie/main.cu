#include <fstream>

#include "CudaTrie\trie_host.h"

static void initDictionary(HostTrie* dictionary, std::string dictionaryFile) {
    std::cout << "caching the dictionary..." << std::endl;
    std::ifstream file(dictionaryFile);
    std::string line;

    if (file.is_open()) {
        while (getline(file, line)) {
            dictionary->addWord(line);
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file" << std::endl;
    }
}

int main() {

    HostTrie trie;
    initDictionary(&trie, "./words.txt");

    trie.buildTrie(45);
    trie.searchFromHost("yellow");
    trie.searchFromHost("stewardesses");
    trie.searchFromHost("pneumonoultramicroscopicsilicovolcanoconiosis");
}


#if 0
// download todo:
static void initDictionary(HostTrie* dictionary, const std::string& dictionaryFile) {
    std::cout << "caching the dictionary..." << std::endl;
    std::ifstream file(dictionaryFile);
    std::string line;

    if (file.is_open()) {
        while (getline(file, line)) {
            dictionary->addWord(line);
        }
        file.close();
    }
    else {
        std::cerr << "File not found. Downloading the dictionary..." << std::endl;
        const std::string url = "https://github.com/dwyl/english-words/raw/refs/heads/master/words.txt";
        std::string downloadCommand;

#ifdef _WIN32
        downloadCommand = "curl -o " + dictionaryFile + " " + url;
#else
        downloadCommand = "curl -o " + dictionaryFile + " " + url + " || wget -O " + dictionaryFile + " " + url;
#endif

        if (std::system(downloadCommand.c_str()) == 0) {
            std::cout << "Download complete. Loading the dictionary..." << std::endl;
            file.open(dictionaryFile);
            if (file.is_open()) {
                while (getline(file, line)) {
                    dictionary->addWord(line);
                }
                file.close();
            }
            else {
                std::cerr << "Unable to open downloaded file" << std::endl;
            }
        }
        else {
            std::cerr << "Download failed." << std::endl;
        }
    }
}
#endif