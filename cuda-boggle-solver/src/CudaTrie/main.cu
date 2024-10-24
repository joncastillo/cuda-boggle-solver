#include <iostream>
#include <fstream>
#include <locale>
#include <codecvt>

#include "CudaTrie\trie_host.h"

static void initDictionary(std::string dictionaryName, HostTrie* dictionary, std::string dictionaryFile) {
    std::wifstream file(dictionaryFile);
    file.imbue(std::locale(file.getloc(), new std::codecvt_utf8_utf16<wchar_t>));

    std::wstring line;

    if (file.is_open()) {
        std::wcout << L"caching the " << std::wstring(dictionaryName.begin(), dictionaryName.end()) << L" dictionary..." << std::endl;

        while (getline(file, line)) {
            dictionary->addWord(line);
        }
        file.close();
    }
    else {
        std::wcerr << L"Unable to open file" << std::endl;
    }
}

int main() {
#if 0
    HostTrie dictionaryEnglish;
    initDictionary("English", & dictionaryEnglish, "./words.txt");
    dictionaryEnglish.buildTrie(45);
    dictionaryEnglish.searchFromHost(L"yellow");
    dictionaryEnglish.searchFromHost(L"stewardesses");
    dictionaryEnglish.searchFromHost(L"pneumonoultramicroscopicsilicovolcanoconiosis");
#endif
    HostTrie dictionaryItalian;
    initDictionary("Italian", &dictionaryItalian, "./italian.txt");
    dictionaryItalian.buildTrie(26);
    dictionaryItalian.searchFromHost(L"precipitevolissimevolmente");
    dictionaryItalian.searchFromHost(L"epicità");
    
    HostTrie dictionaryArabic;
    initDictionary("Arabic", &dictionaryArabic, "./arabic.txt");
    dictionaryArabic.buildTrie(10);
    dictionaryArabic.searchFromHost(L"مستشفى");

}


#if 0
// download todo:
https://github.com/kkrypt0nn/wordlists/blob/main/wordlists/languages/
https://github.com/AustinZuniga/Filipino-wordlist/blob/master/Filipino-wordlist.txt
https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt


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