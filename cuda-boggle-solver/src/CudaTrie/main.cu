#include <iostream>
#include <fstream>
#include <locale>
#include <codecvt>
#include <iomanip>
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#include "CudaTrie\trie_host.cuh"
#include "Tools\UnicodeTools.cuh"
#include "Tools\StringLogicOperations.cuh"

static void initDictionary(const std::string& dictionaryName, HostTrie* dictionary, const std::string& dictionaryFile) {
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

int main() {
    // Set console to use UTF-8 for proper Unicode output on Windows
    SetConsoleOutputCP(CP_UTF8);
    std::setlocale(LC_ALL, "en_US.UTF-8");

    HostTrie dictionaryEnglish;
    initDictionary("English", & dictionaryEnglish, "./words.txt");
    dictionaryEnglish.buildTrie(45);
    dictionaryEnglish.searchFromHost(U"yellow");
    dictionaryEnglish.searchFromHost(U"stewardesses");
    dictionaryEnglish.searchFromHost(U"pneumonoultramicroscopicsilicovolcanoconiosis");

    HostTrie dictionaryItalian;
    initDictionary("Italian", &dictionaryItalian, "./italian.txt");
    dictionaryItalian.buildTrie(26);
    dictionaryItalian.searchFromHost(U"precipitevolissimevolmente");
    dictionaryItalian.searchFromHost(U"epicità");

#if 0
    HostTrie dictionaryArabic;
    initDictionary("Arabic", &dictionaryArabic, "./arabic2.txt");
    dictionaryArabic.buildTrie(10);
    std::u32string arabicHospital = { 0x0645, 0x0633, 0x062A, 0x0634, 0x0641, 0x0649 }; //مستشفى
    std::u32string arabicStewardess = { 0x0645, 0x0636, 0x064A, 0x0641, 0x0629 }; // مضيفة
    dictionaryArabic.searchFromHost(arabicHospital);
    dictionaryArabic.searchFromHost(arabicStewardess);
#endif 

    HostTrie dictionaryFrench;
    initDictionary("French", &dictionaryFrench, "./french.txt");
    dictionaryFrench.buildTrie(29);
    dictionaryFrench.searchFromHost(U"transsubstantieraient");
    dictionaryFrench.searchFromHost(U"mélancoliquement");

    HostTrie dictionarySpanish;
    initDictionary("Spanish", &dictionarySpanish, "./spanish.txt");
    dictionarySpanish.buildTrie(29);
    dictionarySpanish.searchFromHost(U"querido");
    dictionarySpanish.searchFromHost(U"misericordiosamente");


    std::u32string input = U"Sangrá coverd her manoz, and when it dryed in the calde desert aire, Mapez reggrettava the wast of aqua. But that couldn't be helped—these men needed to die. They were Harkonens.        In the heet of the deep dessierto, a huge moisonneuse throbbed and thrummed as enourmous caterpiller traks crawled along the crest of a dune.Maccina chewd up the sand and digested the polvere through a compleks interplay of centrifuges and electromagnetic séperateurs.The moissoneuse vommited out a clowd of pousiere, arena, and debri that settled onto the disturbbed dunes behind the moving machin, while the bins filled up with the rare épicé mellange. The dronning operation sent pulsing vibrazionez beneth the desert, shure to call a ver de arãna... and verry soon.The noise also drownned out the sounds of Fremen vilencia inside the great machina.In the operations pont of the moving fabrika, another Harkonen worker tryed to flee, but a Fremen death - commando, a Fedaykin, ran after him.Disguized in a grimy shipsuit, the attacker had predetory and shure movimients, not at all like the morose sand crew the Harkonnens had hiered.";

    std::string outEnglish =  dictionaryEnglish.searchFromHostParagraph(input);
    std::string outItalian = dictionaryItalian.searchFromHostParagraph(input);
    std::string outFrench = dictionaryFrench.searchFromHostParagraph(input);
    std::string outSpanish = dictionarySpanish.searchFromHostParagraph(input);


    // !(E|I|F|S)

    std::string wordsMask = logicalNotString(logicalOrStrings(logicalOrStrings(logicalOrStrings(outEnglish, outItalian), outFrench), outSpanish));

    auto words = filterWordsByBoolean(input, wordsMask);
    for (auto& word : words) {
        std::wcout << utf32ToWstring(word) << " ";
    }
    std::wcout << std::endl;

}


#if 0
// download todo:
https://github.com/kkrypt0nn/wordlists/blob/main/wordlists/languages/
https://github.com/AustinZuniga/Filipino-wordlist/blob/master/Filipino-wordlist.txt
https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt
https://github.com/loayamin/arabic-words/blob/master/word-list.txt

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