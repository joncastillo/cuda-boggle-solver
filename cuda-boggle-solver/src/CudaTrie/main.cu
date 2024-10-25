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

#include "Service\DictionaryService.cuh"
#include "Service\LogicOperationService.cuh"


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

    DictionaryService& dictionaryService = DictionaryService::get_instance();
    LogicOperationService& operationService = LogicOperationService::get_instance();

    std::cout << dictionaryService.checkWords("English", U"yellow stewardesses pneumonoultramicroscopicsilicovolcanoconiosis") << std::endl;
    std::cout << dictionaryService.checkWords("Italian", U"precipitevolissimevolmente epicità") << std::endl;
    std::cout << dictionaryService.checkWords("French", U"transsubstantieraient mélancoliquement") << std::endl;
    std::cout << dictionaryService.checkWords("Spanish", U"querido misericordiosamente") << std::endl;

#if 0
    HostTrie dictionaryArabic;
    initDictionary("Arabic", &dictionaryArabic, "./arabic2.txt");
    dictionaryArabic.buildTrie(10);
    std::u32string arabicHospital = { 0x0645, 0x0633, 0x062A, 0x0634, 0x0641, 0x0649 }; //مستشفى
    std::u32string arabicStewardess = { 0x0645, 0x0636, 0x064A, 0x0641, 0x0629 }; // مضيفة
    dictionaryArabic.searchFromHost(arabicHospital);
    dictionaryArabic.searchFromHost(arabicStewardess);
#endif 

    std::u32string input = U"Sangrá coverd her manoz, and when it dryed in the calde desert aire, Mapez reggrettava the wast of aqua. But that couldn't be helped—these men needed to die. They were Harkonens.        In the heet of the deep dessierto, a huge moisonneuse throbbed and thrummed as enourmous caterpiller traks crawled along the crest of a dune.Maccina chewd up the sand and digested the polvere through a compleks interplay of centrifuges and electromagnetic séperateurs.The moissoneuse vommited out a clowd of pousiere, arena, and debri that settled onto the disturbbed dunes behind the moving machin, while the bins filled up with the rare épicé mellange. The dronning operation sent pulsing vibrazionez beneth the desert, shure to call a ver de arãna... and verry soon.The noise also drownned out the sounds of Fremen vilencia inside the great machina.In the operations pont of the moving fabrika, another Harkonen worker tryed to flee, but a Fremen death - commando, a Fedaykin, ran after him.Disguized in a grimy shipsuit, the attacker had predetory and shure movimients, not at all like the morose sand crew the Harkonnens had hiered.";

    std::string outEnglish = dictionaryService.checkWords("English", input);
    std::string outItalian = dictionaryService.checkWords("Italian", input);
    std::string outFrench = dictionaryService.checkWords("French", input);
    std::string outSpanish = dictionaryService.checkWords("Spanish", input);


    // !(E|I|F|S)

    std::string wordsMask = operationService.logicalNotString(operationService.logicalOrStrings(operationService.logicalOrStrings(operationService.logicalOrStrings(outEnglish, outItalian), outFrench), outSpanish));

    auto words = operationService.filterWordsByBoolean(input, wordsMask);
    for (auto& word : words) {
        std::wcout << utf32ToWstring(word) << " ";
    }
    std::wcout << std::endl;

}
