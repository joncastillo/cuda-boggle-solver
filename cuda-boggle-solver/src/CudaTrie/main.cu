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

    dictionaryService.createDictionary("My Custom 1", 20);
    dictionaryService.createDictionary("My Custom 2", 20);

    std::u32string inputText1 = U"łæßr æłøtñr. Thøŭ øŭ thæ Zñłøßœ ŭpæßžøıg. øı ßıŭwær tñ yñçr ßŭžøıg đñr þñræ łætßøðŭ ßåñçt thæ gññł tøþæŭ ø hßvæ hßł øı Vßððæjñ, ø ŭhßðð åæ væry hßppy tñ ŭçppðy ævæı þñræ þßtærøßð. åy thæ wßy, ßræ thæ pñðøœæ hßvæøıg ß gññł tøþæ wøth thæ œñłæ? øđ ıñt, tæðð thæþ tñ œhæær çp; whæı thæy łñ œrßœž øt thæy wøðð hßvæ þæ. ñı thæ 4th ñđ Jçðy: ø łøł ıñt ñpæı thæ œßr łññr, Thæ wøıłñw wßŭ rñððæł łñwı ßðð ræßły. Thæ åñy wßŭ ñrøgøñıßðy ŭøttøıg øı thæ đrñıt ŭæßt whæı ø åægßı đøræøıg. Whæı ø đøræł thæ đørŭt ŭhñt ßt høŭ hæßł, hæ ðæßpæł åßœžwßrłŭ ßt thæ ŭßþæ tøþæ thçŭ ŭpñøðøıg þy ßøþ. Hæ æıłæł çp ñı thæ åßœž ŭæßt thæı thæ đðññr øı åßœž thßŭhøıg ñçt væry vøñðæıtðy wøth høŭ ðægŭ; thßtŭ hñw ø ŭhñt høþ øı thæ žıææ. ø łøł ıñt ðæßvæ thæ œæıæ ñđ thæ žøððøıg wøth ŭqçæßððøıg tøræŭ & rßœæøıg æıgøıæ ßŭ łæŭœrøåæł øı thæ Vßððæjñ pßpær,. ø łrñvæ ßwßy qçøtæ ŭðñwðy ŭñ ßŭ ıñt tñ łrßw ßttæıtøñı tñ þy œßr. Thæy łøł ıñt ñpæıðy ŭtßtæ thøŭ, åçt øþpðøæł thøŭ åy ŭßyøıg øt wßŭ ß wæðð ðøt ıøght & ø œñçðł ŭææ thæ ŭøðñwætŭ ñı thæ hñrøzñı. åçðð ŭhøt thßt ßræß øŭ ŭrñçıłæł åy høgh høððŭ & trææŭ. Whßt ø łøł wßŭ tßpæ ß ŭþßðð pæıœæð đðßŭh ðøght tñ thæ åßrræð ñđ þy gçı. øđ yñç ıñtøœæ, øı thæ œæıtær ñđ thæ åæßþ ñđ ðøght øđ yñç ßøþ øt ßt ß wßðð ñr œæððøıg yñç wøðð ŭææ ß åðßœž ñr łßrœž ŭpñt øı thæ œæıtær ñđ thæ œørœðæ ñđ ðøght ßåñçt 3 tñ 6 øıœhæŭ ßœrñŭŭ. Whæı tßpæł tñ ß gçı åßrræð, thæ åçððæt wøðð ŭtrøžæ æxßœtðy øı thæ œæıtær ñđ thæ åðßœž łñt øı thæ ðøght. ßðð ø hßł tñ łñ wßŭ ŭprßy thæþ ßŭ øđ øt wßŭ ß wßtær hñŭæ; thæræ wßŭ ıñ ıææł tñ çŭæ thæ gçı ŭøghtŭ. ø wßŭ ıñt hßppy tñ ŭææ thßt ø łøł ıñt gæt đrñıt pßgæ œñværßgæ.";
    std::u32string inputText2 = U"Thøŭ øŭ thæ Zñłøßœ ŭpæßžøıg. ø ßþ thæ þçrłærær ñđ thæ tßxø łrøvær ñvær åy Wßŭhøıgtñı ŭt & þßpðæ ŭt ðßŭt ıøght, tñ prñvæ thøŭ hæræ øŭ ß åðññł ŭtßøıæł pøæœæ ñđ høŭ ŭhørt. ø ßþ thæ ŭßþæ þßı whñ łøł øı thæ pæñpðæ øı thæ ıñrth åßy ßræß. Thæ ŭ.đ. Pñðøœæ œñçðł hßvæ œßçght þæ ðßŭt ıøght øđ thæy hßł ŭæßrœhæł thæ pßrž prñpærðy øıŭtæßł ñđ hñðłøıg rñßł rßœæŭ wøth thæør þñtñrœøœðæŭ ŭææøıg whñ œñçðł þßžæ thæ þñŭt ıñøŭæ. Thæ œßr łrøværŭ ŭhñçðł hßvæ jçŭt pßržæł thæør œßrŭ ßıł ŭßt thæræ qçøætðy wßøtøıg đñr þæ tñ œñþæ ñçt ñđ œñvær. ŭœhññð œhøðłræı þßžæ ıøœæ tßrgætŭ, ø thøıž ø ŭhßðð wøpæ ñçt ß ŭœhññð åçŭ ŭñþæ þñrıøıg. Thæ þßı whñ tñðł thæ pñðøœæ thßt þy œßr wßŭ årñwı wßŭ ß ıægrñ ßåñçt 40–45 rßthær ŭhßååðy łræŭŭæł. ø wßŭ ßt thøŭ phñıæ åññth hßvæøıg ŭñþæ đçı wøth thæ Vßððæjñ œñpŭ whæı hæ wßŭ wßðžøıg åy. Whæı ø hçıg thæ phñıæ çp thæ łßþ X@ thøıg åægßı tñ røıg & thßt łræw høŭ ßttæıtøñı tñ þæ & þy œßr. ðßŭt œhrøŭtþßŭŭ øı thßt æpßŭñłæ thæ pñðøœæ wæræ wñıłærøıg ßŭ tñ hñw ø œñçðł ŭhññt & høt þy vøœtñþŭ øı thæ łßrž. Jçŭt ŭhññt ñçt thæ đrñıt tøræ & thæı pøœž ñđđ thæ žøłłøæŭ ßŭ thæy œñþæ åñçıœøıg ñçt. Thøŭ øŭ thæ Zñłøßœ ŭpæßžøıg ø thñçgh yñç wñçðł ıææł ß gññł ðßçgh åæđñræ yñç hæßr thæ åßł ıæwŭ. Yñç wñı't gæt thæ ıæwŭ đñr ß whøðæ yæt.ßıł ø œßı't łñ ß thøıg wøth øt! Pŭ œñçðł yñç prøıt thøŭ ıæw œøphær ñı yñçr đrçıt pßgæ? ø gæt ßwđçððy ðñıæðy whæı ø ßþ øgıñræł, ŭñ ðñıæðy ø œñçðł łñ þy Thøıg!!!!!!";

    std::u32string sanitizedText1 = dictionaryService.obtainWords(inputText1);
    std::u32string sanitizedText2 = dictionaryService.obtainWords(inputText2);

    dictionaryService.populateDictionary("My Custom 1", sanitizedText1);
    dictionaryService.populateDictionary("My Custom 2", sanitizedText2);

    double probability1 = dictionaryService.similarityByCommonWords("My Custom 1", "My Custom 2");
    double probability2 = dictionaryService.similarityByCommonSymbols(inputText1, inputText2);

    std::cout << "Probability by common words:   " << probability1 * 100 << "%." << std::endl;
    std::cout << "Probability by common symbols: " << probability2 * 100 << "%." << std::endl;
    system("pause");
#if 0

    std::cout << dictionaryService.checkWords("English", U"yellow stewardesses pneumonoultramicroscopicsilicovolcanoconiosis") << std::endl;
    std::cout << dictionaryService.checkWords("Italian", U"precipitevolissimevolmente epicità") << std::endl;
    std::cout << dictionaryService.checkWords("French", U"transsubstantieraient mélancoliquement") << std::endl;
    std::cout << dictionaryService.checkWords("Spanish", U"querido misericordiosamente") << std::endl;

    HostTrie dictionaryArabic;
    initDictionary("Arabic", &dictionaryArabic, "./arabic2.txt");
    dictionaryArabic.buildTrie(10);
    std::u32string arabicHospital = { 0x0645, 0x0633, 0x062A, 0x0634, 0x0641, 0x0649 }; //مستشفى
    std::u32string arabicStewardess = { 0x0645, 0x0636, 0x064A, 0x0641, 0x0629 }; // مضيفة
    dictionaryArabic.searchFromHost(arabicHospital);
    dictionaryArabic.searchFromHost(arabicStewardess);

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
#endif 

}
