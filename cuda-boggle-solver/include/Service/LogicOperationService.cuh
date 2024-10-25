#include <vector>
#include <string>

class LogicOperationService {
    LogicOperationService() {}
public:
    LogicOperationService(const LogicOperationService&) = delete;
    LogicOperationService& operator=(const LogicOperationService&) = delete;
    static LogicOperationService& get_instance();

    std::vector<int> splitStringToVector(const std::string& input);
    std::string logicalOrStrings(const std::string& str1, const std::string& str2);
    std::string logicalAndStrings(const std::string& str1, const std::string& str2);
    std::string logicalXorStrings(const std::string& str1, const std::string& str2);
    std::string logicalNotString(const std::string& input);
    std::vector<std::u32string> splitU32StringToWords(const std::u32string& input);
    std::vector<std::u32string> filterWordsByBoolean(const std::u32string& utf32Str, const std::string& boolCsv);
};