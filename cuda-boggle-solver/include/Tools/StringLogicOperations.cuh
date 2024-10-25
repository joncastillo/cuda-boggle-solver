#pragma once

#include <iostream>
#include <sstream>
#include <vector>

extern std::vector<int> splitStringToVector(const std::string& input);
extern std::string logicalOrStrings(const std::string& str1, const std::string& str2);
extern std::string logicalAndStrings(const std::string& str1, const std::string& str2);
extern std::string logicalXorStrings(const std::string& str1, const std::string& str2);
extern std::string logicalNotString(const std::string& input);
extern std::vector<std::u32string> splitU32StringToWords(const std::u32string& input);
extern std::vector<std::u32string> filterWordsByBoolean(const std::u32string& utf32Str, const std::string& boolCsv);