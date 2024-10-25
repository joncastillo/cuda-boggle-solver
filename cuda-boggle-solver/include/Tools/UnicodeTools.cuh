#pragma once

#include <string>
#include <vector>

extern void printRawBytes(const std::string& line);
extern std::wstring utf32ToWstring(const std::u32string& utf32Str);
extern std::wstring utf8ToWstring(const std::string& utf8Str);
extern std::u32string utf8ToUtf32(const std::string& utf8Str);
extern std::wstring utf32ToWstring(const std::u32string& utf32Str);
extern std::vector<std::u32string> filterWordsByBoolean(const std::u32string& utf32Str, const std::string& boolCsv);
