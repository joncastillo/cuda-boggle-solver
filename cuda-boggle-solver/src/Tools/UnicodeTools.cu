#include <iostream>
#include <string>
#include <iomanip>
#include <locale>
#include <codecvt>

void printRawBytes(const std::string& line) {
    std::cout << "Raw bytes: ";
    for (unsigned char c : line) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c) << " ";
    }
    std::cout << std::dec << std::endl;
}

std::wstring utf8ToWstring(const std::string& utf8Str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(utf8Str);
}

std::u32string utf8ToUtf32(const std::string& utf8Str) {
    std::u32string utf32Str;
    size_t i = 0;
    while (i < utf8Str.size()) {
        char32_t codepoint = 0;
        unsigned char c = utf8Str[i];

        if (c <= 0x7F) {
            codepoint = c;
            i += 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= utf8Str.size()) 
                break; // invalid sequence
            codepoint = ((c & 0x1F) << 6) | (utf8Str[i + 1] & 0x3F);
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= utf8Str.size()) 
                break; // invalid sequence
            codepoint = ((c & 0x0F) << 12) | ((utf8Str[i + 1] & 0x3F) << 6) | (utf8Str[i + 2] & 0x3F);
            i += 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= utf8Str.size()) 
                break; // invalid sequence
            codepoint = ((c & 0x07) << 18) | ((utf8Str[i + 1] & 0x3F) << 12) | ((utf8Str[i + 2] & 0x3F) << 6) | (utf8Str[i + 3] & 0x3F);
            i += 4;
        }
        else {
            // invalid byte
            break;
        }

        utf32Str.push_back(codepoint);
    }

    return utf32Str;
}

std::wstring utf32ToWstring(const std::u32string& utf32Str) {
    std::wstring wstr;
    for (char32_t c : utf32Str) {
        if (c <= 0xFFFF) {
            wstr.push_back(static_cast<wchar_t>(c));
        }
        else {
            c -= 0x10000;
            wstr.push_back(static_cast<wchar_t>((c >> 10) + 0xD800));
            wstr.push_back(static_cast<wchar_t>((c & 0x3FF) + 0xDC00));
        }
    }
    return wstr;
}
