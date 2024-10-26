#include <iostream>
#include <sstream>
#include <vector>

// Function to split a comma-separated string into a vector of integers
std::vector<int> splitStringToVector(const std::string& input) {
    std::vector<int> result;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, ',')) {
        result.push_back(std::stoi(token));  // Convert each token to an integer (0 or 1)
    }

    return result;
}

// Function to perform logical OR between two comma-separated strings
std::string logicalOrStrings(const std::string& str1, const std::string& str2) {
    std::vector<int> vec1 = splitStringToVector(str1);
    std::vector<int> vec2 = splitStringToVector(str2);

    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Input strings must have the same number of elements.");
    }

    std::stringstream result;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result << (vec1[i] | vec2[i]);  // Perform logical OR
        if (i < vec1.size() - 1) {
            result << ',';  // Add comma separator
        }
    }

    return result.str();
}

// Function to perform logical AND between two comma-separated strings
std::string logicalAndStrings(const std::string& str1, const std::string& str2) {
    std::vector<int> vec1 = splitStringToVector(str1);
    std::vector<int> vec2 = splitStringToVector(str2);

    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Input strings must have the same number of elements.");
    }

    std::stringstream result;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result << (vec1[i] & vec2[i]);  // Perform logical AND
        if (i < vec1.size() - 1) {
            result << ',';  // Add comma separator
        }
    }

    return result.str();
}

// Function to perform logical XOR between two comma-separated strings
std::string logicalXorStrings(const std::string& str1, const std::string& str2) {
    std::vector<int> vec1 = splitStringToVector(str1);
    std::vector<int> vec2 = splitStringToVector(str2);

    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Input strings must have the same number of elements.");
    }

    std::stringstream result;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result << (vec1[i] ^ vec2[i]);  // Perform logical XOR
        if (i < vec1.size() - 1) {
            result << ',';  // Add comma separator
        }
    }

    return result.str();
}

// Function to perform logical NOT on a comma-separated string
std::string logicalNotString(const std::string& input) {
    std::vector<int> vec = splitStringToVector(input);
    std::stringstream result;

    for (size_t i = 0; i < vec.size(); ++i) {
        result << (vec[i] == 0 ? 1 : 0);  // Perform logical NOT
        if (i < vec.size() - 1) {
            result << ',';  // Add comma separator
        }
    }

    return result.str();
}

// Helper function to split a std::u32string (UTF-32 string) into words
std::vector<std::u32string> splitU32StringToWords(const std::u32string& input) {
    std::vector<std::u32string> words;
    std::u32string word;
    for (auto c : input) {
        if (c == U' ' || c == U',' || c == U'.' || c == U'\n' || c == U'\t') {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        else {
            word += c;
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    return words;
}

// Function to filter words in a UTF-32 string based on a CSV boolean string
std::vector<std::u32string> filterWordsByBoolean(const std::u32string& utf32Str, const std::string& boolCsv) {
    std::vector<std::u32string> words = splitU32StringToWords(utf32Str);
    std::vector<int> bools = splitStringToVector(boolCsv);

    if (words.size() != bools.size()) {
        throw std::invalid_argument("Mismatch between number of words and number of booleans.");
    }

    std::vector<std::u32string> result;
    for (size_t i = 0; i < words.size(); ++i) {
        if (bools[i] == 1) {
            result.push_back(words[i]);  // Collect words where boolean is true (1)
        }
    }

    return result;
}

double getAccuracyFromBooleanMask(const std::string& boolCsv) {
    std::vector<int> vec = splitStringToVector(boolCsv);
    size_t count_one = 0;
    for (auto aBool : vec) {
        if (aBool == 1) {
            count_one++;
        }
    }
    return (double)count_one / vec.size();
}
