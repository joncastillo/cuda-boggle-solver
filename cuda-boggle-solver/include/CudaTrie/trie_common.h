#pragma once

// Define the maximum number of children for a Trie node
#define MAX_CHILDREN (1213+(0xFEFF−0xFE70)+1)

// Function to map a character to an index
inline char32_t idxToChar(int index) {
    if (index >= 0 && index <= 25) {
        return U'a' + index; // English lowercase letters (a-z)
    }
    else if (index >= 26 && index <= 35) {
        return U'0' + (index - 26); // Digits (0-9)
    }

    switch (index) {
    case 36: return U'-';
    case 37: return U'.';
    case 38: return U'/';
    case 39: return U'\'';
    case 40: return U',';
    case 41: return U'&';
    case 42: return U'!';
    case 43: return U'#';
    }

    if (index >= 300 && index <= 395) {
        return U'\u00A0' + (index - 300); // Latin-1 Supplement
    }
    else if (index >= 396 && index <= 519) {
        return U'\u0100' + (index - 396); // Latin Extended-A
    }
    else if (index >= 520 && index <= 696) {
        return U'\u0180' + (index - 520); // Latin Extended-B
    }
    else if (index >= 697 && index <= 952) {
        return U'\u0400' + (index - 697); // Cyrillic Unicode block
    }
    else if (index >= 953 && index <= 982) {
        return U'\u0750' + (index - 953); // Arabic Supplement
    }
    else if (index >= 983 && index <= 1212) {
        return U'\uFB50' + (index - 983); // Arabic Presentation Forms-A
    }
    else if (index >= 1213 && index <= 1242) {
        return U'\uFE70' + (index - 1213); // Arabic Presentation Forms-B
    }

    return U'\0'; // Invalid index
}

inline int charToIndex(char32_t c) {
    // Convert uppercase to lowercase if necessary
    if (c >= U'A' && c <= U'Z') {
        c = c + (U'a' - U'A'); // Convert uppercase to lowercase
    }

    // Handle lowercase English letters (a-z)
    if (c >= U'a' && c <= U'z') {
        return c - U'a'; // English lowercase letters mapped to 0-25
    }

    // Handle digits (0-9)
    if (c >= U'0' && c <= U'9') {
        return 26 + (c - U'0'); // Digits mapped to 26-35
    }

    // Handle specific symbols: '-', '.', '/', '\'', ',', '&', '!', '#'
    switch (c) {
    case U'-': return 36;
    case U'.': return 37;
    case U'/': return 38;
    case U'\'': return 39;
    case U',': return 40;
    case U'&': return 41;
    case U'!': return 42;
    case U'#': return 43;
    }

    // Latin-1 Supplement (U+00A0 to U+00FF)
    if (c >= U'\u00A0' && c <= U'\u00FF') {
        return 300 + (c - U'\u00A0');
    }

    // Latin Extended-A (U+0100 to U+017F)
    if (c >= U'\u0100' && c <= U'\u017F') {
        return 396 + (c - U'\u0100');
    }

    // Latin Extended-B (U+0180 to U+024F)
    if (c >= U'\u0180' && c <= U'\u024F') {
        return 520 + (c - U'\u0180');
    }

    // Cyrillic Unicode block (U+0400 to U+04FF)
    if (c >= U'\u0400' && c <= U'\u04FF') {
        return 697 + (c - U'\u0400');
    }

    // Arabic Supplement (U+0750 to U+077F)
    if (c >= U'\u0750' && c <= U'\u077F') {
        return 953 + (c - U'\u0750');
    }

    // Arabic Presentation Forms-A (U+FB50 to U+FDFF)
    if (c >= U'\uFB50' && c <= U'\uFDFF') {
        return 983 + (c - U'\uFB50');
    }

    // Arabic Presentation Forms-B (U+FE70 to U+FEFF)
    if (c >= U'\uFE70' && c <= U'\uFEFF') {
        return 1213 + (c - U'\uFE70');
    }

    return -1; // Invalid character
}