#pragma once

// Define the maximum number of children for a Trie node
#define MAX_SYMBOLS 16  // Number of symbols like '-', '.', '/', etc.
#define OFFSET_SYMBOLS 36 // Adjust offset for symbols in the index
#define MAX_CHILDREN (26 + 10 + MAX_SYMBOLS + 1 + 256) // Change: Adjusted for Unicode characters (256 for additional Unicode support)

// Function to map a character to an index
inline int charToIndex(wchar_t c) {
    // Convert uppercase to lowercase if necessary
    if (c >= L'A' && c <= L'Z') {
        c = c + (L'a' - L'A'); // Convert uppercase to lowercase
    }

    // Handle lowercase English letters
    if (c >= L'a' && c <= L'z') {
        return c - L'a'; // English lowercase letters
    }

    // Handle digits (0-9)
    if (c >= L'0' && c <= L'9') {
        return 26 + (c - L'0'); // Digits follow letters
    }

    // Handle specific symbols: '-', '.', '/', '\'', ',', '&', '!', accented characters
    switch (c) {
        case L'-': return 36;
        case L'.': return 37;
        case L'/': return 38;
        case L'\'': return 39;
        case L',': return 40;
        case L'&': return 41;
        case L'!': return 42;
        case L'à': return 43;
        case L'è': return 44;
        case L'ì': return 45;
        case L'ò': return 46;
        case L'ù': return 47;
        case L'é': return 48;
        case L'â': return 49;
        case L'ô': return 50;
        case L'ç': return 51;
    }

    // Handle Arabic Unicode block (U+0600 to U+06FF)
    if (c >= L'\u0600' && c <= L'\u06FF') {
        return 48 + (c - L'\u0600'); // Arabic characters start at index 48
    }

    return -1; // Invalid character
}

