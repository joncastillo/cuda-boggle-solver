#pragma once

// Define the maximum number of children for a Trie node
#define MAX_CHILDREN (26 + 10 + 7 + 1) // 26 letters, 10 digits, 7 symbols (- . / ' , & !), 1 for word end indicator

// Function to map a character to an index
inline int charToIndex(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - 'a'; // a-z mapped to 0-25
    }
    if (c >= '0' && c <= '9') {
        return 26 + (c - '0'); // 0-9 mapped to 26-35
    }
    switch (c) {
    case '-': return 36;
    case '.': return 37;
    case '/': return 38;
    case '\'': return 39;
    case ',': return 40;
    case '&': return 41;
    case '!': return 42;

    default: return -1; // Invalid character
    }
}
