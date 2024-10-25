#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaTrie\trie_common.h"

__device__ int charToIndex_gpu(char32_t c) {
    // Convert uppercase to lowercase if necessary
    if (c >= U'A' && c <= U'Z') {
        c = c + (U'a' - U'A'); // Convert uppercase to lowercase
    }

    // Handle lowercase English letters
    if (c >= U'a' && c <= U'z') {
        return c - U'a'; // English lowercase letters mapped to 0-25
    }

    // Handle digits (0-9)
    if (c >= U'0' && c <= U'9') {
        return 26 + (c - U'0'); // Digits mapped to 26-35
    }

    // Handle specific symbols: '-', '.', '/', '\'', ',', '&', '!'
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

    // Range-based mapping for Arabic Unicode block (U+0600 to U+06FF)
    if (c >= U'\u0600' && c <= U'\u06FF') {
        return 44 + (c - U'\u0600'); // Arabic characters mapped to 44-299
    }

    // Range-based mapping for Latin-1 Supplement (U+00A0 to U+00FF)
    if (c >= U'\u00A0' && c <= U'\u00FF') {
        return 300 + (c - U'\u00A0'); // Latin-1 Supplement mapped to 300-355
    }

    // Range-based mapping for Latin Extended-A (U+0100 to U+017F)
    if (c >= U'\u0100' && c <= U'\u017F') {
        return 356 + (c - U'\u0100'); // Latin Extended-A mapped to 356-479
    }

    // Range-based mapping for Latin Extended-B (U+0180 to U+024F)
    if (c >= U'\u0180' && c <= U'\u024F') {
        return 480 + (c - U'\u0180'); // Latin Extended-B mapped to 480-656
    }

    // Range-based mapping for Cyrillic Unicode block (U+0400 to U+04FF)
    if (c >= U'\u0400' && c <= U'\u04FF') {
        return 657 + (c - U'\u0400'); // Cyrillic characters mapped to 657-912
    }

    // Range-based mapping for Arabic Supplement (U+0750 to U+077F)
    if (c >= U'\u0750' && c <= U'\u077F') {
        return 300 + (c - U'\u0750'); // Adjust indices accordingly
    }

    // Range-based mapping for Arabic Presentation Forms-A (U+FB50 to U+FDFF)
    if (c >= U'\uFB50' && c <= U'\uFDFF') {
        return 330 + (c - U'\uFB50'); // Adjust indices accordingly
    }

    // Range-based mapping for Arabic Presentation Forms-B (U+FE70 to U+FEFF)
    if (c >= U'\uFE70' && c <= U'\uFEFF') {
        return 370 + (c - U'\uFE70'); // Adjust indices accordingly
    }

    return -1; // Invalid character
}

// CUDA kernel to search for a word in the device-side Trie
__device__ void searchTrie_i(const int* d_trieData, const char32_t* d_word, bool* d_found, int d_word_len) {
    int index = 0;

    for (int i = 0; i < d_word_len; i++) {
        char32_t c = d_word[i];
        int charIndex = charToIndex_gpu(c); // Use the charToIndex function in the device

        if (charIndex == -1) {
            *d_found = false;
            return; // Invalid character found
        }

        if (d_trieData[index * MAX_CHILDREN + charIndex] == -1) {
            *d_found = false;
            return;
        }

        index = d_trieData[index * MAX_CHILDREN + charIndex];
    }

    *d_found = d_trieData[index * MAX_CHILDREN + MAX_CHILDREN - 1] == 1;
}

__global__ void searchTrie(const int* d_trieData, const char32_t* d_word, bool* d_found, int d_word_len) {
    searchTrie_i(d_trieData, d_word, d_found, d_word_len);
}