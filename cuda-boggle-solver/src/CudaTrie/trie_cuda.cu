#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaTrie\trie_common.h"

__device__ int charToIndex_gpu(char32_t c) {
    // Handle lowercase English letters
    if (c >= U'a' && c <= U'z') {
        return c - U'a';
    }

    // Handle digits (0-9)
    if (c >= U'0' && c <= U'9') {
        return 26 + (c - U'0');
    }

    // Handle specific symbols: '-', '.', '/', '\'', ',', '&', '!', accented characters
    switch (c) {
    case U'-': return 36;
    case U'.': return 37;
    case U'/': return 38;
    case U'\'': return 39;
    case U',': return 40;
    case U'&': return 41;
    case U'!': return 42;
    case U'à': return 43;
    case U'è': return 44;
    case U'ì': return 45;
    case U'ò': return 46;
    case U'ù': return 47;
    case L'é': return 48;
    case L'â': return 49;
    case L'ô': return 50;
    case L'ô': return 51;
    }

    // Handle ASCII and extended characters up to 255
    if (c <= 255) {
        return 48 + c;
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