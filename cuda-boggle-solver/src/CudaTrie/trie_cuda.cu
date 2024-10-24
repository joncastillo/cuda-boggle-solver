#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaTrie\trie_common.h"

__device__ int charToIndex_gpu(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - 'a';
    }
    if (c >= '0' && c <= '9') {
        return 26 + (c - '0');
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

// CUDA kernel to search for a word in the device-side Trie
__device__ void searchTrie_i(const int* d_trieData, const char* d_word, bool* d_found, int d_word_len) {
    int index = 0;

    for (int i = 0; i < d_word_len; i++) {
        char c = d_word[i];
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

__global__ void searchTrie(const int* d_trieData, const char* d_word, bool* d_found, int d_word_len) {
    searchTrie_i(d_trieData, d_word, d_found, d_word_len);
}