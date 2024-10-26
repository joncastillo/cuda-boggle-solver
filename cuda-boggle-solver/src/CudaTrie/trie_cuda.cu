#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaTrie\trie_common.h"

__device__ int charToIndex_gpu(char32_t c) {
    int index = 0;

    if (c >= U'A' && c <= U'Z') {
        c = c + (U'a' - U'A');
    }

    if (c >= U'a' && c <= U'z') {
        return index + (c - U'a');
    }
    index += 26;

    if (c >= U'0' && c <= U'9') {
        return index + (c - U'0');
    }
    index += 10;

    switch (c) {
    case U'-': return index;
    case U'.': return index + 1;
    case U'/': return index + 2;
    case U'\'': return index + 3;
    case U',': return index + 4;
    case U'&': return index + 5;
    case U'!': return index + 6;
    case U'#': return index + 7;
    }
    index += 8;

    if (c >= U'\u0600' && c <= U'\u06FF') {
        return index + (c - U'\u0600');
    }
    index += (0x06FF - 0x0600 + 1);

    if (c >= U'\u0750' && c <= U'\u077F') {
        return index + (c - U'\u0750');
    }
    index += (0x077F - 0x0750 + 1);

    if (c >= U'\uFB50' && c <= U'\uFDFF') {
        return index + (c - U'\uFB50');
    }
    index += (0xFDFF - 0xFB50 + 1);

    if (c >= U'\uFE70' && c <= U'\uFEFF') {
        return index + (c - U'\uFE70');
    }
    index += (0xFEFF - 0xFE70 + 1);

    if (c >= U'\u00A0' && c <= U'\u00FF') {
        return index + (c - U'\u00A0');
    }
    index += (0x00FF - 0x00A0 + 1);

    if (c >= U'\u0100' && c <= U'\u017F') {
        return index + (c - U'\u0100');
    }
    index += (0x017F - 0x0100 + 1);

    if (c >= U'\u0180' && c <= U'\u024F') {
        return index + (c - U'\u0180');
    }
    index += (0x024F - 0x0180 + 1);

    if (c >= U'\u0400' && c <= U'\u04FF') {
        return index + (c - U'\u0400');
    }
    index += (0x04FF - 0x0400 + 1);

    return -1;
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