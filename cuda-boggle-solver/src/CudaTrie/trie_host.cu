#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <codecvt>
#include <locale>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "CudaTrie\trie_host.h"
#include "CudaTrie\trie_cuda.cuh"

std::u32string wstringToUtf32(const std::wstring& wstr) {
    std::u32string utf32Str;
    for (wchar_t wc : wstr) {
        utf32Str.push_back(static_cast<char32_t>(wc));
    }
    return utf32Str;
}

// Add a word to the host-side Trie
void HostTrie::addWord(const std::wstring& word) {
    std::u32string utf32Word = wstringToUtf32(word); // Convert to UTF-32
    int index = 0;
    for (const char32_t& c : utf32Word) {
        int charIndex = charToIndex(c);
        if (charIndex == -1) {
            std::wcerr << "Invalid character in word: " << word << std::endl;
            return;
        }

        if (nodes[index].children[charIndex] == -1) {
            nodes.push_back(HostTrieNode());
            nodes[index].children[charIndex] = static_cast<int>(nodes.size()) - 1;
        }
        index = nodes[index].children[charIndex];
    }
    nodes[index].isWordEnd = true;
}

// Build the flatten Trie data array to be transferred to the device. 
void HostTrie::buildTrie(size_t maxWordSize) {
    this->maxWordSize = maxWordSize;
    // Determine the total number of nodes in the Trie
    int numNodes = nodes.size();

    // Create a flatten array to store the Trie data on the device
    thrust::host_vector<int> h_trieData(numNodes * MAX_CHILDREN);

    // Traverse the Trie and fill the flatten array
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
        const HostTrieNode& node = nodes[i];
        std::copy(std::begin(node.children), std::end(node.children), h_trieData.begin() + i * MAX_CHILDREN);
        h_trieData[i * MAX_CHILDREN + MAX_CHILDREN - 1] = node.isWordEnd ? 1 : 0;
    }

    // Transfer the flatten Trie data to the device
    d_trieData.resize(h_trieData.size());
    thrust::copy(h_trieData.begin(), h_trieData.end(), d_trieData.begin());

    // Store the device-side trie data pointer for CUDA kernel usage
    m_pdev_trieData = thrust::raw_pointer_cast(d_trieData.data());
}

bool HostTrie::searchFromHost(const std::wstring& wordToSearchUtf8) {
    std::u32string wordToSearch = wstringToUtf32(wordToSearchUtf8);

    // Run CUDA kernel to test search for word
    bool h_found = false;
    bool* d_found;
    int d_word_len = static_cast<int>(wordToSearch.length());

    // Allocate memory for result on the device
    cudaMalloc((void**)&d_found, sizeof(bool));

    // Allocate memory for the word on the device (UTF-32)
    char32_t* d_word;
    cudaMalloc((void**)&d_word, d_word_len * sizeof(char32_t));

    std::wcout << "Searching for the word: " << wordToSearchUtf8 << "... ";
    // Copy the UTF-32 word from host to device
    cudaMemcpy(d_word, wordToSearch.data(), d_word_len * sizeof(char32_t), cudaMemcpyHostToDevice);

    // Launch the search kernel (adjusted to work with UTF-32)
    int gridSize = (d_word_len + 255) / 256;
    searchTrie << <gridSize, 256 >> > (m_pdev_trieData, d_word, d_found, d_word_len);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Copy result back from device to host
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    if (h_found) {
        std::cout << "Found!" << std::endl;
    }
    else {
        std::cout << "Not Found!" << std::endl;
    }

    // Free device memory
    cudaFree(d_found);
    cudaFree(d_word);

    return h_found;
}