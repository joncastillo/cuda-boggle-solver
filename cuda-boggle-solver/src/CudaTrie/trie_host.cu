#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "CudaTrie\trie_host.h"
#include "CudaTrie\trie_cuda.cuh"

// Add a word to the host-side Trie
void HostTrie::addWord(const std::string& word) {
    std::string wordLowerCased=word;

    std::transform(wordLowerCased.begin(), wordLowerCased.end(), wordLowerCased.begin(), [](unsigned char c) {
        if (std::isalpha(c)) {
            return std::tolower(c);
        }
        else {
            return (int)c; // Keep non-alpha characters unchanged
        }
        });

    int index = 0;
    for (const char& c : wordLowerCased) {
        int charIndex = charToIndex(c);
        if (charIndex == -1) {
            std::cerr << "Invalid character in word: " << word << std::endl;
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

bool HostTrie::searchFromHost(const char* wordToSearch) {
    // Run CUDA kernel to test search for word
    bool h_found = false;
    bool* d_found;
    int d_word_len = static_cast<int>(strlen(wordToSearch));

    cudaMalloc((void**)&d_found, sizeof(bool));
    // Calculate the grid size based on the word length
    int gridSize = (d_word_len + 255) / 256;
    char* d_word;
    cudaMalloc((void**)&d_word, d_word_len + 1);

    std::cout << "Searching for the word: " << wordToSearch << "... ";
    cudaMemcpy(d_word, wordToSearch, d_word_len + 1, cudaMemcpyHostToDevice);

    searchTrie << <gridSize, 256 >> > (m_pdev_trieData, d_word, d_found, d_word_len);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    if (h_found) {
        std::cout << "Found!" << std::endl;
    }
    else {
        std::cout << "Not Found!" << std::endl;
    }
    cudaFree(d_found);
    cudaFree(d_word);
    return h_found;
}