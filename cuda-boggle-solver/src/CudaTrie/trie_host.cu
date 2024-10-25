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

#include "CudaTrie\trie_host.cuh"
#include "CudaTrie\trie_cuda.cuh"
#include "Tools\UnicodeTools.cuh"

// Add a word to the host-side Trie
void HostTrie::addWord(const std::u32string& word) {
    int index = 0;
    for (const char32_t& c : word) {

        // Ignore control characters like LRM (U+200E)
        if (c == U'\u200E' || c == U'\u200F') {
            continue; // Skip Left-to-Right Mark (U+200E) and Right-to-Left Mark (U+200F)
        }


        int charIndex = charToIndex(c);
        if (charIndex == -1) {
            std::wcerr << L"Invalid character in word: " << utf32ToWstring(word) << std::endl;
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

bool HostTrie::searchFromHost(const std::u32string& wordToSearchUtf32) {

    // Run CUDA kernel to test search for word
    bool h_found = false;
    bool* d_found;
    int d_word_len = static_cast<int>(wordToSearchUtf32.length());

    // Allocate memory for result on the device
    cudaMalloc((void**)&d_found, sizeof(bool));

    // Allocate memory for the word on the device (UTF-32)
    char32_t* d_word;
    cudaMalloc((void**)&d_word, d_word_len * sizeof(char32_t));

    std::wcout << "Searching for the word: " << utf32ToWstring(wordToSearchUtf32) << "... ";
    // Copy the UTF-32 word from host to device
    cudaMemcpy(d_word, wordToSearchUtf32.data(), d_word_len * sizeof(char32_t), cudaMemcpyHostToDevice);

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

// Helper function to tokenize the paragraph into words (UTF-32)
std::vector<std::u32string> tokenizeParagraph(const std::u32string& paragraph) {
    std::vector<std::u32string> words;
    std::u32string currentWord;

    for (char32_t c : paragraph) {
        if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U',' || c == U'.') {
            if (!currentWord.empty()) {
                words.push_back(currentWord);
                currentWord.clear();
            }
        }
        else {
            currentWord.push_back(c);
        }
    }

    // Add the last word if the string doesn't end with a delimiter
    if (!currentWord.empty()) {
        words.push_back(currentWord);
    }

    return words;
}

std::string HostTrie::searchFromHostParagraph(const std::u32string& paragraph) {
    // Tokenize the paragraph into individual words
    std::vector<std::u32string> words = tokenizeParagraph(paragraph);
    std::string result;

    for (const std::u32string& wordToSearchUtf32 : words) {
        bool h_found = false;
        bool* d_found;
        int d_word_len = static_cast<int>(wordToSearchUtf32.length());

        // Allocate memory for result on the device
        cudaMalloc((void**)&d_found, sizeof(bool));

        // Allocate memory for the word on the device (UTF-32)
        char32_t* d_word;
        cudaMalloc((void**)&d_word, d_word_len * sizeof(char32_t));

        // Copy the UTF-32 word from host to device
        cudaMemcpy(d_word, wordToSearchUtf32.data(), d_word_len * sizeof(char32_t), cudaMemcpyHostToDevice);

        // Launch the search kernel (adjusted to work with UTF-32)
        int gridSize = (d_word_len + 255) / 256;
        searchTrie << < gridSize, 256 >> > (m_pdev_trieData, d_word, d_found, d_word_len);
        cudaDeviceSynchronize();

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
            return "";
        }

        // Copy result back from device to host
        cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

        // Add the result (1 for found, 0 for not found) to the output string
        result += (h_found ? "1" : "0");
        result += ",";

        // Free device memory
        cudaFree(d_found);
        cudaFree(d_word);
    }

    // Remove the last comma
    if (!result.empty()) {
        result.pop_back();
    }

    return result;
}