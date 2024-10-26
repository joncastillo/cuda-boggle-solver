#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <thrust/device_vector.h>

#include "trie_common.h"

// C++ class representing a Trie node on the host side
class HostTrieNode {
public:
    int children[MAX_CHILDREN];
    bool isWordEnd;


    HostTrieNode() : isWordEnd(false) {
        // Initialize children as -1 to indicate non-existence
        std::fill(std::begin(children), std::end(children), -1);
    }
};

// C++ class representing the Trie on the host side
class HostTrie {
    thrust::device_vector<int> d_trieData;

public:
    int* m_pdev_trieData;
    size_t maxWordSize;

    std::vector<HostTrieNode> nodes;

    void addWord(const std::u32string& word);
    void buildTrie(size_t maxWordSize);
    bool searchFromHost(const std::u32string& wordToSearchUtf32);
    void collectWords(const HostTrie& trie, int nodeIndex, std::u32string currentWord, std::unordered_set<std::u32string>& words);
    std::string searchFromHostParagraph(const std::u32string& paragraph);
    void collectWords_i(int nodeIndex, std::u32string currentWord, std::unordered_set<std::u32string>& words);
    void collectWords(std::unordered_set<std::u32string>& words);


    HostTrie() : maxWordSize(0), m_pdev_trieData(nullptr) {
        nodes.push_back(HostTrieNode());  // Add root node
    }
};