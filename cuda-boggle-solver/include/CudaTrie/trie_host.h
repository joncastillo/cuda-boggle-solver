#pragma once

#include <vector>
#include <string>
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

    void addWord(const std::wstring& word);
    void buildTrie(size_t maxWordSize);
    bool searchFromHost(const std::wstring& wordToSearchUtf8);

    HostTrie() : nodes(MAX_CHILDREN),
        maxWordSize(0),
        m_pdev_trieData(nullptr) {}
};