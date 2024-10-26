#pragma once

// Define the maximum number of children for a Trie node
#define MAX_CHILDREN (1213+(0xFEFF−0xFE70)+1)

// Function to map a character to an index
inline char32_t idxToChar(int index) {
    int baseIndex = 0;

    if (index >= baseIndex && index < baseIndex + 26) {
        return U'a' + (index - baseIndex);
    }
    baseIndex += 26;

    if (index >= baseIndex && index < baseIndex + 10) {
        return U'0' + (index - baseIndex);
    }
    baseIndex += 10;

    if (index >= baseIndex && index < baseIndex + 8) {
        switch (index - baseIndex) {
        case 0: return U'-';
        case 1: return U'.';
        case 2: return U'/';
        case 3: return U'\'';
        case 4: return U',';
        case 5: return U'&';
        case 6: return U'!';
        case 7: return U'#';
        }
    }
    baseIndex += 8;

    if (index >= baseIndex && index < baseIndex + (0x06FF - 0x0600 + 1)) {
        return U'\u0600' + (index - baseIndex);
    }
    baseIndex += (0x06FF - 0x0600 + 1);

    if (index >= baseIndex && index < baseIndex + (0x077F - 0x0750 + 1)) {
        return U'\u0750' + (index - baseIndex);
    }
    baseIndex += (0x077F - 0x0750 + 1);

    if (index >= baseIndex && index < baseIndex + (0xFDFF - 0xFB50 + 1)) {
        return U'\uFB50' + (index - baseIndex);
    }
    baseIndex += (0xFDFF - 0xFB50 + 1);

    if (index >= baseIndex && index < baseIndex + (0xFEFF - 0xFE70 + 1)) {
        return U'\uFE70' + (index - baseIndex);
    }
    baseIndex += (0xFEFF - 0xFE70 + 1);

    if (index >= baseIndex && index < baseIndex + (0x00FF - 0x00A0 + 1)) {
        return U'\u00A0' + (index - baseIndex);
    }
    baseIndex += (0x00FF - 0x00A0 + 1);

    if (index >= baseIndex && index < baseIndex + (0x017F - 0x0100 + 1)) {
        return U'\u0100' + (index - baseIndex);
    }
    baseIndex += (0x017F - 0x0100 + 1);

    if (index >= baseIndex && index < baseIndex + (0x024F - 0x0180 + 1)) {
        return U'\u0180' + (index - baseIndex);
    }
    baseIndex += (0x024F - 0x0180 + 1);

    if (index >= baseIndex && index < baseIndex + (0x04FF - 0x0400 + 1)) {
        return U'\u0400' + (index - baseIndex);
    }
    baseIndex += (0x04FF - 0x0400 + 1);

    return U'\uFFFD'; // Return the Unicode replacement character for invalid index
}

inline int charToIndex(char32_t c) {
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