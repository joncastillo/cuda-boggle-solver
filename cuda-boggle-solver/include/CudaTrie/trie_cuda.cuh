#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ int charToIndex_gpu(char c);

extern __global__ void searchTrie(const int* d_trieData, const char* d_word, bool* d_found, int d_word_len);