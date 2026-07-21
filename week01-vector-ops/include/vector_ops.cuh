#pragma once
#include <cstddef>

void launch_vector_add(const float *a, 
    const float *b, 
    float *c, 
    std::size_t n, 
    int block_size);

void launch_saxpy(const float *a, 
    const float *b, 
    float *c, 
    float alpha, 
    std::size_t n, 
    int block_size);