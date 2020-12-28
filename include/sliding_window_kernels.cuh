#pragma once

#include "my_cufft_utils.hpp"

///////////////////////////////////////////////////////////////////////////////////
// Sliding Window
// Kernel function to compute sliding window sums
// of samples such that:
// For N samples and window size, W:
//    window_sums[0] = samples[0] + samples[1] + ... + samples[N-W-1]
//    window_sums[1] = samples[1] + samples[2] + ... + samples[N-W]
//    window_sums[2] = samples[2] + samples[3] + ... + samples[N-W+1]
//    ...
//    window_sums[N-W-1] = samples[N-W-1] + samples[N-W] + ... + samples[N-1]
///////////////////////////////////////////////////////////////////////////////////

// Original Implementation
__global__ void sliding_window_original( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples );

// Shared Memory Implementation
__global__ void sliding_window_sh_mem( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples );

// Vectorized Loads Implementation
__global__ void sliding_window_vectorized_loads( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples );

// Vectorized Loads Implementation 2
__global__ void sliding_window_vectorized_loads2( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples );
