#ifndef __ADD_KERNEL__
#define __ADD_KERNEL__

#include "my_cuda_utils.hpp"

//////////////////////////////////////
// THE Kernel (add)
// Kernel function to add the elements 
// of two arrays x_vals and y_vals. 
//////////////////////////////////////
__global__ void add(float4* __restrict__ d_results, float4* const __restrict__ d_x_vals, 
    float4* const __restrict__ d_y_vals, const int num_items );




#endif
