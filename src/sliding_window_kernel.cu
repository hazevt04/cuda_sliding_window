#include "sliding_window_kernel.cuh"

//////////////////////////////////////
// THE Kernel (sliding window)
// Calculate sliding window average
//////////////////////////////////////
__global__ void sliding_window(float2* __restrict__ results, float2* const __restrict__ vals, 
    const int window_size, const int num_items ) {
   
   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_items; index+=stride) {
      float2 t_val = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; w_index++) {
        t_val.x = t_val.x + vals[index + w_index].x;
        t_val.y = t_val.y + vals[index + w_index].y;
      }
      DIVIDE_COMPLEX_BY_SCALAR( t_val, t_val, (float)window_size );

      results[index].x = t_val.x;
      results[index].y = t_val.y;
   }
}

