#include "sliding_window_kernels.cuh"

//////////////////////////////////////
// Calculate sliding window sums
//////////////////////////////////////
__global__ void sliding_window_original(
   cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {
   
   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for ( int index = global_index; index < num_windowed_samples; index+=stride ) {
      cufftComplex t_window_sum = samples[index];

      for ( int w_index = 1; w_index < window_size; ++w_index ) {
         t_window_sum = cuCaddf( t_window_sum, samples[index + w_index] );
      }

      window_sums[index] = t_window_sum;
   }
} // end of __global__ void sliding_window_original


// Shared Memory Implementation
__global__ void sliding_window_sh_mem( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {

   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;

   __shared__ cufftComplex sh_samples[6144];

   for ( int index = global_index; index < num_windowed_samples; index+=stride ) {
      
      for( int w_index = 0; w_index < window_size; ++w_index ) {
         sh_samples[threadIdx.x * window_size + w_index] = samples[index + w_index];
      }
      __syncthreads();
      
      for( int w_index = 0; w_index < window_size; ++w_index ) {
         sh_samples[blockIdx.x * window_size] = cuCaddf( sh_samples[blockIdx.x * window_size], 
            sh_samples[threadIdx.x * window_size + w_index] );
      }
      __syncthreads();

      if (threadIdx.x == 0) {
         window_sums[index] = sh_samples[blockIdx.x * window_size];
      }
   }
} // end of Shared Memory Implementation


__global__ void sliding_window_unrolled_2x(
   cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {
   
   // Assuming one stream
   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for( int index = global_index; index < num_windowed_samples/2; index+=stride ) {
      float2 temp0 = samples[(index*2)];
      float2 temp1 = samples[(index*2) + 1];

      // Due to overlap between consecutive windows the
      // loop logic stays the same as the original!
      for ( int w_index = 1; w_index < window_size; ++w_index ) {
         temp0 += samples[(index*2) + w_index];
         temp1 += samples[(index*2) + w_index + 1];
      }

      window_sums[(index*2)] = temp0;
      window_sums[((index*2) + 1)] = temp1;
   } 

} // end of __global__ void sliding_window_unrolled_2x


__global__ void sliding_window_unrolled_4x(
   cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {
   
   // Assuming one stream
   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for( int index = global_index; index < num_windowed_samples/4; index+=stride ) {
      float2 temp0 = samples[(index*4)];
      float2 temp1 = samples[(index*4) + 1];
      float2 temp2 = samples[(index*4) + 2];
      float2 temp3 = samples[(index*4) + 3];

      // Due to overlap between consecutive windows the
      // loop logic stays the same as the original!
      for ( int w_index = 1; w_index < window_size; ++w_index ) {
         temp0 += samples[(index*4) + w_index];
         temp1 += samples[(index*4) + w_index + 1];
         temp2 += samples[(index*4) + w_index + 2];
         temp3 += samples[(index*4) + w_index + 3];
      }

      window_sums[(index*4)] = temp0;
      window_sums[(index*4) + 1] = temp1;
      window_sums[(index*4) + 2] = temp2;
      window_sums[(index*4) + 3] = temp3;
   } 

} // end of __global__ void sliding_window_unrolled_4x


__global__ void sliding_window_unrolled_8x(
   cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {
   
   // Assuming one stream
   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for( int index = global_index; index < num_windowed_samples/8; index+=stride ) {
      float2 temp0 = samples[(index*8)];
      float2 temp1 = samples[(index*8) + 1];
      float2 temp2 = samples[(index*8) + 2];
      float2 temp3 = samples[(index*8) + 3];
      float2 temp4 = samples[(index*8) + 4];
      float2 temp5 = samples[(index*8) + 5];
      float2 temp6 = samples[(index*8) + 6];
      float2 temp7 = samples[(index*8) + 7];

      // Due to overlap between consecutive windows the
      // loop logic stays the same as the original!
      for ( int w_index = 1; w_index < window_size; ++w_index ) {
         temp0 += samples[(index*8) + w_index];
         temp1 += samples[(index*8) + w_index + 1];
         temp2 += samples[(index*8) + w_index + 2];
         temp3 += samples[(index*8) + w_index + 3];
         temp4 += samples[(index*8) + w_index + 4];
         temp5 += samples[(index*8) + w_index + 5];
         temp6 += samples[(index*8) + w_index + 6];
         temp7 += samples[(index*8) + w_index + 7];
      }

      window_sums[(index*8)] = temp0;
      window_sums[(index*8) + 1] = temp1;
      window_sums[(index*8) + 2] = temp2;
      window_sums[(index*8) + 3] = temp3;
      window_sums[(index*8) + 4] = temp4;
      window_sums[(index*8) + 5] = temp5;
      window_sums[(index*8) + 6] = temp6;
      window_sums[(index*8) + 7] = temp7;
   } 

} // end of __global__ void sliding_window_unrolled_8x
