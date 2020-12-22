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
      cufftComplex t_window_sum = make_cuFloatComplex(0.f, 0.f);

      for ( int w_index = 0; w_index < window_size; ++w_index ) {
         t_window_sum = make_cuFloatComplex( ( t_window_sum.x + samples[index + w_index].x ),
            ( t_window_sum.y + samples[index + w_index].y ) );
      }

      window_sums[index] = make_cuFloatComplex( t_window_sum.x, t_window_sum.y );
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

   extern __shared__ cufftComplex sh_samples[];

   for( int index = global_index; index < num_windowed_samples; index+=stride ) {
      
      for( int w_index = 0; w_index < window_size; ++w_index ) {
         sh_samples[window_size * threadIdx.x + w_index] = samples[ global_index + w_index ];
      }
      __syncthreads();

      for( int w_index = 1; w_index < window_size; ++w_index ) {
         sh_samples[threadIdx.x * window_size] = cuCaddf( sh_samples[threadIdx.x * window_size], sh_samples[threadIdx.x * window_size + w_index] );
         __syncthreads();
      }

      window_sums[global_index] = sh_samples[threadIdx.x * window_size];
   }
}
