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


// Simplest Shared Memory Implementation
// Accumulate windows of samples from global memory into the shared memory
__global__ void sliding_window_sh_mem_simple( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {

   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;

   extern __shared__ cufftComplex sh_samples[];
   sh_samples[threadIdx.x] = make_cuFloatComplex( 0.f, 0.f );
   __syncthreads();

   for( int index = global_index; index < num_windowed_samples; index+=stride ) {
      
      for( int w_index = 0; w_index < window_size; ++w_index ) {
         sh_samples[threadIdx.x] = cuCaddf( sh_samples[threadIdx.x], samples[ global_index + w_index ] );
         __syncthreads();
      }

      window_sums[global_index] = sh_samples[threadIdx.x];
   }
}

// Multi-Window Shared Memory Implementation
// Shared memory is large enough for blockDim.x windows worth of samples being loaded from global memory.
// Then accumulate the windows such that the first index of each window, index = threadIdx.x * window_size, has the sum for that window.
//
// Less threads per block are possible since each block is using more shared memory, however.
// Also, there are probably multiple copies of samples from global memory given how sliding window works.
// This makes this implementation wasteful. This is hopefully a 'stepping-stone' to trying out reduce for sliding window
__global__ void sliding_window_sh_mem_multi_window( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {

   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;

   extern __shared__ cufftComplex sh_samples[];
   sh_samples[threadIdx.x] = make_cuFloatComplex( 0.f, 0.f );
   __syncthreads();

   for( int index = global_index; index < num_windowed_samples; index+=stride ) {
      // Load blockDim.x windows worth of samples into shared memory
      for( int w_index = 0; w_index < window_size; ++w_index ) {
         sh_samples[window_size * threadIdx.x + w_index] = samples[ global_index + w_index ];
      }
      __syncthreads();

      // Accummulate the windows in shared memory. At the end, the sum for each window will be in the
      // the first location for each window, index = threadIdx.x * window_size
      for( int w_index = 1; w_index < window_size; ++w_index ) {
         sh_samples[threadIdx.x * window_size] = cuCaddf( sh_samples[threadIdx.x * window_size], sh_samples[threadIdx.x * window_size + w_index] );
         __syncthreads();
      }

      window_sums[global_index] = sh_samples[threadIdx.x * window_size];
   }
}


// Multi Window Reduce Shared Memory Implementation
__global__ void sliding_window_sh_mem_multi_window_reduce( cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {

   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;

   extern __shared__ cufftComplex sh_samples[];
   sh_samples[threadIdx.x] = make_cuFloatComplex( 0.f, 0.f );
   __syncthreads();

   for( int index = global_index; index < num_windowed_samples; index+=stride ) {
      // Load blockDim.x windows worth of samples into shared memory
      // Each thread loads 2 samples
      for( int w_index = 0; w_index < window_size; ++w_index ) {
         sh_samples[(threadIdx.x * window_size) + w_index] = samples[ global_index + w_index ];
      }
      __syncthreads();

      // Reduce the windows in shared memory. 
      // At the end, the sum for each window will be in the
      // the first location for each window, index = threadIdx.x * window_size
      for( unsigned int s = window_size/2; s > 0; s>>=1) {
         if ( threadIdx.x < s ) {
            for( int hw_index = 0; hw_index < window_size/2; ++hw_index ) {
               sh_samples[ (threadIdx.x * window_size) + hw_index ] = cuCaddf( sh_samples[ (threadIdx.x * window_size) + hw_index ], 
                  sh_samples[ (threadIdx.x * window_size) + hw_index + s ] );
            }
         }
         __syncthreads();
      } 

      window_sums[global_index] = sh_samples[threadIdx.x * window_size];
   }
}
