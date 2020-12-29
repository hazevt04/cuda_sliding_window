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

}


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
      float2 temp0 = samples[index];
      float2 temp1 = samples[index + 1];

      // Due to overlap between consecutive windows the
      // loop logic stays the same as the original!
      for ( int w_index = 1; w_index < window_size; ++w_index ) {
         temp0 += samples[index + w_index];
         temp1 += samples[index + w_index + 1];
      }

      window_sums[index] = temp0;
      window_sums[index + 1] = temp1;
   } 

} // end of __global__ void sliding_window_unrolled_2x


// Can't work due to float4 aligment making it impossible to 
// have consecutive windows overlap
__global__ void sliding_window_vectorized_loads(
   cufftComplex* __restrict__ window_sums, 
   cufftComplex* const __restrict__ samples, 
   const int window_size, 
   const int num_windowed_samples ) {
   
   // Assuming one stream
   int global_index = blockIdx.x * blockDim.x + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for( int index = global_index; index < num_windowed_samples/2; index+=stride ) {
      float4 temp = make_float4( 0.f, 0.f, 0.f, 0.f );
      float2 temp2 = make_float2( 0.f, 0.f ); 
      
      for( int w_index = 0; w_index < window_size/2; ++w_index ) {
         if ( ( blockIdx.x == 0 ) && ( threadIdx.x == 1 ) ) {
            printf( "\t(float4) samples[%d + %d] = { %f, %f, %f, %f }\n", 
               index, w_index,
               reinterpret_cast<float4*>(samples)[index + w_index].x,
               reinterpret_cast<float4*>(samples)[index + w_index].y,
               reinterpret_cast<float4*>(samples)[index + w_index].z,
               reinterpret_cast<float4*>(samples)[index + w_index].w
            );
         }

         temp = reinterpret_cast<float4*>(samples)[index + w_index];

         temp2.x += temp.x;
         temp2.x += temp.z;
         temp2.y += temp.y;
         temp2.y += temp.w;
         
         if ( ( blockIdx.x == 0 ) && ( threadIdx.x == 1 ) ) {
            printf( "\t(float2) temp2 = { %f, %f }\n\n", temp2.x, temp2.y );
         }
      }

      //reinterpret_cast<float4*>(window_sums)[index] = temp;
      window_sums[index] = temp2;
   } 
}

