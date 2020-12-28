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

      /*window_sums[index] = make_cuFloatComplex( t_window_sum.x, t_window_sum.y );*/
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
      float2 temp0 = samples[index];
      float2 temp1 = samples[index + 1];

      for ( int w_index = 1; w_index < window_size; ++w_index ) {
         temp0 += samples[index + w_index];
         temp1 += samples[index + w_index + 1];
      }

      window_sums[index] = temp0;
      window_sums[index + 1] = temp1;
   } 


   //for ( int index = global_index; index < num_windowed_samples/2; index+=stride ) {
      
   //   float4 t_two_window_sums = make_float4( 0.f, 0.f, 0.f, 0.f );

   //   for ( int w_index = 0; w_index < window_size/2; ++w_index ) {
   //      if ( ( threadIdx.x == 0 ) && ( blockIdx.x == 0) ) { 
   //         printf( "\tblockIdx.x is %d, index is %d and w_index is %d, t_two_window_sums before is { %f, %f, %f, %f }\n", 
   //            blockIdx.x, index, w_index,
   //            t_two_window_sums.x, t_two_window_sums.y, t_two_window_sums.z, t_two_window_sums.w );
   //      }

   //      t_two_window_sums += reinterpret_cast<float4*>(samples)[index + w_index];
         
   //      if ( ( threadIdx.x == 0 ) && ( blockIdx.x == 0) ) { 
   //         printf( "\tblockIdx.x is %d, index is %d and w_index is %d, samples[%d + %d] as float4 is { %f, %f, %f, %f }\n", 
   //            blockIdx.x, index, w_index, index, w_index,
   //            reinterpret_cast<float4*>(samples)[index + w_index].x,
   //            reinterpret_cast<float4*>(samples)[index + w_index].y,
   //            reinterpret_cast<float4*>(samples)[index + w_index].z,
   //            reinterpret_cast<float4*>(samples)[index + w_index].w
   //         );
   //         printf( "\tblockIdx.x is %d, index is %d and w_index is %d, t_two_window_sums after is { %f, %f, %f, %f }\n\n", 
   //            blockIdx.x, index, w_index,
   //            t_two_window_sums.x, t_two_window_sums.y, t_two_window_sums.z, t_two_window_sums.w );
   //      }

   //   }
      
   //   reinterpret_cast<float4*>(window_sums)[index] = t_two_window_sums;
   //} // end of for ( int index = global_index; index < num_windowed_samples/2; index+=stride ) {

   //if ( ( global_index == num_windowed_samples/2 ) && ( ( num_windowed_samples & 1 ) == 1 ) ) {
      
   //   cufftComplex t_window_sum = samples[global_index];

   //   for( int w_index = 1; w_index < window_size; ++w_index ) {
   //      t_window_sum = cuCaddf( t_window_sum, samples[global_index + w_index] );
   //   }

   //   window_sums[global_index] = t_window_sum;
   //}
} // end of __global__ void sliding_window_vectorized_loads


__global__ void sliding_window_vectorized_loads2(
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
      for ( int w_index = 1; w_index < window_size; w_index +=2 ) {
         temp += reinterpret_cast<float4*>(samples)[index + w_index];
         temp2.x += temp.x + temp.z;
         temp2.y += temp.y + temp.w;
      }

      //reinterpret_cast<float4*>(window_sums)[index] = temp;
      window_sums[index] = temp2;
      window_sums[index] = temp2;
   } 
}

