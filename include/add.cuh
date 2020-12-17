#ifndef __ADD_CUH__
#define __ADD_CUH__

#include <cuda_runtime.h>
#include "my_cuda_utils.hpp"

#define ROLLED_SEL_VAL (1<<0)
#define UNROLLED_FOUR_SEL_VAL (1<<1)
#define UNROLLED_EIGHT_SEL_VAL (1<<2)

#define RAND_MAX_RECIP (1.0f/(float)RAND_MAX)

int init_kernel( float4** x_vals, float4** y_vals, 
   float4** results, float4** exp_vals, float4** d_x_vals, 
   float4** d_y_vals, float4** d_results, size_t* num_bytes, 
   const int num_items );


int gen_kernel_data( float4* x_vals, float4* y_vals, 
   const int num_items, const unsigned int seed );


int gen_expected_data( float4* const x_vals, 
   float4* const y_vals, float4* exp_vals, const int num_items );


int run_kernel( float4* const x_vals, float4* const y_vals, 
   float4* results, float4* const d_x_vals, float4* const d_y_vals,
   float4* d_results, const size_t num_bytes, const int num_items );


int check_kernel( float4* const results, float4* const exp_vals, 
   const int num_items );


int deinit_kernel( float4* x_vals, float4* y_vals, float4* results,
   float4* exp_vals, float4* d_x_vals, float4* d_y_vals, 
   float4* d_results );

#endif
