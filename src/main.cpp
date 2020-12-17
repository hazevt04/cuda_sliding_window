// C++ File for main

#include "add.cuh"

// just to compile it. it's not actually used
#include "my_cufft_utils.hpp"

// Empirically-determined maximum value for num_items. 
// 1<<29 exits with the vague 'killed' message. 
// 1<<30 causes a cuda error.
// Assuming floats, this makes each array take 1 GiB
#define MAX_NUM_ITEMS (1<<28)

int main(int argc, char **argv) {

  float4* x_vals = NULL;
  float4* y_vals = NULL;
  float4* results = NULL;
  float4* exp_vals = NULL;
  float4* d_x_vals = NULL;
  float4* d_y_vals = NULL;
  float4* d_results = NULL;

  // Empirically-determined maximum number
  int num_items = 1<<21;
  size_t num_bytes = num_items * sizeof(float4);
  
  // Initialize the memory
  int status = SUCCESS;
    
  printf("Initializing memory to run CUDA kernel\n");

  try_func( status, "Init Kernel Failed", 
      init_kernel( &x_vals, &y_vals, &results, &exp_vals, 
      &d_x_vals, &d_y_vals, &d_results, &num_bytes, num_items ) );

  printf( "Trying to generate data for the kernel to run\n" );
  
  try_func( status, "Generate Kernel Data Failed", 
    gen_kernel_data( x_vals, y_vals, num_items, 5 ) );

  try_func( status, "Generate Expected Data Failed",
    gen_expected_data( x_vals, y_vals, exp_vals, num_items ) );

  printf( "Trying to run CUDA kernel\n" );

  try_func( status, "Run Add Kernel Failed", 
    run_kernel( x_vals, y_vals, results, d_x_vals, d_y_vals, d_results, 
      num_bytes, num_items ) );
  
  try_func( status, "Check Kernel Failed.", 
      check_kernel( results, exp_vals, num_items ) );
  
   
  try_func( status, "Deinit Kernel Failed.", 
      deinit_kernel( x_vals, y_vals, results, exp_vals, 
      d_x_vals, d_y_vals, d_results ) );

  return 0;
}
// end of C++ file for main
