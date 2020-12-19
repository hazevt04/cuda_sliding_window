#pragma once

#include "my_cufft_utils.hpp"
#include "pinned_mapped_vec_file_io_funcs.hpp"

#include "sliding_window_kernels.cuh"

constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1000.f;
constexpr float AMPLITUDE = 50.f;

   
#include "my_args.hpp"

class SlidingWindowGPU {
public:
   SlidingWindowGPU(){}
   
   SlidingWindowGPU( 
      const int new_num_samples, 
      const int new_window_size,
      const int new_threads_per_block,
      const int new_seed,
      const mode_select_t new_mode_select,
      const std::string new_filename,
      const bool new_debug 
   );
   
   SlidingWindowGPU( 
      const my_args_t my_args ):
         SlidingWindowGPU(
            my_args.num_samples,
            my_args.window_size,
            my_args.threads_per_block,
            my_args.seed,
            my_args.mode_select,
            my_args.filename,
            my_args.debug ) 
   {}

   void run();
   void cpu_run();
   void gen_expected_window_sums() { cpu_run(); };

   void print_results( const std::string& prefix = "Window Sums: " ) {
      print_cufftComplexes( window_sums.data(), num_samples, prefix.c_str(),  " ",  "\n" );
   }

   ~SlidingWindowGPU();

private:
   void initialize_samples();
   void calc_exp_window_sums();

   pinned_mapped_vector<cufftComplex> samples;
   pinned_mapped_vector<cufftComplex> window_sums;

   cufftComplex* d_samples = nullptr;
   cufftComplex* d_window_sums = nullptr;

   mode_select_t mode_select = default_mode_select;
   
   std::string filename = default_filename;
   std::string filepath = "";

   bool debug = false;

   int device_id = -1;
   int num_blocks = default_num_blocks;
   int threads_per_block = default_threads_per_block;
   int num_samples = default_num_samples;
   int adjusted_num_samples = default_adjusted_num_samples;

   int seed = default_seed;
   int window_size = default_window_size;

   size_t num_sample_bytes = default_num_sample_bytes;
   size_t adjusted_num_sample_bytes = default_adjusted_num_sample_bytes;
   std::unique_ptr<cudaStream_t> stream_ptr;

};