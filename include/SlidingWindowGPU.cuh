#pragma once

#include "pinned_mapped_allocator.hpp"
#include "pinned_mapped_vec_file_io_funcs.hpp"

#include "my_args.hpp"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_utils.hpp"


constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1000.f;
constexpr float AMPLITUDE = 50.f;

// For increasing mode, the sample values go from {1.0,1.0} to {N-1,N-1}
// window_sum is a 32-bit int. Therefore the max window_sum value is 
// {2147483647.0, 2147483647.0}
// The sum of the first N integers is (N * (N-1))/2
// N such that (N * (N-1))/2 < 2147483647 is 65535
// The sum of the integers from 1 to 65535 is 2147450880.
// Therefore the max value for num_samples in Increasing mode is 65535
constexpr int MAX_NUM_SAMPLES_INCREASING = 65535;

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

   void check_results( const std::string& prefix );

   void run();
   void cpu_run();
   void gen_expected_window_sums();

   void print_results( const std::string& prefix );
   
   ~SlidingWindowGPU();

private:
   void initialize_samples();
   void calc_exp_window_sums();
   void clear_results( const std::string& prefix );
   void run_warmup();
   void run_original( const std::string& prefix );
   void run_unrolled_2x( const std::string& prefix );
   void run_unrolled_4x( const std::string& prefix );
   void run_unrolled_8x( const std::string& prefix );


   pinned_mapped_vector<cufftComplex> samples;
   pinned_mapped_vector<cufftComplex> window_sums;

   cufftComplex* d_samples = nullptr;
   cufftComplex* d_window_sums = nullptr;

   cufftComplex* exp_window_sums = nullptr;

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

   int num_windowed_samples = default_num_windowed_samples;

   size_t num_sample_bytes = default_num_sample_bytes;
   size_t adjusted_num_sample_bytes = default_adjusted_num_sample_bytes;
   std::unique_ptr<cudaStream_t> stream_ptr;

};
