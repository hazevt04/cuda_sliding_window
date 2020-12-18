#pragma once

#include <string>

typedef enum mode_select_e { Sinusoidal, Random, Filebased } mode_select_t;

const std::string default_filename = "input_samples.dat";
constexpr int default_threads_per_block = 1024;
constexpr int default_seed = 0;
constexpr int default_num_samples = 4000;
constexpr int default_window_size = 48;
constexpr int default_num_sample_bytes = default_num_samples * sizeof(cufftComplex);

constexpr int default_adjusted_num_samples = 4096;
constexpr int default_adjusted_num_sample_bytes = default_adjusted_num_samples * sizeof(cufftComplex);
constexpr int default_num_blocks = (default_adjusted_num_samples + (default_threads_per_block-1))/default_threads_per_block;

const mode_select_t default_mode_select = mode_select_t::Sinusoidal;

mode_select_t decode_mode_select_string( std::string mode_select_string ) {
   if ( mode_select_string == "Sinusoidal" ) {
      return mode_select_t::Sinusoidal;
   } else if ( mode_select_string == "Random" ) {
      return mode_select_t::Random;
   } else if ( mode_select_string == "Filebased" ) {
      return mode_select_t::Filebased;
   } else {
      std::cout << "WARNING: Invalid mode select string: " << mode_select_string << "\n";
      std::cout << "Selecting mode_select_t::Sinusoidal\n"; 
      return mode_select_t::Sinusoidal;
   }
}

typedef struct my_args_s {
   mode_select_t mode_select = default_mode_select;
   std::string filename = default_filename;
   int threads_per_block = default_threads_per_block;
   int num_samples = default_num_samples;
   int window_size = default_window_size;
   int seed = default_seed;
   bool debug = false;
   bool help_shown =false;

} my_args_t;
