#pragma once

#include <string>

#include <cufft.h>

typedef enum mode_select_e { Sinusoidal, Random, Filebased } mode_select_t;

const std::string default_filename = "input_samples.dat";
constexpr int default_threads_per_block = 96;
constexpr int default_seed = 0;
constexpr int default_num_samples = 4000;
constexpr int default_window_size = 48;
constexpr int default_num_windowed_samples = default_num_samples - default_window_size;
constexpr int default_num_sample_bytes = default_num_samples * sizeof(cufftComplex);
constexpr size_t default_num_shared_bytes = default_window_size * sizeof(cufftComplex);

constexpr int default_adjusted_num_samples = 4096;
constexpr int default_adjusted_num_sample_bytes = default_adjusted_num_samples * sizeof(cufftComplex);
constexpr int default_num_blocks = (default_adjusted_num_samples + (default_threads_per_block-1))/default_threads_per_block;

const mode_select_t default_mode_select = mode_select_t::Sinusoidal;

mode_select_t decode_mode_select_string( std::string mode_select_string ); 

std::string get_mode_select_string( mode_select_t mode_select );

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
