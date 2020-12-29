
#include "SlidingWindowGPU.cuh"

#include "sliding_window_kernels.cuh"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_utils.hpp"

SlidingWindowGPU::SlidingWindowGPU( 
   const int new_num_samples, 
   const int new_window_size,
   const int new_threads_per_block,
   const int new_seed,
   const mode_select_t new_mode_select,
   const std::string new_filename,
   const bool new_debug ):
      num_samples( new_num_samples ),
      window_size( new_window_size ),
      threads_per_block( new_threads_per_block ),
      seed( new_seed ),
      mode_select( new_mode_select ),
      filename( new_filename ),
      debug( new_debug ) {

   try {
      cudaError_t cerror = cudaSuccess;         
      
      dout << __func__ << "(): Mode Select is " << get_mode_select_string( mode_select ) << "\n";

      if ( mode_select == mode_select_t::Increasing ) {
         if ( num_samples > MAX_NUM_SAMPLES_INCREASING ) {
            std::cout << "WARNING: num_samples, " << num_samples << " too large. The sum will not fit in a 32-bit integer.\n";
            std::cout << "Changing num_samples to the max: " << MAX_NUM_SAMPLES_INCREASING << "\n";
            num_samples = MAX_NUM_SAMPLES_INCREASING;
         }
      }

      try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );

      stream_ptr = my_make_unique<cudaStream_t>();
      try_cudaStreamCreate( stream_ptr.get() );
      dout << __func__ << "(): after cudaStreamCreate()\n"; 

      dout << __func__ << "(): num_samples is " << num_samples << "\n";

      num_blocks = (num_samples + (threads_per_block-1))/threads_per_block;

      dout << __func__ << "(): num_blocks is " << num_blocks << "\n";

      adjusted_num_samples = threads_per_block * num_blocks;
      adjusted_num_sample_bytes = adjusted_num_samples * sizeof( cufftComplex );

      num_windowed_samples = num_samples - window_size;
      dout << __func__ << "(): number of windowed samples is "
         << num_windowed_samples << "\n";

      dout << __func__ << "(): adjusted number of samples for allocation is " 
         << adjusted_num_samples << "\n";
      dout << __func__ << "(): adjusted number of sample bytes for cudaMemcpyAsync is "
         << adjusted_num_sample_bytes << "\n";

      char* user_env = getenv( "USER" );
      if ( user_env == nullptr ) {
         throw std::runtime_error( std::string{__func__} + 
            "(): Empty USER env. USER environment variable needed for paths to files" ); 
      }
      
      std::string filepath_prefix = "/home/" + std::string{user_env} + "/Sandbox/CUDA/cuda_sliding_window/";

      filepath = filepath_prefix + filename;

      dout << __func__ << "(): Filepath is " << filepath << "\n";

      window_sums.reserve( adjusted_num_samples );
      window_sums.resize( adjusted_num_samples );
      std::fill( window_sums.begin(), window_sums.end(), make_cuFloatComplex( 0.f, 0.f ) );
      
      d_window_sums.reserve( adjusted_num_samples );

      dout << __func__ << "() after d_window_sums.reserve()\n";      

      samples.reserve( adjusted_num_samples );
      dout << __func__ << "() after samples.reserve()\n";
      samples.resize( adjusted_num_samples );

      d_samples.reserve( adjusted_num_samples );
      dout << __func__ << "() after d_samples.reserve()\n";      
      
      d_samples.reserve( adjusted_num_samples );
      dout << __func__ << "() after d_samples.reserve()\n";
      //d_samples.resize( adjusted_num_samples );
      //dout << __func__ << "() after d_samples.resize()\n";

      exp_window_sums = new cufftComplex[num_samples];
      for( int index = 0; index < num_samples; ++index ) {
         exp_window_sums[index] = make_cuFloatComplex(0.f,0.f);
      }


   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }      
} // end of SlidingWindowGPU()


void SlidingWindowGPU::initialize_samples() {
   try {
      std::fill( samples.begin(), samples.end(), make_cuFloatComplex(0.f,0.f) );

      if( mode_select == mode_select_t::Sinusoidal ) {
         dout << __func__ << "(): Sinusoidal Sample Mode Selected\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            float t_val_real = AMPLITUDE*sin(2*PI*FREQ*index);
            float t_val_imag = AMPLITUDE*cos(2*PI*FREQ*index);
            samples[index] = make_cuFloatComplex( t_val_real, t_val_imag );
         }
      } else if ( mode_select == mode_select_t::Increasing ) {
         dout << __func__ << "(): Increasing Sample Mode Selected.\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            samples[index] = make_cuFloatComplex( (float)(index+1), (float)(index+1) );
         }
      } else if ( mode_select == mode_select_t::Random ) {
         dout << __func__ << "(): Random Sample Test Selected. Seed is " << seed << "\n";
         gen_cufftComplexes( samples.data(), num_samples, -AMPLITUDE, AMPLITUDE, seed );

      } else if ( mode_select == mode_select_t::Filebased ) {
         dout << __func__ << "(): File-Based Sample Test Selected. File is " << filename << "\n";
         read_binary_file<cufftComplex>( 
            samples,
            filepath.c_str(),
            num_samples, 
            debug 
         );
      } // end of else-ifs for mode_select

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   } // end of try      
} // end of void initialize_samples()


void SlidingWindowGPU::gen_expected_window_sums() { 
   cpu_run(); 
}


void SlidingWindowGPU::print_results( const std::string& prefix = "Window Sums: " ) {
   print_cufftComplexes( window_sums.data(), num_samples, prefix.c_str(),  " ",  "\n" );
}


void SlidingWindowGPU::calc_exp_window_sums() {

   // exp_window_sums must already be all zeros
   dout << __func__ << "(): exp_window_sums[0] = { " 
      << exp_window_sums[0].x << ", " << exp_window_sums[0].y << " }\n"; 

   for( int index = 0; index < window_size; ++index ) {
      exp_window_sums[0] = cuCaddf( exp_window_sums[0], samples[index] );
   }

   dout << __func__ << "(): after initial summation, exp_window_sums[0] = { " 
      << exp_window_sums[0].x << ", " << exp_window_sums[0].y << " }\n"; 
      
   dout << __func__ << "(): num_windowed_samples is " << num_windowed_samples << "\n"; 
   for( int index = 1; index < num_windowed_samples; ++index ) {
      cufftComplex temp = cuCsubf( exp_window_sums[index-1], samples[index-1] );
      exp_window_sums[index] = cuCaddf( temp, samples[index + window_size-1] );
   } 

} // end of calc_exp_window_sums()


void SlidingWindowGPU::cpu_run() {
   try { 
      float cpu_milliseconds = 0.f;
      
      dout << __func__ << "(): num_samples is " << num_samples << "\n";
      
      Time_Point start = Steady_Clock::now();

      calc_exp_window_sums();
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      cpu_milliseconds = duration_ms.count();
      
      float samples_per_second = (num_samples*1000.f)/cpu_milliseconds;
      std::cout << "It took the CPU " << cpu_milliseconds << " milliseconds to process " << num_samples << " samples\n";
      std::cout << "That's a rate of " << samples_per_second/1e6 << " samples processed per second\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
}


void SlidingWindowGPU::check_results( const std::string& prefix = "Original" ) {
   try {
      float max_diff = 1;
      bool all_close = false;
      if ( debug ) {
         print_results( "Window Sums: " );
         std::cout << "\n"; 
      }
      dout << __func__ << "(): " << prefix << "Window Sums Check:\n"; 
      all_close = cufftComplexes_are_close( window_sums.data(), exp_window_sums, num_samples, max_diff, "window sums: ", debug );
      if (!all_close) {
         throw std::runtime_error{ std::string{__func__} + 
            std::string{"(): "} + prefix + 
            std::string{"Mismatch between actual window_sums from GPU and expected window_sums."} };
      }
      std::cout << prefix << "All " << num_samples << " Window Sums matched expected Window Sums. Test Passed.\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }

}

void SlidingWindowGPU::run_warmup() {
   try {
      cudaError_t cerror = cudaSuccess;
      int num_shared_bytes = 0;
      
      std::fill( window_sums.begin(), window_sums.end(), make_cuFloatComplex( 0.f, 0.f ) );

      try_cuda_func_throw( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(),
         adjusted_num_sample_bytes, cudaMemcpyHostToDevice ) );

      sliding_window_original<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_window_sums.data(), 
         d_samples.data(),
         window_size,
         num_windowed_samples 
      );

      try_cuda_func_throw( cerror, cudaMemcpyAsync( window_sums.data(), d_window_sums.data(),
         adjusted_num_sample_bytes, cudaMemcpyDeviceToHost ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
}


void SlidingWindowGPU::run_original( const std::string& prefix = "Original: " ) {
   try {
      cudaError_t cerror = cudaSuccess;
      float gpu_milliseconds = 0.f;
      int num_shared_bytes = 0;

      std::fill( window_sums.begin(), window_sums.end(), make_cuFloatComplex( 0.f, 0.f ) );
      Time_Point start = Steady_Clock::now();
      
      try_cuda_func_throw( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(),
         adjusted_num_sample_bytes, cudaMemcpyHostToDevice ) );
      
      sliding_window_original<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_window_sums.data(), 
         d_samples.data(),
         window_size,
         num_windowed_samples 
      );

      try_cuda_func_throw( cerror, cudaMemcpyAsync( window_sums.data(), d_window_sums.data(),
         adjusted_num_sample_bytes, cudaMemcpyDeviceToHost ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << prefix << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";
      std::cout << prefix << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
} // end of void SlidingWindowGPU::run_original( const std::string& prefix = "Original: " ) {


void SlidingWindowGPU::run_unrolled_2x( const std::string& prefix = "Unrolled 2x: " ) {
   try {
      cudaError_t cerror = cudaSuccess;
      float gpu_milliseconds = 0.f;
      int num_shared_bytes = 0;

      // adjusting number of blocks and memory bound for using half as many threads
      num_blocks = ((adjusted_num_samples/2) + ( threads_per_block - 1 ))/threads_per_block;

      std::fill( window_sums.begin(), window_sums.end(), make_cuFloatComplex( 0.f, 0.f ) );
      Time_Point start = Steady_Clock::now();
      
      try_cuda_func_throw( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(),
         adjusted_num_sample_bytes, cudaMemcpyHostToDevice ) );
      
      sliding_window_unrolled_2x<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_window_sums.data(), 
         d_samples.data(),
         window_size,
         num_windowed_samples 
      );

      try_cuda_func_throw( cerror, cudaMemcpyAsync( window_sums.data(), d_window_sums.data(),
         adjusted_num_sample_bytes, cudaMemcpyDeviceToHost ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << prefix << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";
      std::cout << prefix << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
} // end of void SlidingWindowGPU::run_unrolled_2x( const std::string& prefix = "Original: " ) {


void SlidingWindowGPU::run_unrolled_4x( const std::string& prefix = "Unrolled 4x: " ) {
   try {
      cudaError_t cerror = cudaSuccess;
      float gpu_milliseconds = 0.f;
      int num_shared_bytes = 0;

      // adjusting number of blocks and memory bound for using half as many threads
      num_blocks = ((adjusted_num_samples/4) + ( threads_per_block - 1 ))/threads_per_block;

      std::fill( window_sums.begin(), window_sums.end(), make_cuFloatComplex( 0.f, 0.f ) );
      Time_Point start = Steady_Clock::now();
      
      try_cuda_func_throw( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(),
         adjusted_num_sample_bytes, cudaMemcpyHostToDevice ) );
      
      sliding_window_unrolled_4x<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_window_sums.data(), 
         d_samples.data(),
         window_size,
         num_windowed_samples 
      );

      try_cuda_func_throw( cerror, cudaMemcpyAsync( window_sums.data(), d_window_sums.data(),
         adjusted_num_sample_bytes, cudaMemcpyDeviceToHost ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << prefix << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";
      std::cout << prefix << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
} // end of void SlidingWindowGPU::run_unrolled_4x( const std::string& prefix = "Unrolled 4x: " )


void SlidingWindowGPU::run_unrolled_8x( const std::string& prefix = "Unrolled 8x: " ) {
   try {
      cudaError_t cerror = cudaSuccess;
      float gpu_milliseconds = 0.f;
      int num_shared_bytes = 0;

      // adjusting number of blocks and memory bound for using half as many threads
      num_blocks = ((adjusted_num_samples/8) + ( threads_per_block - 1 ))/threads_per_block;

      std::fill( window_sums.begin(), window_sums.end(), make_cuFloatComplex( 0.f, 0.f ) );
      Time_Point start = Steady_Clock::now();
      
      try_cuda_func_throw( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(),
         adjusted_num_sample_bytes, cudaMemcpyHostToDevice ) );
      
      sliding_window_unrolled_8x<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_window_sums.data(), 
         d_samples.data(),
         window_size,
         num_windowed_samples 
      );

      try_cuda_func_throw( cerror, cudaMemcpyAsync( window_sums.data(), d_window_sums.data(),
         adjusted_num_sample_bytes, cudaMemcpyDeviceToHost ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << prefix << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";
      std::cout << prefix << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
} // end of 


void SlidingWindowGPU::run() {
   try {
      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n\n"; 
      
      dout << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      dout << __func__ << "(): adjusted_num_sample_bytes is " << adjusted_num_sample_bytes << "\n"; 

      initialize_samples();
      gen_expected_window_sums();

      if ( debug ) {
         print_cufftComplexes( samples.data(), num_samples, "Samples: ", " ", "\n" ); 
         print_cufftComplexes( exp_window_sums, num_samples, "Expected Window Sums: ", " ", "\n" ); 
      }

      run_warmup();

      run_original( "Original: " );
      check_results( "Original: " );
      
      run_unrolled_2x( "Unrolled 2x: " );
      check_results( "Unrolled 2x: " );

      run_unrolled_4x( "Unrolled 4x: " );
      check_results( "Unrolled 4x: " );

      run_unrolled_8x( "Unrolled 8x: " );
      check_results( "Unrolled 8x: " );

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }
}

SlidingWindowGPU::~SlidingWindowGPU() {
   dout << __func__ << "() called\n";
   samples.clear();    
   window_sums.clear();
   
   d_samples.clear();    
   d_window_sums.clear();

   if ( exp_window_sums ) delete [] exp_window_sums;
   
   if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
   dout << __func__ << "() done\n";
}


