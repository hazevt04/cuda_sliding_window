// C++ File for main

#include "SlidingWindowGPU.cuh"

#include "parse_args.hpp"

int main(int argc, char **argv) {
   try {
      my_args_t my_args;
      parse_args( my_args, argc, argv ); 

      if ( my_args.help_shown ) {
         return EXIT_SUCCESS;
      }

      SlidingWindowGPU sliding_window_gpu{ my_args };

      sliding_window_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
