#pragma once

#include <exception>
#include <stdexcept>

#include "managed_allocator_host.hpp"

#include "my_file_io_funcs.hpp"

template<typename T>
void write_binary_file(
   managed_vector_host<T>& vals, const char* filename, const bool debug = false) {
      
   try {
      write_binary_file_inner(vals.data(), filename, (int)vals.size(), debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
   
}

template<typename T>
void write_binary_file(
   managed_vector_host<T>& vals, const char* filename, const int num_vals, const bool debug = false) {
      
   try {
      write_binary_file_inner(vals.data(), filename, num_vals, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
   
}


template<typename T>
void read_binary_file(
   managed_vector_host<T>& vals, const char* filename, const bool debug = false) {

   try {
      read_binary_file_inner<T>( vals.data(), filename, (int)vals.size(), debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}

template<typename T>
void read_binary_file(
  managed_vector_host<T>& vals, const char* filename, const int num_vals, const bool debug = false) {

  try {
     read_binary_file_inner<T>( vals.data(), filename, num_vals, debug );
  } catch (std::exception& ex) {
     throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
  }
}


