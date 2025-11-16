# CMake build configuration for gpt-oss
# This file configures the build system for the Metal backend C/C++ extensions.
# The Metal backend provides GPU-accelerated inference on Apple Silicon (M1/M2/M3).

cmake_minimum_required(VERSION 3.26)
project(gpt_oss LANGUAGES C CXX)

# Auto-detect whether to build the Metal backend based on platform
# Metal is only available on Apple Silicon (ARM64 macOS)
if(NOT DEFINED GPTOSS_BUILD_METAL)
  # Check if we're on Apple Silicon (arm64 architecture)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Apple Silicon detected → enabling GPTOSS_BUILD_METAL")
    set(GPTOSS_BUILD_METAL ON)
  else()
    # On other platforms (Intel Mac, Linux, Windows), Metal is not available
    message(STATUS "Non-Apple Silicon → disabling GPTOSS_BUILD_METAL")
    set(GPTOSS_BUILD_METAL OFF)
  endif()
else()
  # User explicitly set GPTOSS_BUILD_METAL via command line or environment
  message(STATUS "GPTOSS_BUILD_METAL manually set to: ${GPTOSS_BUILD_METAL}")
endif()

# Declare as a cache variable so it can be overridden by users
# Example: cmake -DGPTOSS_BUILD_METAL=OFF .
set(GPTOSS_BUILD_METAL "${GPTOSS_BUILD_METAL}" CACHE BOOL "Enable Metal backend")

# If Metal backend is enabled, configure and build it
if(GPTOSS_BUILD_METAL)
  # Enable Objective-C language support (required for Metal API bindings)
  enable_language(OBJC)

  # Build the Metal backend located in gpt_oss/metal/
  # This includes:
  # - Metal shader compilation (.metal files)
  # - C/C++ source compilation
  # - Python extension module creation
  add_subdirectory(gpt_oss/metal)
endif()
