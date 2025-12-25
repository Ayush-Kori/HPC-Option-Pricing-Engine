cmake_minimum_required(VERSION 3.15)
project(HPC_OptionPricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# 1. Find OpenMP
find_package(OpenMP REQUIRED)

# 2. Manually locate MKL if the automated find_package fails
# On Ubuntu, the headers are usually in /usr/include/mkl
include_directories(/usr/include/mkl)

# 3. Create the executable
add_executable(OptionPricer main.cpp)

# 4. Link the Single Dynamic Library (mkl_rt)
# This is the "SDL" you agreed to in the purple screen earlier!
target_link_libraries(OptionPricer PRIVATE 
    mkl_rt 
    OpenMP::OpenMP_CXX 
    dl
)

# 5. Optimization flags
if(NOT MSVC)
    target_compile_options(OptionPricer PRIVATE -O3 -march=native)
endif()
