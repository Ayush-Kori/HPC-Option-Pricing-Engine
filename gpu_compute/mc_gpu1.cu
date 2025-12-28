#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__
void mc_kernel_accumulate(double* global_sum,
                          int paths_per_thread,
                          double S0,
                          double K,
                          double drift,
                          double vol,
                          unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double local_sum = 0.0;

    for (int i = 0; i < paths_per_thread; ++i) {
        double Z  = curand_normal_double(&state);
        double ST = S0 * exp(drift + vol * Z);
        local_sum += fmax(ST - K, 0.0);
    }

    // ONE atomic add per thread
    atomicAdd(global_sum, local_sum);
}

double mc_gpu_price(int total_paths,
                    double S0,
                    double K,
                    double T,
                    double r,
                    double sigma,
                    double q,
                    float& gpu_time_ms)
{
    const int threads_per_block = 256;
    const int blocks = 120;
    const int total_threads = threads_per_block * blocks;
    const int paths_per_thread = total_paths / total_threads;

    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double vol   = sigma * sqrt(T);

    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mc_kernel_accumulate<<<blocks, threads_per_block>>>(
        d_sum,
        paths_per_thread,
        S0, K,
        drift, vol,
        777ULL
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    double h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);

    return exp(-r * T) * h_sum / total_paths;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Using GPU          : " << prop.name << "\n";
    std::cout << "Compute Capability : "
              << prop.major << "." << prop.minor << "\n\n";

    int    N     = 100'000'000;
    double S0    = 100.0;
    double K     = 105.0;
    double T     = 1.0;
    double r     = 0.05;
    double sigma = 0.2;
    double q     = 0.02;

    float gpu_time_ms;

    double price = mc_gpu_price(
        N, S0, K, T, r, sigma, q, gpu_time_ms
    );

    std::cout << "GPU Monte Carlo Price : " << price << "\n";
    std::cout << "GPU Kernel Time (ms)  : " << gpu_time_ms << "\n";
}

