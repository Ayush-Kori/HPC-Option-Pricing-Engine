// mc_gpu.cu
// ============================================================
// GPU Monte Carlo Pricing (FP64) with explicit GPU verification
// ============================================================

#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thrust for reduction
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

// ------------------------------------------------------------
// CUDA KERNEL: one thread = one Monte Carlo path
// ------------------------------------------------------------
__global__
void mc_kernel(double* payoffs,
               int N,
               double S0,
               double K,
               double drift,
               double vol,
               unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double Z  = curand_normal_double(&state);
    double ST = S0 * exp(drift + vol * Z);

    payoffs[tid] = fmax(ST - K, 0.0);
}

// ------------------------------------------------------------
// GPU MONTE CARLO DRIVER
// ------------------------------------------------------------
double mc_gpu_price(int N,
                    double S0,
                    double K,
                    double T,
                    double r,
                    double sigma,
                    double q,
                    float& gpu_time_ms)
{
    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double vol   = sigma * sqrt(T);

    double* d_payoffs;
    cudaMalloc(&d_payoffs, N * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mc_kernel<<<blocks, threadsPerBlock>>>(
        d_payoffs, N, S0, K, drift, vol, 777ULL
    );

    thrust::device_ptr<double> ptr(d_payoffs);
    double sum_payoff = thrust::reduce(ptr, ptr + N, 0.0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_payoffs);

    return exp(-r * T) * sum_payoff / N;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    // ---------------- GPU VERIFICATION ----------------
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Using GPU              : " << prop.name << "\n";
    std::cout << "Compute Capability     : "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "Global Memory (MB)     : "
              << prop.totalGlobalMem / (1024 * 1024) << "\n\n";

    // ---------------- PROBLEM SETUP ----------------
    int    N     = 100'000'000;   // 100 million paths
    double S0    = 100.0;
    double K     = 105.0;
    double T     = 1.0;
    double r     = 0.05;
    double sigma = 0.2;
    double q     = 0.02;

    float gpu_time_ms = 0.0f;

    double price = mc_gpu_price(
        N, S0, K, T, r, sigma, q, gpu_time_ms
    );

    std::cout << "GPU Monte Carlo Price  : " << price << "\n";
    std::cout << "GPU Compute Time (ms)  : " << gpu_time_ms << "\n";

    return 0;
}

