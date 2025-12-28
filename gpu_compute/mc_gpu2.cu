#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define OPTIMAL_BATCH 1024   // discovered sweet spot

__global__
void mc_kernel_autotuned(double* global_sum,
                         long long total_paths,
                         double S0,
                         double K,
                         double drift,
                         double vol,
                         unsigned long long seed)
{
    extern __shared__ double sdata[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double local_sum = 0.0;

    // --- STRIDED + BATCHED PATH PROCESSING ---
    for (long long start = tid; start < total_paths; start += total_threads * OPTIMAL_BATCH) {

        #pragma unroll 4
        for (int i = 0; i < OPTIMAL_BATCH; ++i) {
            long long path_id = start + i * total_threads;
            if (path_id >= total_paths) break;

            double Z  = curand_normal_double(&state);
            double ST = S0 * exp(drift + vol * Z);
            local_sum += fmax(ST - K, 0.0);
        }
    }

    // --- BLOCK REDUCTION ---
    sdata[lane] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride)
            sdata[lane] += sdata[lane + stride];
        __syncthreads();
    }

    if (lane == 0)
        atomicAdd(global_sum, sdata[0]);
}

double mc_gpu_price(long long total_paths,
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

    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double vol   = sigma * sqrt(T);

    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mc_kernel_autotuned<<<blocks, threads_per_block,
                           threads_per_block * sizeof(double)>>>(
        d_sum,
        total_paths,
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

    double S0 = 100.0, K = 105.0, T = 1.0;
    double r = 0.05, sigma = 0.2, q = 0.02;

    float gpu_time_ms;

    long long total_paths = 10'000'000;   // 1M / 100M / 1000M

    double price = mc_gpu_price(
        total_paths, S0, K, T, r, sigma, q, gpu_time_ms
    );

    std::cout << "Total Paths          : " << total_paths << "\n";
    std::cout << "GPU Monte Carlo Price: " << price << "\n";
    std::cout << "GPU Kernel Time (ms) : " << gpu_time_ms << "\n";
}
