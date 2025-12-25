#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random> // For Naive
#include <mkl.h>  // For HPC
#include <omp.h>

// --- NAIVE MONTE CARLO ---
// Uses standard C++ <random> which is scalar and harder to parallelize efficiently
double price_monte_carlo_naive(int n_sims, double S0, double K, double T, double r, double sigma) {
    std::default_random_engine generator(777);
    std::normal_distribution<double> distribution(0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);
    double payoff_sum = 0.0;

    for (int i = 0; i < n_sims; i++) {
        double Z = distribution(generator);
        double ST = S0 * std::exp(drift + vol_sqrt_T * Z);
        payoff_sum += std::max(ST - K, 0.0);
    }

    return (payoff_sum / n_sims) * std::exp(-r * T);
}

// --- HPC MONTE CARLO ---
// Uses Intel MKL VSL (Vector Statistics Library) and OpenMP
double price_monte_carlo_hpc(int n_sims, double S0, double K, double T, double r, double sigma) {
    // 1. Pre-allocate buffer for random numbers (SIMD friendly)
    std::vector<double> Z(n_sims);
    
    // 2. Vectorized Random Number Generation (MKL)
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 777);
    // Generates all random numbers in one high-speed batch
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_sims, Z.data(), 0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);
    double payoff_sum = 0.0;

    // 3. Parallel Payoff Calculation (OpenMP)
    #pragma omp parallel for reduction(+:payoff_sum)
    for (int i = 0; i < n_sims; i++) {
        double ST = S0 * std::exp(drift + vol_sqrt_T * Z[i]);
        payoff_sum += std::max(ST - K, 0.0);
    }

    vslDeleteStream(&stream);
    return (payoff_sum / n_sims) * std::exp(-r * T);
}

int main() {
    const int N = 10000000; // 10 Million Paths
    double S0 = 100.0, K = 105.0, T = 1.0, r = 0.05, sigma = 0.2;

    std::cout << "Starting Simulation for " << N << " paths..." << std::endl;

    // Benchmark Naive
    auto s1 = std::chrono::high_resolution_clock::now();
    double p1 = price_monte_carlo_naive(N, S0, K, T, r, sigma);
    auto e1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t1 = e1 - s1;

    // Benchmark HPC
    auto s2 = std::chrono::high_resolution_clock::now();
    double p2 = price_monte_carlo_hpc(N, S0, K, T, r, sigma);
    auto e2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t2 = e2 - s2;

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Naive Price: " << p1 << " | Time: " << t1.count() << " ms" << std::endl;
    std::cout << "HPC Price:   " << p2 << " | Time: " << t2.count() << " ms" << std::endl;
    std::cout << "Speedup:     " << t1.count() / t2.count() << "x" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return 0;
}