#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <mkl.h>
#include <omp.h>

// --- BLACK-SCHOLES (ANALYTICAL) ---
double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

void bs_naive(int n, double* S, double K, double T, double r, double sigma, double* res) {
    for (int i = 0; i < n; i++) {
        double d1 = (std::log(S[i] / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        res[i] = S[i] * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    }
}

void bs_hpc(int n, double* S, double K, double T, double r, double sigma, double* res, double* d1_b, double* d2_b, double* c1_b, double* c2_b) {
    vmlSetMode(VML_EP);
    double v_sqrt_T = sigma * std::sqrt(T);
    double df = std::exp(-r * T);
    double mu = (r + 0.5 * sigma * sigma) * T;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        d1_b[i] = (std::log(S[i] / K) + mu) / v_sqrt_T;
        d2_b[i] = d1_b[i] - v_sqrt_T;
    }
    vdCdfNorm(n, d1_b, c1_b);
    vdCdfNorm(n, d2_b, c2_b);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res[i] = S[i] * c1_b[i] - K * df * c2_b[i];
    }
}

// --- MONTE CARLO (NUMERICAL) ---
double mc_naive(int n, double S0, double K, double T, double r, double sigma) {
    std::default_random_engine gen(777);
    std::normal_distribution<double> dist(0.0, 1.0);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_T = sigma * std::sqrt(T);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol_T * dist(gen));
        sum += std::max(ST - K, 0.0);
    }
    return (sum / n) * std::exp(-r * T);
}

double mc_hpc(int n, double S0, double K, double T, double r, double sigma) {
    std::vector<double> Z(n);
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 777);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, Z.data(), 0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_T = sigma * std::sqrt(T);
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol_T * Z[i]);
        sum += std::max(ST - K, 0.0);
    }
    vslDeleteStream(&stream);
    return (sum / n) * std::exp(-r * T);
}

int main() {
    const int N = 10000000;
    double S0 = 100.0, K = 105.0, T = 1.0, r = 0.05, sigma = 0.2;
    std::vector<double> S_vec(N, S0), res_n(N), res_h(N);
    std::vector<double> d1(N), d2(N), c1(N), c2(N);

    std::cout << "Benchmarking 10 Million Paths...\n";

    // --- BLACK SCHOLES ---
    auto s1 = std::chrono::high_resolution_clock::now();
    bs_naive(N, S_vec.data(), K, T, r, sigma, res_n.data());
    auto e1 = std::chrono::high_resolution_clock::now();
    
    auto s2 = std::chrono::high_resolution_clock::now();
    bs_hpc(N, S_vec.data(), K, T, r, sigma, res_h.data(), d1.data(), d2.data(), c1.data(), c2.data());
    auto e2 = std::chrono::high_resolution_clock::now();

    // --- MONTE CARLO ---
    auto s3 = std::chrono::high_resolution_clock::now();
    double mc_p_n = mc_naive(N, S0, K, T, r, sigma);
    auto e3 = std::chrono::high_resolution_clock::now();

    auto s4 = std::chrono::high_resolution_clock::now();
    double mc_p_h = mc_hpc(N, S0, K, T, r, sigma);
    auto e4 = std::chrono::high_resolution_clock::now();

    std::cout << "\n[RESULT COMPARISON]\n";
    std::cout << "BS Analytical Price: " << res_h[0] << "\n";
    std::cout << "MC Numerical Price:  " << mc_p_h << "\n";

    std::cout << "\n[PERFORMANCE COMPARISON]\n";
    printf("BS Naive: %10.2f ms\n", std::chrono::duration<double, std::milli>(e1 - s1).count());
    printf("BS HPC:   %10.2f ms\n", std::chrono::duration<double, std::milli>(e2 - s2).count());
    printf("MC Naive: %10.2f ms\n", std::chrono::duration<double, std::milli>(e3 - s3).count());
    printf("MC HPC:   %10.2f ms\n", std::chrono::duration<double, std::milli>(e4 - s4).count());

    return 0;
}