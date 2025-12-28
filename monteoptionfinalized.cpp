#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <mkl.h>
#include <omp.h>
#include <random>

struct Results {
    double price, delta, gamma, vega, theta, rho;
};

// --- HELPERS ---
inline double normal_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}
double cdf_naive(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

// --- 1. BSM NAIVE ---
void bsm_naive(int n, double* S, double K, double T, double r, double sigma, double q, Results* res) {
    for (int i = 0; i < n; i++) {
        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S[i] / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;
        double exp_rt = std::exp(-r * T);
        double exp_qt = std::exp(-q * T);
        res[i].price = S[i] * exp_qt * cdf_naive(d1) - K * exp_rt * cdf_naive(d2);
        res[i].delta = exp_qt * cdf_naive(d1);
    }
}

// --- 2. BSM HPC (MKL + Chunking) ---
void bsm_hpc(int n, double* S, double K, double T, double r, double sigma, double q, Results* res) {
    const int CHUNK = 1024;
    #pragma omp parallel
    {
        std::vector<double> d1_b(CHUNK), d2_b(CHUNK), c1_b(CHUNK), c2_b(CHUNK);
        #pragma omp for
        for (int i = 0; i < n; i += CHUNK) {
            int len = std::min(CHUNK, n - i);
            double sqrtT = std::sqrt(T), v_sqrtT = sigma * sqrtT;
            double exp_rt = std::exp(-r * T), exp_qt = std::exp(-q * T);
            for (int j = 0; j < len; j++) {
                d1_b[j] = (std::log(S[i+j] / K) + (r - q + 0.5 * sigma * sigma) * T) / v_sqrtT;
                d2_b[j] = d1_b[j] - v_sqrtT;
            }
            vdCdfNorm(len, d1_b.data(), c1_b.data());
            vdCdfNorm(len, d2_b.data(), c2_b.data());
            for (int j = 0; j < len; j++) {
                int idx = i + j;
                res[idx].price = S[idx] * exp_qt * c1_b[j] - K * exp_rt * c2_b[j];
                res[idx].delta = exp_qt * c1_b[j];
                res[idx].gamma = (exp_qt * normal_pdf(d1_b[j])) / (S[idx] * v_sqrtT);
            }
        }
    }
}

// --- 3. MC NAIVE ---
double mc_naive(int n, double S0, double K, double T, double r, double sigma, double q) {
    std::mt19937 gen(777);
    std::normal_distribution<double> dist(0.0, 1.0);
    double sum = 0.0, drift = (r - q - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol * dist(gen));
        sum += std::max(ST - K, 0.0);
    }
    return (sum / n) * std::exp(-r * T);
}

// --- 4. MC HPC ---
double mc_hpc(int n, double S0, double K, double T, double r, double sigma, double q) {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 777);
    const int CHUNK = 8192;
    double total_payoff = 0.0, drift = (r - q - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    #pragma omp parallel reduction(+:total_payoff)
    {
        std::vector<double> z(CHUNK);
        #pragma omp for
        for (int i = 0; i < n; i += CHUNK) {
            int len = std::min(CHUNK, n - i);
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, len, z.data(), 0.0, 1.0);
            for (int j = 0; j < len; j++) {
                total_payoff += std::max(S0 * std::exp(drift + vol * z[j]) - K, 0.0);
            }
        }
    }
    vslDeleteStream(&stream);
    return (total_payoff / n) * std::exp(-r * T);
}

int main() {
    const int N = 10000000; // 10 Million
    double S0 = 100.0, K = 105.0, T = 1.0, r = 0.05, sigma = 0.2, q = 0.02;
    std::vector<double> S_vec(N, S0);
    std::vector<Results> res_naive(N), res_hpc(N);

    auto timer = [](auto func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Benchmarking 10M Options...\n\n";

    double t1 = timer([&](){ bsm_naive(N, S_vec.data(), K, T, r, sigma, q, res_naive.data()); });
    double t2 = timer([&](){ bsm_hpc(N, S_vec.data(), K, T, r, sigma, q, res_hpc.data()); });
    
    double mc_p_n, mc_p_h;
    double t3 = timer([&](){ mc_p_n = mc_naive(N, S0, K, T, r, sigma, q); });
    double t4 = timer([&](){ mc_p_h = mc_hpc(N, S0, K, T, r, sigma, q); });

    // Accuracy Calculation
    double ground_truth = res_hpc[0].price;
    double mc_error = std::abs(mc_p_h - ground_truth);

    std::cout << "--- RESULTS ---\n";
    std::cout << "BSM Price (Analytical): " << ground_truth << "\n";
    std::cout << "MC Price (Numerical):  " << mc_p_h << "\n";
    std::cout << "MC Absolute Error:     " << mc_error << "\n\n";

    std::cout << "--- PERFORMANCE ---\n";
    printf("BSM Naive: %10.2f ms\n", t1);
    printf("BSM HPC:   %10.2f ms (Speedup: %.1fx)\n", t2, t1/t2);
    printf("MC Naive:  %10.2f ms\n", t3);
    printf("MC HPC:    %10.2f ms (Speedup: %.1fx)\n", t4, t3/t4);

    return 0;
}