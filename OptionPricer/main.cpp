/*
================================================================================
BLACK–SCHOLES–MERTON OPTION PRICING ENGINE
================================================================================

This code prices a European Call Option under the Black–Scholes–Merton (BSM) model
using both:
  1) Closed-form analytical formulas
  2) Monte Carlo simulation under the risk-neutral measure

It also computes analytical Greeks (Delta, Gamma) and validates Monte Carlo
pricing against the analytical solution.

--------------------------------------------------------------------------------
FINANCIAL MODEL
--------------------------------------------------------------------------------

We assume the underlying asset price S_t follows Geometric Brownian Motion (GBM):

    dS_t = (r - q) S_t dt + σ S_t dW_t

Where:
    S_t : asset price at time t
    r   : risk-free interest rate
    q   : continuous dividend yield
    σ   : volatility
    W_t : standard Brownian motion

Under this model, the terminal price has a closed-form solution:

    S_T = S_0 * exp( (r - q - 0.5 σ²) T + σ sqrt(T) Z )

where Z ~ N(0, 1)

--------------------------------------------------------------------------------
OPTION CONTRACT
--------------------------------------------------------------------------------

European Call Option:
    Payoff at maturity T = max(S_T - K, 0)

The option can only be exercised at maturity.

--------------------------------------------------------------------------------
RISK-NEUTRAL PRICING PRINCIPLE
--------------------------------------------------------------------------------

Under the risk-neutral measure, the option price is:

    Price = exp(-r T) * E[ Payoff ]

This expectation is computed either:
  - analytically (Black–Scholes formula)
  - numerically (Monte Carlo simulation)

================================================================================
*/


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
    double price, delta, gamma;
};

// --- HELPERS ---
inline double normal_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

double cdf_naive(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// --- 1. BSM NAIVE ---
void bsm_naive(int n, const double* S, double K, double T,
               double r, double sigma, double q, Results* res) {
    const double sqrtT = std::sqrt(T);
    const double v_sqrtT = sigma * sqrtT;
    const double exp_rt = std::exp(-r * T);
    const double exp_qt = std::exp(-q * T);

    for (int i = 0; i < n; i++) {
        double d1 = (std::log(S[i] / K) +
                    (r - q + 0.5 * sigma * sigma) * T) / v_sqrtT;
        double d2 = d1 - v_sqrtT;

        res[i].price = S[i] * exp_qt * cdf_naive(d1)
                     - K * exp_rt * cdf_naive(d2);
        res[i].delta = exp_qt * cdf_naive(d1);
        res[i].gamma = (exp_qt * normal_pdf(d1)) / (S[i] * v_sqrtT);
    }
}

// --- 2. BSM HPC (MKL + OpenMP + Chunking) ---
void bsm_hpc(int n, const double* S, double K, double T,
             double r, double sigma, double q, Results* res) {

    #pragma omp master
    {
        std::cout << "[BSM HPC] Active threads      = "
                  << omp_get_num_threads() << "\n";
        std::cout << "[BSM HPC] Max OpenMP threads  = "
                  << omp_get_max_threads() << "\n";
    }



    const int CHUNK = 1024;
    const double sqrtT = std::sqrt(T);
    const double v_sqrtT = sigma * sqrtT;
    const double exp_rt = std::exp(-r * T);
    const double exp_qt = std::exp(-q * T);

    #pragma omp parallel
    {
        std::vector<double> d1(CHUNK), d2(CHUNK);
        std::vector<double> cdf1(CHUNK), cdf2(CHUNK);

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i += CHUNK) {
            int len = std::min(CHUNK, n - i);

            for (int j = 0; j < len; j++) {
                d1[j] = (std::log(S[i+j] / K) +
                        (r - q + 0.5 * sigma * sigma) * T) / v_sqrtT;
                d2[j] = d1[j] - v_sqrtT;
            }

            vdCdfNorm(len, d1.data(), cdf1.data());
            vdCdfNorm(len, d2.data(), cdf2.data());

            for (int j = 0; j < len; j++) {
                int idx = i + j;
                res[idx].price = S[idx] * exp_qt * cdf1[j]
                               - K * exp_rt * cdf2[j];
                res[idx].delta = exp_qt * cdf1[j];
                res[idx].gamma = (exp_qt * normal_pdf(d1[j]))
                               / (S[idx] * v_sqrtT);
            }
        }
    }
}

// --- 3. MC NAIVE ---
double mc_naive(int n, double S0, double K, double T,
                double r, double sigma, double q) {
    std::mt19937 gen(777);
    std::normal_distribution<double> dist(0.0, 1.0);

    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double vol = sigma * std::sqrt(T);

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol * dist(gen));
        sum += std::max(ST - K, 0.0);
    }
    return (sum / n) * std::exp(-r * T);
}

// --- 4. MC HPC (FIXED: thread-local MKL RNG) ---
double mc_hpc(int n, double S0, double K, double T,
              double r, double sigma, double q) {
    const int CHUNK = 8192;
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double vol = sigma * std::sqrt(T);

    double total_payoff = 0.0;

    #pragma omp parallel reduction(+:total_payoff)
    {

        #pragma omp master
    {
        std::cout << "[MC HPC] Active threads      = "
                  << omp_get_num_threads() << "\n";
        std::cout << "[MC HPC] Max OpenMP threads  = "
                  << omp_get_max_threads() << "\n";
    }


        int tid = omp_get_thread_num();
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MT19937, 777 + tid);

        std::vector<double> z(CHUNK);

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i += CHUNK) {
            int len = std::min(CHUNK, n - i);

            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
                          stream, len, z.data(), 0.0, 1.0);

            for (int j = 0; j < len; j++) {
                total_payoff += std::max(
                    S0 * std::exp(drift + vol * z[j]) - K, 0.0
                );
            }
        }

        vslDeleteStream(&stream);
    }

    return (total_payoff / n) * std::exp(-r * T);
}

// --- MAIN ---
int main() {
    const int N = 100'000'000;
    double S0 = 100.0, K = 105.0, T = 1.0;
    double r = 0.05, sigma = 0.2, q = 0.02;

    std::vector<double> S_vec(N, S0);
    std::vector<Results> res_naive(N), res_hpc(N);

    auto timer = [](auto&& f) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    std::cout << std::fixed << std::setprecision(6);

    double t1 = timer([&]{ bsm_naive(N, S_vec.data(), K, T, r, sigma, q, res_naive.data()); });
    double t2 = timer([&]{ bsm_hpc  (N, S_vec.data(), K, T, r, sigma, q, res_hpc.data()); });

    double mc_n, mc_h;
    double t3 = timer([&]{ mc_n = mc_naive(N, S0, K, T, r, sigma, q); });
    double t4 = timer([&]{ mc_h = mc_hpc  (N, S0, K, T, r, sigma, q); });

    double err = std::abs(mc_h - res_hpc[0].price);

    std::cout << "\nBSM Price : " << res_hpc[0].price << "\n";
    std::cout << "MC Price  : " << mc_h << "\n";
    std::cout << "Abs Error : " << err << "\n\n";

    printf("BSM Naive : %8.2f ms\n", t1);
    printf("BSM HPC   : %8.2f ms (%.1fx)\n", t2, t1/t2);
    printf("MC Naive  : %8.2f ms\n", t3);
    printf("MC HPC    : %8.2f ms (%.1fx)\n", t4, t3/t4);
}
