#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <atomic>
#include <optional>
#include <thread>
#include <iomanip>
#include <mkl.h>
#include <omp.h>
#include <pthread.h> // For thread affinity on Linux

// --- 1. LOW-LATENCY INFRASTRUCTURE ---

// Structure representing a pricing task
struct PriceRequest {
    double S;
    double K;
};

// Single-Producer Single-Consumer (SPSC) Lock-Free Queue
class SPSCQueue {
private:
    static constexpr size_t Size = 2048; // Must be power of 2
    PriceRequest buffer[Size];
    
    // alignas(64) ensures head and tail are on different cache lines to prevent "False Sharing"
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};

public:
    // Pushes a new request (called by Market Data thread)
    bool push(const PriceRequest& req) {
        size_t t = tail.load(std::memory_order_relaxed);
        size_t next_t = (t + 1) & (Size - 1);
        if (next_t == head.load(std::memory_order_acquire)) return false; // Full
        
        buffer[t] = req;
        tail.store(next_t, std::memory_order_release);
        return true;
    }

    // Pops a request (called by Pricing Engine thread)
    std::optional<PriceRequest> pop() {
        size_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return std::nullopt; // Empty
        
        PriceRequest req = buffer[h];
        head.store((h + 1) & (Size - 1), std::memory_order_release);
        return req;
    }
};

// --- 2. OPTIMIZED HPC MC KERNEL ---

double mc_hpc_kernel(int n, double S0, double K, double T, double r, double sigma) {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 777);
    
    const int CHUNK = 16384; // Optimized for L1/L2 cache locality
    double total_payoff = 0.0;
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double df = std::exp(-r * T);

    #pragma omp parallel reduction(+:total_payoff)
    {
        // 64-byte aligned buffer for AVX-512/AVX2 SIMD efficiency
        double* z = (double*)mkl_malloc(CHUNK * sizeof(double), 64);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i += CHUNK) {
            int len = std::min(CHUNK, n - i);
            
            // Vectorized Random Number Generation
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, len, z, 0.0, 1.0);
            
            #pragma omp simd
            for (int j = 0; j < len; j++) {
                double ST = S0 * std::exp(drift + vol * z[j]);
                double diff = ST - K;
                // Branchless-friendly logic
                total_payoff += (diff > 0.0) ? diff : 0.0;
            }
        }
        mkl_free(z);
    }
    vslDeleteStream(&stream);
    return (total_payoff / n) * df;
}

// --- 3. SYSTEM TUNING: THREAD AFFINITY ---

void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

// --- 4. MAIN EXECUTION FLOW ---

int main() {
    SPSCQueue market_data_queue;
    std::atomic<bool> is_running{true};
    const int SIM_PATHS = 1000000; // 1M paths per tick for ultra-low latency

    std::cout << "Starting High-Frequency Pricing Engine..." << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << " | Paths per tick: " << SIM_PATHS << "\n" << std::endl;

    // CONSUMER THREAD: The "Worker" performing heavy math
    std::thread pricing_engine([&]() {
        pin_thread_to_core(1); // Pin engine to Core 1 to avoid context switching
        std::cout << "[Engine] Pinned to Core 1. Monitoring Queue..." << std::endl;
        
        while (is_running) {
            auto task = market_data_queue.pop();
            if (task) {
                auto start = std::chrono::high_resolution_clock::now();
                
                double price = mc_hpc_kernel(SIM_PATHS, task->S, task->K, 1.0, 0.05, 0.2);
                
                auto end = std::chrono::high_resolution_clock::now();
                double latency = std::chrono::duration<double, std::milli>(end - start).count();
                
                std::cout << std::fixed << std::setprecision(4)
                          << "[MATCH] Price for S=" << task->S << " is " << price 
                          << " | Compute Latency: " << latency << "ms" << std::endl;
            }
            // In a real HFT app, we would use a "busy-wait" loop. 
            // For this demo, we yield to keep the CPU fan quiet.
            std::this_thread::yield(); 
        }
    });

    // PRODUCER THREAD: Simulating a live Market Data Feed
    std::thread market_feed([&]() {
        pin_thread_to_core(2); // Pin feed to Core 2
        std::vector<double> ticks = {100.0, 101.5, 99.2, 103.4, 105.1};
        
        for (double spot : ticks) {
            std::this_thread::sleep_for(std::chrono::milliseconds(800)); // Simulate time between ticks
            
            if (market_data_queue.push({spot, 105.0})) {
                std::cout << "[Market] Ingested Tick: " << spot << std::endl;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(2));
        is_running = false;
    });

    market_feed.join();
    pricing_engine.join();

    std::cout << "\nEngine Shutdown Gracefully." << std::endl;
    return 0;
}