# High-Performance Black–Scholes & Monte Carlo Option Pricing Engine

This project implements a **European Call Option pricing engine** under the  
**Black–Scholes–Merton (BSM)** framework using both:

- **Closed-form analytical formulas**
- **Monte Carlo simulation under the risk-neutral measure**

The implementation focuses on **numerical correctness**, **financial soundness**, and
**high-performance computing (HPC)** using **SIMD (Intel MKL)** and **multithreading (OpenMP)**.

---

## 1. Financial Model

The underlying asset price \( S_t \) is assumed to follow **Geometric Brownian Motion (GBM)**:

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t
\]

Where:
- \( r \) : risk-free interest rate  
- \( q \) : continuous dividend yield  
- \( \sigma \) : volatility  
- \( W_t \) : standard Brownian motion  

The closed-form solution at maturity \( T \) is:

\[
S_T = S_0 \exp\left((r - q - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z \right),
\quad Z \sim \mathcal{N}(0,1)
\]

---

## 2. Option Contract

**European Call Option**

- Payoff at maturity:
\[
\text{Payoff} = \max(S_T - K, 0)
\]
- Exercise allowed **only at maturity**

---

## 3. Risk-Neutral Pricing Principle

Under the **risk-neutral measure**, the option price is:

\[
C = e^{-rT} \mathbb{E}[\text{Payoff}]
\]

This expectation is evaluated using:
- **Analytical Black–Scholes formula**
- **Monte Carlo simulation**

---

## 4. Analytical Black–Scholes Pricing (BSM)

### Definitions

\[
d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}},
\quad
d_2 = d_1 - \sigma \sqrt{T}
\]

### Call Option Price

\[
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)
\]

Where \( N(\cdot) \) is the standard normal CDF.

---

## 5. Greeks Computed Analytically

### Delta (First Derivative)

Measures sensitivity of option price to the underlying price:

\[
\Delta = \frac{\partial C}{\partial S} = e^{-qT} N(d_1)
\]

Used for **hedging**.

---

### Gamma (Second Derivative)

Measures sensitivity of Delta to the underlying price:

\[
\Gamma = \frac{\partial^2 C}{\partial S^2}
= \frac{e^{-qT} \phi(d_1)}{S \sigma \sqrt{T}}
\]

Where:
\[
\phi(d_1) = \frac{1}{\sqrt{2\pi}} e^{-d_1^2 / 2}
\]

High gamma indicates rapid delta changes (risk concentration near ATM).

---

## 6. Monte Carlo Pricing

Monte Carlo simulation estimates the risk-neutral expectation:

\[
C \approx e^{-rT} \frac{1}{N}
\sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)
\]

### Key Points
- Only the **price** is computed via Monte Carlo
- Greeks are **not** computed via Monte Carlo to avoid high variance
- Analytical Greeks are preferred whenever available

---

## 7. Why Greeks Are Not Computed via Monte Carlo

| Greek | MC Suitability | Reason |
|-----|---------------|-------|
| Price | ✅ | Expectation |
| Delta | ⚠️ | Noisy estimators |
| Gamma | ❌ | Very high variance |

Analytical Greeks are:
- Exact
- Faster
- Numerically stable

This is a **deliberate design decision**, not a limitation.

---

## 8. High-Performance Computing Design

### SIMD (Vectorization)
- Intel MKL Vector Math Library (VML)
- Functions like `vdCdfNorm` and `vdRngGaussian`
- Internally uses AVX / AVX-512 instructions

### Multithreading
- OpenMP parallelism across CPU cores
- Each thread owns:
  - Its own RNG stream
  - Its own working buffers

### RNG Design
- One **MKL VSL RNG stream per thread**
- Ensures:
  - Thread safety
  - Statistical independence
  - Reproducibility

---

## 9. Validation

Monte Carlo pricing is validated against the analytical Black–Scholes price:

\[
\text{Absolute Error} = |C_{\text{MC}} - C_{\text{BS}}|
\]

This confirms:
- Correct stochastic modeling
- Proper risk-neutral implementation

---

## 10. Summary

This project demonstrates:

- Exact analytical pricing and Greeks under BSM
- Correct risk-neutral Monte Carlo simulation
- Proper financial modeling choices
- HPC techniques using SIMD + multithreading
- Production-quality numerical design

---

## One-Line Description (Interview-Ready)

> A high-performance European option pricing engine that combines analytical Black–Scholes formulas with SIMD-accelerated Monte Carlo simulation under the risk-neutral measure.

