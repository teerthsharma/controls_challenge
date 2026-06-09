# Feature Request: Aether-Link Inspired Adaptive DSP Controller

## Objective
The baseline PID controller in the comma Controls Challenge achieves a total cost of ~106.8. To drop below the 100-point threshold and create a highly responsive, low-jerk controller, we can adapt the advanced Digital Signal Processing (DSP) telemetry pipeline from the **AETHER-Link** project. 

While AETHER-Link outputs a binary decision for storage prefetching, its underlying continuous feature extraction (Velocity, Variance, Spectral Energy, Entropy) can be repurposed to dynamically modulate control gains in an adaptive lateral acceleration controller.

## Proposed Architecture

1. **Error Stream instead of LBA Stream:**
   Instead of feeding Logical Block Addresses (LBAs), we treat the tracking error (`target_lataccel - current_lataccel`) as our incoming signal stream.

2. **Telemetry DSP Pipeline:**
   Port the `TelemetryDSP` from Rust to Python to run inside the steering loop:
   * **Velocity ($V$):** Rate of change of the error signal (acts as the Derivative component).
   * **Variance ($\sigma^2$):** Welford’s running variance over the error stream. This indicates track stability. If variance is high, the controller should dampen its response to avoid high jerk.
   * **Chebyshev Spectral Energy ($C$):** Running RMS of delta-differences. High spectral energy implies high-frequency noise or sharp curves, requiring adaptive gain reduction.
   * **History ($H$):** Exponential Moving Average (EMA) of error, acting as a smoother Integral component.

3. **Adaptive Continuous Output:**
   Instead of mapping to a Bloch sphere and measuring POVM effects (which result in a `bool`), we will use the telemetry features to dynamically adjust a base Proportional-Integral-Derivative (PID) controller. 

   * $P_{adapt} = P_{base} \times f(\text{Variance}, \text{Spectrum})$
   * $D_{adapt} = D_{base} \times g(\text{Velocity})$

## Theoretical Gain Visualization

By leveraging Aether-Link's sub-microsecond DSP math, we continuously shift the controller's aggressiveness. 

**Graph 1: Adaptive Gain ($P_{adapt}$) vs Error Variance ($\sigma^2$)**
```text
  P_adapt
    ^
P_0 | *\   <- Clean track: fast, aggressive tracking
    |   \
    |    * \
    |       \ 
    |        * \   <- Moderate noise: dampening begins
    |           *---_ 
    |                --__ <- High variance: P-gain squashed to minimize jerk
    +--------------------------> Variance (σ²)
```
*Mathematical Behavior:* $P_{adapt} \propto \frac{1}{1 + \alpha\sigma^2 + \beta C}$

**Graph 2: Jerk Minimization (AETHER vs Standard PID)**
```text
LatAccel Error
    ^
    |    /\        /     [PID] Overshoots & oscillates (High Jerk & High Cost)
    |   /  \  /\  /
    |  /    \/  \/
    | /  __---__         [AETHER] Adaptive dampening smooths the curve (Low Jerk)
    |/ --       --__
    +--------------------------> Time
```

## Expected Improvements

By dynamically suppressing overshoot immediately as noise is detected (via the spectral and variance tensors), the expected outcome is:
1. **Dramatically Lower Jerk:** Less oscillation means jerk cost is significantly slashed.
2. **Comparable LatAccel Cost:** We still track the baseline accurately but avoid the penalties of over-correction.

## Implementation Plan

1. Create a new controller `controllers/aether.py`.
2. Implement the `AetherDSP` class, porting the Welford variance and Chebyshev spectral energy algorithms from `aether-link/src/lib.rs`.
3. Wrap this inside a class extending `BaseController`.
4. Tune the base parameters against `eval.py` to minimize the `lataccel_cost` and `jerk_cost`.