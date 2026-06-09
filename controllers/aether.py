from . import BaseController
import numpy as np

class AetherDSP:
    """
    Ported from AETHER-Link's TelemetryDSP (Rust).
    Maintains running statistics over the error stream to allow adaptive gain scheduling.
    """
    def __init__(self):
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0
        self.last_delta = 0.0
        self.spectral_energy = 0.0

    def update(self, delta):
        # Welford online variance
        self.count += 1
        delta_w = delta - self.mean
        self.mean += delta_w / self.count
        delta_new = delta - self.mean
        self.m2 += delta_w * delta_new

        # Chebyshev spectral energy (running RMS of delta differences)
        ddiff = delta - self.last_delta
        self.spectral_energy = 0.95 * self.spectral_energy + 0.05 * (ddiff * ddiff)
        self.last_delta = delta

    @property
    def variance(self):
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1.0)


class Controller(BaseController):
    """
    Aether-Link inspired DSP adaptive controller.
    Uses running variance and spectral energy of the error signal
    to dynamically adjust control effort, minimizing jerk.
    """
    def __init__(self):
        # Tuned Base Gains
        self.p_base = 0.4
        self.i_base = 0.1
        self.d_base = -0.1
        
        self.error_integral = 0
        self.prev_error = 0
        
        self.dsp = AetherDSP()

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # The primary stream value is the tracking error
        error = target_lataccel - current_lataccel
        error_diff = error - self.prev_error
        
        # Update Aether DSP state
        self.dsp.update(error)
        
        # Feature extraction
        variance = self.dsp.variance
        spectrum = np.sqrt(self.dsp.spectral_energy) if self.dsp.spectral_energy > 0 else 0.0
        
        # Adaptive gain modulation
        # High variance or high spectral energy (noise/jitter) -> dampens the P gain to avoid jerk.
        p_adapt = self.p_base / (1.0 + variance * 5.0 + spectrum * 2.0)
        
        # The Integral term builds over time
        self.error_integral += error
        
        # Compute control output
        steer = (p_adapt * error) + (self.i_base * self.error_integral) + (self.d_base * error_diff)
        
        self.prev_error = error
        return steer
