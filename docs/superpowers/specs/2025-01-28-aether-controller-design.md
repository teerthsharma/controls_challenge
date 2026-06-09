# Design: Aether v2 Adaptive Controller for comma Controls Challenge

## Date
2025-01-28

## Objective
Replace the broken Aether v1 adaptive controller (total_cost ~116–131) with a hard-engineered, physics-aware controller that beats the baseline PID (~100) and lands on the leaderboard (<100).

## Architecture

### 1. Feedforward (Bicycle Model)
```
steer_ff = K_ff * target_lataccel / (v_ego^2 + epsilon)
```
At high speed, lateral acceleration `a_lat = v^2 / R`. Small steer angle maps to large `a_lat`. Feedforward inverts this relationship, providing the bulk of the control signal without waiting for error to build.

### 2. Disturbance Rejection (Road Roll)
```
steer_roll = K_roll * roll_lataccel / (v_ego^2 + epsilon)
```
Banked road introduces gravity-induced lateral acceleration. Compensate proactively so the feedback loop only handles residual tracking error.

### 3. Preview Control
```
steer_preview = K_preview * (future_plan.lataccel[lookahead_steps] - target_lataccel)
```
The simulator provides 5 seconds of future trajectory. A weighted average of the first N future target steps gives a rate-of-change hint. Start steering early before the curve hits.

### 4. Feedback PID on Residual
After subtracting feedforward and disturbance terms, a conventional PID runs on the residual error:
```
error = target_lataccel - current_lataccel
steer_fb = Kp * error + Ki * integral(error) + Kd * derivative(error)
```
Derivative is computed from consecutive errors (no state filter needed for this simulator). Integral is clamped to [-2, 2] to prevent windup.

### 5. Adaptive Damping (Conservative)
If running variance of error > threshold AND spectral energy > threshold, reduce Kp by a fixed factor (e.g., 0.7). This only triggers in noisy regimes; clean tracking is preserved.

## Interface
Extends `BaseController` in `controllers/aether.py`.
Single file. No external dependencies beyond `numpy`.

## Testing Plan
1. Smoke test on `00000.csv` → verify no crash.
2. Quick eval on 10 segments vs PID → verify total_cost lower.
3. Full eval on 500 segments → gather statistics, verify mean total_cost < 100.
4. Generate `report.html` for PR submission.

## Success Criteria
- `total_cost` mean < 100 across 500+ segments.
- Beat PID baseline by at least 5 points.
- Code is small (<100 lines), hard-engineered, no clever tricks.
