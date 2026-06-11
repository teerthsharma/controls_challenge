from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import BaseController


COEFF_PATH = Path(__file__).with_name("hybrid_banked_pid_coeffs.json")
COEFF_ENV = "HYBRID_BANKED_PID_COEFF_PATH"
EPS = 1e-9
PRIMES = (2, 3, 5)
VP_ZERO_SENTINEL = 64


def _finite(value: Any, default: float = 0.0) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return default
  return out if np.isfinite(out) else default


def _clip(value: float, low: float, high: float) -> float:
  return float(np.clip(_finite(value), low, high))


def _normalize(values: Iterable[float]) -> np.ndarray:
  vec = np.asarray([_finite(v) for v in values], dtype=np.float64)
  return vec / (float(np.linalg.norm(vec)) + EPS)


def _series(values: Any, limit: int = 49) -> np.ndarray:
  if values is None:
    return np.zeros(0, dtype=np.float64)
  try:
    arr = np.asarray(list(values)[:limit], dtype=np.float64)
  except (TypeError, ValueError):
    return np.zeros(0, dtype=np.float64)
  return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _future_features(future_plan: Any) -> tuple[float, float, float, float, float]:
  lat = _series(getattr(future_plan, "lataccel", None), limit=49)
  if lat.size == 0:
    return 0.0, 0.0, 0.0, 0.0, 0.0
  mean = float(np.mean(lat))
  last = float(lat[-1])
  slope = float((lat[-1] - lat[0]) / max(lat.size - 1, 1))
  span = float(np.max(lat) - np.min(lat))
  curvature = float(np.mean(np.diff(lat, n=2))) if lat.size >= 3 else 0.0
  return mean, last, slope, span, curvature


def _quantize_real(value: float, scale: int = 1000, clip: int = 1_000_000) -> int:
  if not np.isfinite(value):
    return 0
  return int(np.clip(np.rint(float(value) * scale), -clip, clip))


def _quantize_vector(values: Iterable[float], scale: int = 1000, clip: int = 1_000_000) -> np.ndarray:
  return np.asarray([_quantize_real(v, scale=scale, clip=clip) for v in values], dtype=np.int64)


def _v_p(value: int, p: int) -> int:
  n = abs(int(value))
  if n == 0:
    return VP_ZERO_SENTINEL
  count = 0
  while n % p == 0:
    count += 1
    n //= p
  return count


def _dist_p(x: int, y: int, p: int) -> float:
  diff = int(x) - int(y)
  if diff == 0:
    return 0.0
  return float(p ** (-_v_p(diff, p)))


def _padic_distance_vector(a: Iterable[int], b: Iterable[int]) -> float:
  aa = np.asarray(list(a), dtype=np.int64)
  bb = np.asarray(list(b), dtype=np.int64)
  if aa.shape != bb.shape:
    raise ValueError("p-adic vectors must have identical shape")
  distances = []
  for p in PRIMES:
    distances.append(max((_dist_p(x, y, p) for x, y in zip(aa, bb)), default=0.0))
  return float(max(distances, default=0.0))


def _nearest_bank(signature: Iterable[int], banks: dict[str, dict[str, Any]]) -> str | None:
  sig = list(signature)
  best_key = None
  best_distance = float("inf")
  for key, bank in banks.items():
    candidate = bank.get("signature")
    if not isinstance(candidate, list):
      continue
    try:
      distance = _padic_distance_vector(sig[:len(candidate)], candidate)
    except (TypeError, ValueError):
      continue
    if distance < best_distance:
      best_key = key
      best_distance = distance
  return best_key


class Welford:
  def __init__(self) -> None:
    self.count = 0
    self.mean = 0.0
    self.m2 = 0.0

  def update(self, value: float) -> None:
    self.count += 1
    delta = value - self.mean
    self.mean += delta / self.count
    self.m2 += delta * (value - self.mean)

  @property
  def variance(self) -> float:
    if self.count < 2:
      return 0.0
    return float(self.m2 / (self.count - 1))

  @property
  def std(self) -> float:
    return float(np.sqrt(max(self.variance, 0.0)))


class RingEntropy:
  def __init__(self, size: int = 16) -> None:
    self.values = np.full(size, 1e-3, dtype=np.float64)
    self.idx = 0

  def update(self, value: float) -> None:
    self.values[self.idx] = max(abs(value), 1e-3)
    self.idx = (self.idx + 1) % self.values.size

  @property
  def entropy(self) -> float:
    return float(max(0.0, np.log(self.values.size) - np.mean(np.log(self.values))))


class Controller(BaseController):
  """Banked p-adic PID with Teerth-style topology regime selection."""

  FALLBACK_COEFFS = {
    "p": 0.195,
    "i": 0.100,
    "d": -0.053,
    "ff": 0.0,
    "roll": 0.0,
    "future": 0.0,
    "prev": 0.0,
    "safety_fallback": 1.0,
    "adaptive": 0.18,
    "clutch_d_scale": 0.0,
    "chaos_action_scale": 0.18,
    "p_chaos_drop": 0.18,
    "i_chaos_drop": 0.25,
    "d_chaos_lift": 0.10,
    "ff_plan_lift": 0.06,
    "road_damping": 0.08,
    "s3_topology": 0.12,
    "s3_error_push": 0.16,
    "s3_chaos_track": 0.08,
    "s3_curvature_damping": 0.20,
    "integral_limit": 8.0,
  }

  def __init__(self) -> None:
    self.payload = self._load_payload()
    self.global_coeffs = dict(self.FALLBACK_COEFFS)
    self.global_coeffs.update(self.payload.get("__global__", {}))
    self.banks = self._load_banks(self.payload)
    self.topology_banks = self._load_banks({"banks": self.payload.get("topology_banks", {})})
    self.active_bank_key: str | None = None
    self.active_topology_bank_key: str | None = None

    self.prev_error = 0.0
    self.error_integral = 0.0
    self.prev_action = 0.0
    self.step_count = 0

    self.error_stats = Welford()
    self.diff_stats = Welford()
    self.s3_stats = Welford()
    self.entropy = RingEntropy()
    self.spectral_energy = 0.0
    self.last_error_diff = 0.0
    self.prev_s3_jump = 0.0
    self.last_s3_jump = 0.0
    self.last_s3_curvature = 0.0
    self.last_s3_chaos = 0.0
    self.last_effective_error = 0.0
    self.q_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self.last_clutch = False
    self.last_d_scale = 1.0
    self.warmup_rows: list[np.ndarray] = []
    self.topology_signature = np.zeros(8, dtype=np.int64)
    self.topology_features = np.zeros(8, dtype=np.float64)

  def _load_payload(self) -> dict[str, Any]:
    coeff_path = Path(os.environ.get(COEFF_ENV, str(COEFF_PATH)))
    if not coeff_path.exists():
      return {}
    try:
      with coeff_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    except (OSError, json.JSONDecodeError):
      return {}
    return payload if isinstance(payload, dict) else {}

  def _load_banks(self, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_banks = payload.get("banks", {})
    if not isinstance(raw_banks, dict):
      return {}
    banks: dict[str, dict[str, Any]] = {}
    for key, raw in raw_banks.items():
      if not isinstance(raw, dict) or not isinstance(raw.get("signature"), list):
        continue
      try:
        signature = [int(v) for v in raw["signature"]]
      except (TypeError, ValueError):
        continue
      coeffs = raw.get("coeffs", {})
      banks[str(key)] = {
        "signature": signature,
        "coeffs": coeffs if isinstance(coeffs, dict) else {},
      }
    return banks

  def _coeff_value(self, coeffs: dict[str, Any], key: str) -> float:
    default = self.FALLBACK_COEFFS[key]
    value = coeffs.get(key, self.global_coeffs.get(key, default))
    if isinstance(value, list):
      value = value[0] if value else default
    return _finite(value, float(default))

  def _build_signature(
      self,
      target: float,
      current: float,
      state: Any,
      future_mean: float,
      future_last: float,
      future_span: float,
      error: float,
      error_diff: float,
  ) -> np.ndarray:
    roll = _finite(getattr(state, "roll_lataccel", 0.0))
    speed = _finite(getattr(state, "v_ego", 0.0))
    accel = _finite(getattr(state, "a_ego", 0.0))
    return _quantize_vector([
      target,
      future_mean,
      future_last,
      current + 0.4,
      roll + 0.5,
      speed / 100.0 + 0.4,
      accel + 0.7,
      future_span + 0.6,
      error,
      self.error_integral / max(self._coeff_value(self.global_coeffs, "integral_limit"), EPS),
      error_diff,
      self.prev_action,
    ])

  def _select_coeffs(self, signature: np.ndarray) -> dict[str, Any]:
    coeffs = dict(self.global_coeffs)
    self.active_bank_key = _nearest_bank(signature[:8].tolist(), self.banks)
    if self.active_bank_key is not None:
      coeffs.update(self.banks[self.active_bank_key].get("coeffs", {}))
    if self.topology_signature.any():
      self.active_topology_bank_key = _nearest_bank(self.topology_signature.tolist(), self.topology_banks)
      if self.active_topology_bank_key is not None:
        coeffs.update(self.topology_banks[self.active_topology_bank_key].get("coeffs", {}))
    return coeffs

  def _record_warmup(
      self,
      target: float,
      current: float,
      roll: float,
      speed: float,
      accel: float,
      future_mean: float,
      future_slope: float,
      future_curvature: float,
      error: float,
      error_diff: float,
  ) -> None:
    if self.step_count >= 80:
      return
    self.warmup_rows.append(np.asarray([
      target,
      current,
      roll,
      speed / 40.0,
      accel,
      future_mean,
      future_slope * 10.0,
      future_curvature * 100.0,
      error,
      error_diff,
    ], dtype=np.float64))
    if len(self.warmup_rows) > 80:
      self.warmup_rows = self.warmup_rows[-80:]

  def _component_fraction(self, points: np.ndarray) -> float:
    if len(points) < 4:
      return 0.0
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    upper = dist[np.triu_indices(len(points), k=1)]
    threshold = float(np.quantile(upper, 0.35)) if upper.size else 0.0
    if threshold <= EPS:
      return 0.0
    parent = list(range(len(points)))

    def find(i: int) -> int:
      while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
      return i

    for i in range(len(points)):
      for j in range(i + 1, len(points)):
        if dist[i, j] <= threshold:
          ri = find(i)
          rj = find(j)
          if ri != rj:
            parent[rj] = ri
    components = len({find(i) for i in range(len(points))})
    return float((components - 1) / max(len(points) - 1, 1))

  def _loop_proxy(self, points: np.ndarray) -> float:
    if len(points) < 6:
      return 0.0
    centered = points - np.mean(points, axis=0, keepdims=True)
    try:
      _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
      return 0.0
    if singular_values.size < 2:
      return 0.0
    return float(singular_values[1] / (singular_values[0] + EPS))

  def _refresh_topology_signature(self) -> None:
    if self.topology_signature.any() or len(self.warmup_rows) < 20:
      return
    points = np.asarray(self.warmup_rows, dtype=np.float64)
    points = (points - np.mean(points, axis=0, keepdims=True)) / (np.std(points, axis=0, keepdims=True) + EPS)
    error = points[:, 8]
    d_error = points[:, 9]
    velocity = np.linalg.norm(np.diff(points, axis=0), axis=1)
    hist, _ = np.histogram(velocity, bins=8)
    total = float(np.sum(hist))
    probs = hist.astype(np.float64) / total if total > 0.0 else np.zeros_like(hist, dtype=np.float64)
    probs = probs[probs > 0.0]
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    tracking = _normalize([float(np.mean(error)), float(np.mean(d_error)), self.error_integral])
    road = _normalize([
      float(np.mean(points[:, 2])),
      float(np.mean(points[:, 3])),
      float(np.mean(points[:, 4])),
    ])
    plan = _normalize([
      float(np.mean(points[:, 5])),
      float(np.mean(points[:, 6])),
      float(np.mean(points[:, 7])),
    ])
    self.topology_features = np.asarray([
      self._component_fraction(points),
      self._loop_proxy(points),
      np.tanh(entropy / 2.0),
      np.tanh(float(np.var(error))),
      np.tanh(float(np.mean(np.abs(d_error))) * 3.0),
      np.tanh(float(np.std(points[:, 7]))),
      np.clip(float(np.dot(tracking, plan)), -1.0, 1.0),
      np.clip(float(np.dot(tracking, road)), -1.0, 1.0),
    ], dtype=np.float64)
    self.topology_signature = _quantize_vector(self.topology_features, scale=1000, clip=2000)

  def _mark_s3_chaos(
      self,
      error: float,
      raw_diff: float,
      roll: float,
      future_curvature: float,
  ) -> tuple[np.ndarray, float, float, float]:
    q_now = _normalize([error, raw_diff, self.error_integral, roll + 0.35 * future_curvature])
    jump = float(np.arccos(np.clip(np.dot(q_now, self.q_ref), -1.0, 1.0)))
    curvature = abs(jump - self.prev_s3_jump)
    self.s3_stats.update(jump)

    variance_norm = min(1.0, self.error_stats.variance / 0.35)
    spectral_norm = min(1.0, self.spectral_energy / 0.20)
    entropy_norm = min(1.0, self.entropy.entropy / 5.0)
    jump_norm = min(1.0, jump / 1.2)
    curvature_norm = min(1.0, curvature / 0.8)
    topology_frag = float(self.topology_features[0]) if self.topology_features.size else 0.0
    topology_loop = float(self.topology_features[1]) if self.topology_features.size else 0.0
    chaos = _clip(
      0.22 * variance_norm
      + 0.18 * spectral_norm
      + 0.12 * entropy_norm
      + 0.24 * jump_norm
      + 0.16 * curvature_norm
      + 0.05 * topology_frag
      + 0.03 * topology_loop,
      0.0,
      1.0,
    )

    self.prev_s3_jump = jump
    self.last_s3_jump = jump
    self.last_s3_curvature = curvature
    self.last_s3_chaos = chaos
    return q_now, jump, curvature, chaos

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    target = _finite(target_lataccel)
    current = _finite(current_lataccel)
    future_mean, future_last, future_slope, future_span, future_curvature = _future_features(future_plan)
    roll = _finite(getattr(state, "roll_lataccel", 0.0))
    speed = _finite(getattr(state, "v_ego", 0.0))
    accel = _finite(getattr(state, "a_ego", 0.0))

    error = target - current
    raw_diff = error - self.prev_error
    self._record_warmup(
      target, current, roll, speed, accel, future_mean, future_slope, future_curvature, error, raw_diff
    )
    if self.step_count >= 80:
      self._refresh_topology_signature()
    self.error_stats.update(error)
    self.diff_stats.update(abs(raw_diff))
    self.entropy.update(error)
    raw_ddiff = raw_diff - self.last_error_diff
    self.spectral_energy = 0.95 * self.spectral_energy + 0.05 * raw_ddiff * raw_ddiff
    q_now, s3_jump, s3_curvature, chaos = self._mark_s3_chaos(error, raw_diff, roll, future_curvature)

    diff_boundary = self.diff_stats.mean + 2.0 * self.diff_stats.std
    s3_boundary = self.s3_stats.mean + 2.0 * self.s3_stats.std
    clutch = (
      self.diff_stats.count > 8
      and self.s3_stats.count > 8
      and (abs(raw_diff) > diff_boundary or s3_jump > s3_boundary)
    )

    signature = self._build_signature(
      target, current, state, future_mean, future_last, future_span, error, raw_diff
    )
    coeffs = self._select_coeffs(signature)

    integral_limit = _clip(self._coeff_value(coeffs, "integral_limit"), 1.0, 20.0)
    adaptive = _clip(self._coeff_value(coeffs, "adaptive"), 0.0, 0.6)
    topology_strength = _clip(self._coeff_value(coeffs, "s3_topology"), 0.0, 0.5)
    clutch_strength = adaptive if clutch else 0.0
    integral_gain = 1.0 - 0.65 * clutch_strength - 0.25 * topology_strength * chaos
    self.error_integral = _clip(
      self.error_integral + integral_gain * error,
      -integral_limit,
      integral_limit,
    )

    clutch_d_scale = _clip(self._coeff_value(coeffs, "clutch_d_scale"), 0.0, 1.0)
    error_diff = raw_diff * (1.0 - (1.0 - clutch_d_scale) * clutch_strength)
    spectral_norm = min(1.0, self.spectral_energy / 0.20)

    tracking = _normalize([error, error_diff, self.error_integral])
    road = _normalize([roll, speed / 40.0, accel])
    plan = _normalize([future_mean, future_slope * 10.0, future_curvature * 100.0])
    plan_alignment = _clip(float(np.dot(tracking, plan)), -1.0, 1.0)
    road_conflict = abs(_clip(float(np.dot(tracking, road)), -1.0, 1.0))
    coherence = 1.0 - chaos
    track_push = topology_strength * self._coeff_value(coeffs, "s3_error_push") * coherence * max(plan_alignment, 0.0)
    chaos_track = topology_strength * self._coeff_value(coeffs, "s3_chaos_track") * chaos
    effective_error = error * _clip(1.0 + track_push + chaos_track, 0.85, 1.18)
    self.last_effective_error = float(effective_error)

    p_scale = 1.0 - adaptive * self._coeff_value(coeffs, "p_chaos_drop") * chaos
    i_scale = 1.0 - adaptive * self._coeff_value(coeffs, "i_chaos_drop") * chaos
    d_scale = 1.0 + adaptive * self._coeff_value(coeffs, "d_chaos_lift") * spectral_norm
    d_scale += adaptive * self._coeff_value(coeffs, "road_damping") * road_conflict
    if clutch:
      d_scale *= 1.0 - (1.0 - clutch_d_scale) * clutch_strength
    curvature_damping = topology_strength * self._coeff_value(coeffs, "s3_curvature_damping") * min(1.0, s3_curvature / 0.8)
    d_scale *= _clip(1.0 - curvature_damping, 0.70, 1.0)
    ff_scale = 1.0 + adaptive * self._coeff_value(coeffs, "ff_plan_lift") * max(plan_alignment, 0.0)

    p_scale = _clip(p_scale, 0.70, 1.15)
    i_scale = _clip(i_scale, 0.65, 1.05)
    d_scale = _clip(d_scale, 0.0, 1.35)
    ff_scale = _clip(ff_scale, 0.92, 1.12)

    future_signal = future_mean + future_slope
    fallback_action = (
      self.FALLBACK_COEFFS["p"] * effective_error
      + self.FALLBACK_COEFFS["i"] * self.error_integral
      + self.FALLBACK_COEFFS["d"] * error_diff
    )
    learned_action = (
      ff_scale * self._coeff_value(coeffs, "ff") * target
      + p_scale * self._coeff_value(coeffs, "p") * effective_error
      + i_scale * self._coeff_value(coeffs, "i") * self.error_integral
      + d_scale * self._coeff_value(coeffs, "d") * error_diff
      + self._coeff_value(coeffs, "roll") * roll
      + self._coeff_value(coeffs, "future") * future_signal
    )

    safety = _clip(self._coeff_value(coeffs, "safety_fallback"), 0.0, 1.0)
    action = safety * fallback_action + (1.0 - safety) * learned_action
    prev = _clip(self._coeff_value(coeffs, "prev"), 0.0, 0.5)
    action = (1.0 - prev) * action + prev * self.prev_action
    action = _clip(action, -2.0, 2.0)

    self.prev_error = error
    self.prev_action = action
    self.last_error_diff = error_diff
    self.q_ref = _normalize(0.97 * self.q_ref + 0.03 * q_now)
    self.last_clutch = bool(clutch)
    self.last_d_scale = float(d_scale)
    self.step_count += 1
    return action
