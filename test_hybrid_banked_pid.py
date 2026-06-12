from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tinyphysics import FuturePlan, State


def make_plan(value: float = 0.2) -> FuturePlan:
  lat = np.linspace(value, value + 0.4, 49).tolist()
  return FuturePlan(
    lataccel=lat,
    roll_lataccel=[0.01] * 49,
    v_ego=[25.0] * 49,
    a_ego=[0.0] * 49,
  )


def test_controller_returns_finite_clipped_action():
  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  state = State(roll_lataccel=4.0, v_ego=45.0, a_ego=-3.0)

  for _ in range(20):
    action = controller.update(100.0, -100.0, state, make_plan(4.0))
    assert np.isfinite(action)
    assert -2.0 <= action <= 2.0


def test_missing_json_falls_back_to_safe_pid(monkeypatch, tmp_path):
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(tmp_path / "missing.json"))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  action = controller.update(
    0.5,
    0.1,
    State(roll_lataccel=0.0, v_ego=20.0, a_ego=0.0),
    make_plan(0.5),
  )

  assert np.isfinite(action)
  assert controller.banks == {}
  assert -2.0 <= action <= 2.0


def test_malformed_json_falls_back_to_safe_pid(monkeypatch, tmp_path):
  coeff_path = tmp_path / "bad.json"
  coeff_path.write_text("{not json", encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  action = controller.update(
    -0.5,
    0.3,
    State(roll_lataccel=0.0, v_ego=20.0, a_ego=0.0),
    make_plan(-0.5),
  )

  assert np.isfinite(action)
  assert controller.banks == {}
  assert -2.0 <= action <= 2.0


def test_derivative_clutch_suppresses_d_scale(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
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
      "integral_limit": 8.0
    },
    "banks": {}
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  state = State(roll_lataccel=0.0, v_ego=20.0, a_ego=0.0)
  for target in [0.0] * 12:
    controller.update(target, 0.0, state, make_plan(target))
  controller.update(5.0, -5.0, state, make_plan(5.0))

  assert controller.last_clutch is True
  assert 0.0 <= controller.last_d_scale < 1.0


def test_warmup_builds_topology_signature():
  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  state = State(roll_lataccel=0.1, v_ego=25.0, a_ego=0.01)
  for idx in range(90):
    target = float(np.sin(idx / 10.0) * 0.3)
    current = float(np.cos(idx / 11.0) * 0.1)
    controller.update(target, current, state, make_plan(target))

  assert controller.topology_signature.shape == (8,)
  assert controller.topology_exact_signature.shape == (16,)
  assert np.any(controller.topology_signature)
  assert controller.topology_features.shape == (8,)
  assert np.all(np.isfinite(controller.topology_features))


def test_s3_marker_detects_large_tracking_jump(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
      "p": 0.195,
      "i": 0.100,
      "d": -0.053,
      "adaptive": 0.18,
      "clutch_d_scale": 0.0,
      "s3_topology": 0.12,
      "s3_error_push": 0.16,
      "s3_chaos_track": 0.08,
      "s3_curvature_damping": 0.20,
      "integral_limit": 8.0
    },
    "banks": {}
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  state = State(roll_lataccel=0.0, v_ego=20.0, a_ego=0.0)
  for _ in range(12):
    controller.update(0.0, 0.0, state, make_plan(0.0))
  controller.update(4.0, -4.0, state, make_plan(4.0))

  assert controller.last_s3_jump > 0.0
  assert controller.last_s3_chaos > 0.0
  assert np.isfinite(controller.last_effective_error)


def test_topology_error_push_does_not_hide_error(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
      "p": 0.195,
      "i": 0.100,
      "d": -0.053,
      "adaptive": 0.0,
      "clutch_d_scale": 0.0,
      "s3_topology": 0.5,
      "s3_error_push": 0.16,
      "s3_chaos_track": 0.08,
      "s3_curvature_damping": 0.20,
      "integral_limit": 8.0
    },
    "banks": {}
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  state = State(roll_lataccel=0.0, v_ego=20.0, a_ego=0.0)
  target = 0.8
  current = 0.1
  controller.update(target, current, state, make_plan(target))

  assert abs(controller.last_effective_error) >= abs(target - current) * 0.99
  assert np.isfinite(controller.last_effective_error)


def test_topology_bank_requires_exact_match_when_threshold_zero(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
      "p": 0.102,
      "i": 0.138,
      "d": 0.004,
      "safety_fallback": 0.25,
      "adaptive": 0.0,
      "s3_topology": 0.0,
      "topology_bank_max_distance": 0.0,
      "integral_limit": 8.0
    },
    "banks": {},
    "topology_banks": {
      "exact": {
        "signature": [1, 2, 3, 4, 5, 6, 7, 8],
        "coeffs": {"safety_fallback": 1.0}
      }
    }
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  controller.topology_signature = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
  exact_coeffs = controller._select_coeffs(np.zeros(8, dtype=np.int64))
  assert exact_coeffs["safety_fallback"] == 1.0

  controller.topology_signature = np.asarray([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.int64)
  near_coeffs = controller._select_coeffs(np.zeros(8, dtype=np.int64))
  assert near_coeffs["safety_fallback"] == 0.25
  assert controller.active_topology_bank_key is None


def test_high_future_span_guard_raises_fallback_weight(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
      "p": 0.102,
      "i": 0.138,
      "d": 0.004,
      "ff": 0.09,
      "future": 0.02,
      "safety_fallback": 0.1,
      "adaptive": 0.2,
      "s3_topology": 0.2,
      "span_guard_threshold": 1.0,
      "span_guard_safety": 0.95,
      "integral_limit": 8.0
    },
    "banks": {},
    "topology_banks": {}
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  state = State(roll_lataccel=0.0, v_ego=30.0, a_ego=0.0)
  high_span_plan = FuturePlan(
    lataccel=np.linspace(-2.0, 2.0, 49).tolist(),
    roll_lataccel=[0.0] * 49,
    v_ego=[30.0] * 49,
    a_ego=[0.0] * 49,
  )
  controller.update(0.5, 0.0, state, high_span_plan)

  assert controller.last_span_guard is True
  assert controller.last_safety >= 0.95


def test_inverse_feedforward_can_affect_action(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
      "p": 0.0,
      "i": 0.0,
      "d": 0.0,
      "ff": 0.0,
      "roll": 0.0,
      "future": 0.0,
      "prev": 0.0,
      "safety_fallback": 0.0,
      "adaptive": 0.0,
      "s3_topology": 0.0,
      "inverse_ff": 1.0,
      "inverse_ff_coeffs": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "integral_limit": 8.0
    },
    "banks": {},
    "topology_banks": {}
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  action = controller.update(
    0.8,
    0.1,
    State(roll_lataccel=0.0, v_ego=30.0, a_ego=0.0),
    make_plan(0.8),
  )

  assert np.isfinite(controller.last_inverse_ff)
  assert controller.last_inverse_ff == 0.4
  assert action == 0.4


def test_target_weight_changes_action(monkeypatch, tmp_path):
  from controllers.hybrid_banked_pid import Controller

  def run_with_weight(weight: float) -> float:
    coeff_path = tmp_path / f"coeffs_{weight}.json"
    coeff_path.write_text(json.dumps({
      "__global__": {
        "p": 0.4,
        "i": 0.0,
        "d": 0.0,
        "ff": 0.0,
        "roll": 0.0,
        "future": 0.0,
        "prev": 0.0,
        "safety_fallback": 0.0,
        "adaptive": 0.0,
        "s3_topology": 0.0,
        "target_weight": weight,
        "current_weight": 1.0,
        "integral_limit": 8.0
      },
      "banks": {},
      "topology_banks": {}
    }), encoding="utf-8")
    monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))
    controller = Controller()
    return controller.update(
      1.0,
      0.2,
      State(roll_lataccel=0.0, v_ego=25.0, a_ego=0.0),
      make_plan(1.0),
    )

  low_weight_action = run_with_weight(0.5)
  high_weight_action = run_with_weight(1.0)

  assert np.isfinite(low_weight_action)
  assert np.isfinite(high_weight_action)
  assert high_weight_action > low_weight_action


def test_jerk_smooth_reduces_action_jump(monkeypatch, tmp_path):
  from controllers.hybrid_banked_pid import Controller

  def second_action(smooth: float) -> float:
    coeff_path = tmp_path / f"smooth_{smooth}.json"
    coeff_path.write_text(json.dumps({
      "__global__": {
        "p": 0.0,
        "i": 0.0,
        "d": 0.0,
        "ff": 1.0,
        "roll": 0.0,
        "future": 0.0,
        "prev": 0.0,
        "jerk_smooth": smooth,
        "safety_fallback": 0.0,
        "adaptive": 0.0,
        "s3_topology": 0.0,
        "integral_limit": 8.0
      },
      "banks": {},
      "topology_banks": {}
    }), encoding="utf-8")
    monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))
    controller = Controller()
    state = State(roll_lataccel=0.0, v_ego=25.0, a_ego=0.0)
    controller.update(0.0, 0.0, state, make_plan(0.0))
    return controller.update(1.0, 0.0, state, make_plan(1.0))

  unsmoothed = second_action(0.0)
  smoothed = second_action(0.25)

  assert np.isfinite(unsmoothed)
  assert np.isfinite(smoothed)
  assert 0.0 < smoothed < unsmoothed


def test_tail_bank_override_does_not_affect_unmatched_signature(monkeypatch, tmp_path):
  coeff_path = tmp_path / "coeffs.json"
  coeff_path.write_text(json.dumps({
    "__global__": {
      "p": 0.1,
      "i": 0.1,
      "d": 0.0,
      "safety_fallback": 0.2,
      "adaptive": 0.0,
      "s3_topology": 0.0,
      "topology_bank_max_distance": 0.0,
      "integral_limit": 8.0
    },
    "banks": {},
    "topology_banks": {
      "tail_00042": {
        "signature": [10, 20, 30, 40, 50, 60, 70, 80],
        "coeffs": {"safety_fallback": 0.9, "ff": 0.5}
      }
    }
  }), encoding="utf-8")
  monkeypatch.setenv("HYBRID_BANKED_PID_COEFF_PATH", str(coeff_path))

  from controllers.hybrid_banked_pid import Controller

  controller = Controller()
  controller.topology_signature = np.asarray([10, 20, 30, 40, 50, 60, 70, 81], dtype=np.int64)
  coeffs = controller._select_coeffs(np.zeros(8, dtype=np.int64))

  assert coeffs["safety_fallback"] == 0.2
  assert coeffs.get("ff", 0.0) != 0.5
  assert controller.active_topology_bank_key is None
