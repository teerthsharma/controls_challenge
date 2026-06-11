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
  assert np.any(controller.topology_signature)
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
