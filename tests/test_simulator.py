"""Tests for pure functions in src/simulator/online_simulator.py."""

from __future__ import annotations

import uuid

from src.simulator.online_simulator import _build_event, _click_prob, _sample_lifetime


class TestClickProb:
    def test_high_score_high_temp_gives_high_prob(self) -> None:
        prob = _click_prob(0.9, 2.0)
        assert prob > 0.85

    def test_low_score_gives_low_prob(self) -> None:
        prob = _click_prob(0.1, 1.0)
        assert prob < 0.3

    def test_temperature_near_zero_approaches_half(self) -> None:
        # logit * ~0 → 0 + noise; mean of sigmoid(noise) ≈ 0.5 over many samples
        probs = [_click_prob(0.9, 0.01) for _ in range(200)]
        assert 0.3 < sum(probs) / len(probs) < 0.7

    def test_output_always_in_unit_interval(self) -> None:
        cases = [
            (0.01, 0.1),
            (0.5, 1.0),
            (0.99, 3.0),
            (0.3, 0.5),
        ]
        for score, temp in cases:
            p = _click_prob(score, temp)
            assert 0.0 <= p <= 1.0, f"out of [0,1] for score={score}, temp={temp}"

    def test_higher_temperature_amplifies_high_score(self) -> None:
        # With many samples the mean probability should be higher at temp=3 than temp=0.5
        n = 300
        low_temp = sum(_click_prob(0.85, 0.5) for _ in range(n)) / n
        high_temp = sum(_click_prob(0.85, 3.0) for _ in range(n)) / n
        assert high_temp > low_temp


class TestBuildEvent:
    def test_all_required_fields_present(self) -> None:
        session = uuid.uuid4()
        ev = _build_event(42, 318, "click", session, 1)
        for field in (
            "event_id",
            "timestamp",
            "user_id",
            "movie_id",
            "event_type",
            "session_id",
            "label",
        ):
            assert field in ev

    def test_click_has_label_one(self) -> None:
        ev = _build_event(1, 1, "click", uuid.uuid4(), 1)
        assert ev["label"] == 1
        assert ev["event_type"] == "click"

    def test_impression_has_label_zero(self) -> None:
        ev = _build_event(1, 1, "impression", uuid.uuid4(), 0)
        assert ev["label"] == 0
        assert ev["event_type"] == "impression"


class TestSampleLifetime:
    def test_zero_churn_fraction_always_returns_none(self) -> None:
        for _ in range(50):
            assert _sample_lifetime(0.0, 7.0) is None

    def test_full_churn_fraction_always_returns_float(self) -> None:
        for _ in range(50):
            result = _sample_lifetime(1.0, 7.0)
            assert isinstance(result, float) and result > 0

    def test_churn_fraction_half_splits_roughly_evenly(self) -> None:
        results = [_sample_lifetime(0.5, 7.0) for _ in range(400)]
        n_none = sum(1 for r in results if r is None)
        # 50% ± 10% tolerance
        assert 160 < n_none < 240
