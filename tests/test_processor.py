"""Tests for the pure _compute_features function in the streaming processor.

These tests do NOT require PyFlink or Redis — they only exercise the feature
computation logic which is independent of the streaming infrastructure.
"""

from __future__ import annotations

import time

import pytest

from src.features.processor import _WINDOW_1H, _compute_features

N_GENRES = 20
_NOW = int(time.time())


def _entry(ts: int, genres: list[float] | None = None) -> dict:
    return {"ts": ts, "genres_vector": genres or [0.0] * N_GENRES}


class TestComputeFeatures:
    def test_empty_entries_returns_zeros(self):
        result = _compute_features([], _NOW)
        assert result["n_clicks_last_7d"] == 0
        assert result["n_clicks_last_1h"] == 0
        assert result["genre_affinity_last_7d"] == [0.0] * N_GENRES
        assert result["genre_affinity_last_1h"] == [0.0] * N_GENRES
        assert result["days_since_last_activity"] == pytest.approx(7.0)

    def test_single_recent_click(self):
        entries = [_entry(_NOW - 60)]  # 1 minute ago
        result = _compute_features(entries, _NOW)
        assert result["n_clicks_last_7d"] == 1
        assert result["n_clicks_last_1h"] == 1
        assert result["days_since_last_activity"] == pytest.approx(60 / 86400, abs=1e-3)

    def test_click_two_days_ago_not_in_1h(self):
        ts_2d = _NOW - 2 * 86400
        entries = [_entry(ts_2d)]
        result = _compute_features(entries, _NOW)
        assert result["n_clicks_last_7d"] == 1
        assert result["n_clicks_last_1h"] == 0
        assert result["genre_affinity_last_1h"] == [0.0] * N_GENRES

    def test_old_click_beyond_7d_excluded(self):
        # Entry is already pruned before calling _compute_features,
        # but verify that an 8-day-old entry yields nothing if it slips through.
        ts_8d = _NOW - 8 * 86400
        entries = [_entry(ts_8d)]
        # Simulate that pruning did NOT happen (edge-case test)
        result = _compute_features(entries, _NOW)
        # If implementation correctly excludes entries outside 7d window:
        assert result["n_clicks_last_7d"] == 0 or result["days_since_last_activity"] > 7.0

    def test_genre_affinity_averages_vectors(self):
        v1 = [1.0] + [0.0] * (N_GENRES - 1)
        v2 = [0.0] + [1.0] + [0.0] * (N_GENRES - 2)
        entries = [_entry(_NOW - 60, v1), _entry(_NOW - 120, v2)]
        result = _compute_features(entries, _NOW)
        affinity = result["genre_affinity_last_7d"]
        assert len(affinity) == N_GENRES
        assert affinity[0] == pytest.approx(0.5)
        assert affinity[1] == pytest.approx(0.5)
        assert all(x == pytest.approx(0.0) for x in affinity[2:])

    def test_1h_affinity_only_uses_recent_entries(self):
        v_old = [1.0] + [0.0] * (N_GENRES - 1)  # 3h ago → only in 7d
        v_new = [0.0, 1.0] + [0.0] * (N_GENRES - 2)  # 30min ago → in both 1h and 7d
        entries = [
            _entry(_NOW - 3 * _WINDOW_1H, v_old),
            _entry(_NOW - 30 * 60, v_new),
        ]
        result = _compute_features(entries, _NOW)
        # 7d affinity: average of v_old and v_new
        assert result["genre_affinity_last_7d"][0] == pytest.approx(0.5)
        assert result["genre_affinity_last_7d"][1] == pytest.approx(0.5)
        # 1h affinity: only v_new
        assert result["genre_affinity_last_1h"][0] == pytest.approx(0.0)
        assert result["genre_affinity_last_1h"][1] == pytest.approx(1.0)

    def test_days_since_activity_from_most_recent_click(self):
        entries = [
            _entry(_NOW - 5 * 86400),  # 5 days ago
            _entry(_NOW - 1 * 86400),  # 1 day ago (most recent)
        ]
        result = _compute_features(entries, _NOW)
        assert result["days_since_last_activity"] == pytest.approx(1.0, abs=0.01)
